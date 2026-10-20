# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
# Image-encoder design reference: https://github.com/ShilongHong/DAGNet
# Project copyright and SPDX notice cover project-owned contributions only.
# Architecture reference only; DAGNet source code is not bundled.
# See THIRD_PARTY_NOTICES.md for the architectural acknowledgment.
import torch
import torch.nn as nn

from models.image_encoder import CrossViewImageEncoder


class BiplanarImageEncoder(nn.Module):
    """
    使用 CrossViewImageEncoder 的双平面 backbone
    输入:
        image: [B, 2, H, W]  (两个正交视图)
    输出:
        features: [B, C, 16, 16]
        其中 C 由编码器配置决定（当前默认单视角 512，融合 512）
    """

    def __init__(self):
        super().__init__()
        self.encoder = CrossViewImageEncoder()

        self.single_view_channels = int(self.encoder.single_view_out_channels)
        self.out_channels = int(self.encoder.fused_out_channels)

    def forward(self, image: torch.Tensor):
        """
        Args:
            image: [B, 2, H, W]，其中:
                image[:, 0] 为第一个视角 (eg. AP)
                image[:, 1] 为第二个视角 (eg. ML)

        Returns:
            features_2d: [B, 2, C2d, 16, 16]，两个平面特征 (未拼接):
                features_2d[:, 0] -> 视角1特征
                features_2d[:, 1] -> 视角2特征
            features_3d: [B, C3d, 16, 16]，由 CrossViewImageEncoder 的 fused_3d_features 得到，
                专供 3D 分支使用
        """
        if image.dim() != 4:
            raise ValueError(f"Expected image with 4 dims [B, 2, H, W], got shape {image.shape}")
        if image.shape[1] != 2:
            raise ValueError(
                f"BiplanarImageEncoder expects 2-channel input (two views), got {image.shape[1]} channels"
            )

        # 按视角拆分，并增加通道维到 1
        ap_image = image[:, 0:1, :, :]  # [B, 1, H, W]
        lateral_image = image[:, 1:2, :, :]  # [B, 1, H, W]

        outputs = self.encoder(ap_image, lateral_image)

        view_features = outputs["view_features"]
        stage4_ap_features = view_features["stage4_ap_features"]  # [B, C2d, 16, 16]
        stage4_lateral_features = view_features["stage4_lateral_features"]  # [B, C2d, 16, 16]
        fused_3d_features = outputs["fused_3d_features"]  # [B, C3d, 16, 16]

        if stage4_ap_features.shape[2:] != stage4_lateral_features.shape[2:]:
            raise ValueError(
                f"Dual-plane features spatial shape mismatch: "
                f"stage4_ap_features {stage4_ap_features.shape}, stage4_lateral_features {stage4_lateral_features.shape}"
            )
        if stage4_ap_features.shape[1] != self.single_view_channels or stage4_lateral_features.shape[1] != self.single_view_channels:
            raise ValueError(
                f"Single-view channels mismatch: expect {self.single_view_channels}, "
                f"got stage4_ap_features {stage4_ap_features.shape}, stage4_lateral_features {stage4_lateral_features.shape}"
            )
        if fused_3d_features.shape[1] != self.out_channels:
            raise ValueError(
                f"Fused channels mismatch: expect {self.out_channels}, got {fused_3d_features.shape}"
            )

        # 堆叠两个视角的特征，不再拼接，得到 [B, 2, C2d, 16, 16]
        features_2d = torch.stack([stage4_ap_features, stage4_lateral_features], dim=1)

        # 直接使用 fused_3d_features 作为 3D 分支输入
        features_3d = fused_3d_features

        return features_2d, features_3d


