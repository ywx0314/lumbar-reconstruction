# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import tensor_ops
from models import coordinate_decoding
from models.runtime import MODEL_SETTINGS


class LocalizationFieldBase(nn.Module):
    def __init__(self, backbone, weight_field, normalizer, backbone_channels=1536):
        super().__init__()
        self.backbone = backbone
        # 默认情况下，2D 特征是单视角通道 (768)，3D 是融合通道 (1536)
        in_channels_2d = getattr(backbone, 'single_view_channels', 768)
        in_channels_3d = backbone_channels
        self.heatmap_head = DynamicLocalizationHead(
            weight_field, normalizer, 
            in_channels_2d=in_channels_2d, 
            in_channels_3d=in_channels_3d
        )
        self.input_resolution = MODEL_SETTINGS.image_side


class DynamicLocalizationHead(nn.Module):
    def __init__(self, weight_field, normalizer, in_channels_2d=768, in_channels_3d=1536):
        super().__init__()
        self.depth_bins = MODEL_SETTINGS.depth_bins
        self.weight_field = weight_field
        self.stride_test = MODEL_SETTINGS.stride_test
        self.centered_stride = MODEL_SETTINGS.centered_stride
        self.projection_box_size_mm = MODEL_SETTINGS.projection_box_size_mm
        self.image_side = MODEL_SETTINGS.image_side
        self.feature_channels = MODEL_SETTINGS.feature_channels
        
        # 当输入通道已与 feature_channels 一致时，跳过 1x1 映射，避免无效整形。
        if in_channels_2d == self.feature_channels:
            self.layer_2d = nn.Identity()
        else:
            self.layer_2d = nn.Sequential(
                nn.Conv2d(in_channels_2d, self.feature_channels, kernel_size=1, bias=False),
                normalizer(self.feature_channels),
                nn.SiLU(),
            )

        if in_channels_3d == self.feature_channels:
            self.layer_3d = nn.Identity()
        else:
            self.layer_3d = nn.Sequential(
                nn.Conv2d(in_channels_3d, self.feature_channels, kernel_size=1, bias=False),
                normalizer(self.feature_channels),
                nn.SiLU(),
            )

    @torch.jit.export
    def forward(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        num_points: int,
        vertebra_point_counts: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            features_2d: backbone 2D 特征 (N, 2, C2d, H, W)
            features_3d: backbone 3D 融合特征 (N, C3d, H, W)
            num_points: 查询点的数量
            vertebra_point_counts: 每个椎体的点数（用于 MultiVertebraeField）
        """
        batch_size = features_2d.shape[0]
        device = features_2d.device
        
        # 权重场（SpectralKernelField/MultiVertebraeField）在本项目里与 batch 无关：
        # 本质上只是从固定 canonical embedding 生成 (P, D) 的 weights。
        # 让它按 batch_size 复制会造成 (N, P, D) 的巨大显存占用（例如 N=8, P=10000, D=8208）。
        # 因此这里强制 batch_size=1，只计算一次，然后复用到整个 batch。
        weights_batch_size = 1
        
        weights = self.weight_field(num_points, batch_size=1, device=device)

        if weights.dim() == 3:
            weights = weights[0]  # (P, (c+1)*C)所有查询点相同
        
        return self.call_with_weights(features_2d, features_3d, weights)

    @torch.jit.export
    def call_with_weights(self, features_2d, features_3d, weights):
        # features_2d shape: [N, 2, C2d, H, W]
        # features_3d shape: [N, C3d, H, W]
        
        # 分别处理两个视角的 2D 特征
        batch_size, num_views, c_in, h, w = features_2d.shape
        features_2d_reshaped = features_2d.reshape(batch_size * num_views, c_in, h, w)
        features_2d_processed = self.layer_2d(features_2d_reshaped)  # [N*2, C_link, H, W]
        features_2d_processed = features_2d_processed.reshape(batch_size, num_views, -1, h, w) # [N, 2, C_link, H, W]
        
        # 处理融合的 3D 特征
        features_3d_processed = self.layer_3d(features_3d)  # [N, C_link, H, W]

        coords2d, coords3d_rel_pred = self.apply_weights3d(
            features_2d_processed,
            features_3d_processed,
            weights,
            n_out_channels=self.depth_bins,
        )
        coords2d_pred = coordinate_decoding.heatmap_to_image(
            coords2d, self.image_side, self.stride_test, self.centered_stride
        )
        coords3d_rel_pred = coordinate_decoding.heatmap_to_metric(
            coords3d_rel_pred,
            self.image_side,
            self.stride_test,
            self.centered_stride,
            self.projection_box_size_mm,
        )
        return coords2d_pred, coords3d_rel_pred


    @torch.jit.export
    def apply_weights3d(
        self,
        features_2d: torch.Tensor,
        features_3d: torch.Tensor,
        weights: torch.Tensor,
        n_out_channels: int,
    ):
        # features_2d: [N, 2, C_link, H, W] - 双视角特征，分别处理
        # features_3d: [N, C_link, H, W] - 融合特征，用于 3D 分支
        # weights: [P, (C_link+1)*C_out] - 每个查询点的权重
        
        # 1. 准备权重 (用于 conv2d 优化)
        # 在转换权重类型前，如果可能的话清理内存碎片
        device = features_3d.device
        
        w_tensor, b_tensor = self.transpose_weights(
            weights.to(features_3d.dtype), self.feature_channels
        )
        
        # 2. 解码 3D 坐标 (仅使用融合特征 features_3d)
        # F.conv2d 会创建大输出张量，在创建前清理内存
        logits_3d_full = F.conv2d(features_3d, w_tensor, bias=b_tensor).float()
        logits_3d_full = torch.unflatten(logits_3d_full, 1, (-1, n_out_channels))  # (N, P, C_out, H, W)
        
        # 提取 3D logits 并转换为 heatmap (现在所有通道都是 logits)
        logits_3d = logits_3d_full  # (N, P, depth_bins, H, W)
        heatmap3d = tensor_ops.softmax(logits_3d, dim=[4, 3, 2])  # (N, P, depth_bins, H, W)
        
        # 在解码前删除不再需要的 logits（heatmap3d 已经包含了所需信息）
        del logits_3d_full, logits_3d
        
        # 解码为 3D 相对坐标
        coords3d = tensor_ops.decode_heatmap(heatmap3d, dim=[4, 3, 2])  # (N, P, 3) [x, y, z]
        
        # 解码完成后删除 heatmap（coords3d 已经包含了最终结果）
        del heatmap3d
        
        # 3. 解码双视角的 2D 坐标 (分别使用各自视角的特征)
        N, V, C, H, W = features_2d.shape
        features_2d_flat = features_2d.reshape(N * V, C, H, W)
        
        # 在计算 2D logits 前清理内存（3D logits 可能已经占用大量内存）
        # 一次性计算两个视角的 logits
        logits_2d_full = F.conv2d(features_2d_flat, w_tensor, bias=b_tensor).float()
        logits_2d_full = torch.unflatten(logits_2d_full, 1, (-1, n_out_channels))  # (N*2, P, C_out, H, W)
        
        # Original decoder: sum depth_bins logits before spatial softmax.
        logits_2d = torch.sum(logits_2d_full, dim=2) # (N*2, P, H, W)
        # 删除不再需要的大张量
        del logits_2d_full
        heatmap2d = tensor_ops.softmax(logits_2d, dim=[3, 2]) # (N*2, P, H, W)
        del logits_2d
        coords2d = tensor_ops.decode_heatmap2d(heatmap2d, num_views=V)  # [N, 2, P, 2]
        del heatmap2d
        
        return coords2d, coords3d

    @torch.jit.export
    def transpose_weights(self, weights: torch.Tensor, n_in_channels: int):
        n_out_channels = self.depth_bins
        # 确保 weights 是 2 维的 (P, (c+1)*C_out)
        if weights.dim() == 3:
            weights = weights[0]
        
        P = weights.shape[0]
        total_dim = n_in_channels + 1
        
        # 使用 view 重塑权重矩阵（view 不分配新内存，只是改变视图）
        weights_reshaped = weights.view(P, total_dim, n_out_channels)  # (P, c+1, C_out)
        
        # 提取权重部分 (P, c, C_out) 和偏置部分 (P, C_out)
        w_tensor = weights_reshaped[:, :-1, :]  # (P, c, C_out)
        b_tensor = weights_reshaped[:, -1, :]   # (P, C_out)
        
        # 将权重转置并展平，准备用于 1x1 conv2d
        # 目标形状: (P*C_out, c, 1, 1)
        # 使用 permute 然后 reshape，这样更高效
        w_tensor = w_tensor.permute(0, 2, 1).reshape(-1, n_in_channels, 1, 1)
        b_tensor = b_tensor.reshape(-1)
        
        return w_tensor.contiguous(), b_tensor.contiguous()
