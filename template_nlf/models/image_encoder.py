# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
# Image-encoder design reference: https://github.com/ShilongHong/DAGNet
# Project copyright and SPDX notice cover project-owned contributions only.
# Architecture reference only; DAGNet source code is not bundled.
# See THIRD_PARTY_NOTICES.md for the architectural acknowledgment.
import torch
from torch import nn

from models.cross_view.fcem import FCEM
from models.cross_view.dvca import DVCA
from models.cross_view.cafm import CAFM


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, kernel_size=3, stride=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class DownsampleStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, blocks: int = 2):
        super().__init__()
        layers = [ConvBNAct(in_ch, out_ch, kernel_size=3, stride=2)]
        layers.extend(ResidualBlock(out_ch) for _ in range(max(blocks - 1, 0)))
        self.stage = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage(x)


class CrossViewImageEncoder(nn.Module):
    def __init__(self, depth_mult=3, expansion=1 / 4, act="silu", backbone="custom_cnn"):
        super().__init__()
        if backbone not in {"custom_cnn", "convnext"}:
            raise ValueError(f"不支持的 backbone: {backbone}")

        output_channels = [64, 128, 256, 512]
        self.output_channels = output_channels
        self.single_view_out_channels = output_channels[-1]
        self.fused_out_channels = output_channels[-1]

        # 512 -> 128（stride=4）
        self.stem = ConvBNAct(1, output_channels[0], kernel_size=7, stride=4)
        self.layer1 = ResidualBlock(output_channels[0])
        self.layer2 = DownsampleStage(output_channels[0], output_channels[1], blocks=2)
        self.layer3 = DownsampleStage(output_channels[1], output_channels[2], blocks=2)
        self.layer4 = DownsampleStage(output_channels[2], output_channels[3], blocks=2)

        self.fdim = FCEM(output_channels[0], size=128)
        self.cfe2 = DVCA(output_channels[1], output_channels[2])
        self.cfe3 = DVCA(output_channels[2], output_channels[3])
        self.cfe4 = DVCA(output_channels[3])
        # 2*512 -> 512
        self.cgfm = CAFM(output_channels[3] * 2, self.fused_out_channels, expansion=1.0)

    # Serialized module names are retained for existing checkpoint compatibility.
    @property
    def fcem(self):
        return self.fdim

    @property
    def dvca_stage2(self):
        return self.cfe2

    @property
    def dvca_stage3(self):
        return self.cfe3

    @property
    def dvca_stage4(self):
        return self.cfe4

    @property
    def cafm(self):
        return self.cgfm

    def forward(self, ap_image, lateral_image):
        ap_stem_features = self.stem(ap_image)
        lateral_stem_features = self.stem(lateral_image)
        ap_spatial_gate = lateral_spatial_gate = 1
        stage1_ap_features = self.layer1(ap_stem_features)
        stage1_lateral_features = self.layer1(lateral_stem_features)
        stage1_ap_features, stage1_lateral_features = self.fcem(stage1_ap_features, stage1_lateral_features)
        stage2_ap_features = self.layer2(stage1_ap_features)
        stage2_lateral_features = self.layer2(stage1_lateral_features)
        stage2_ap_features, stage2_lateral_features, ap_spatial_gate, lateral_spatial_gate = self.dvca_stage2(stage2_ap_features * ap_spatial_gate, stage2_lateral_features * lateral_spatial_gate)
        stage3_ap_features = self.layer3(stage2_ap_features)
        stage3_lateral_features = self.layer3(stage2_lateral_features)
        stage3_ap_features, stage3_lateral_features, ap_spatial_gate, lateral_spatial_gate = self.dvca_stage3(stage3_ap_features * ap_spatial_gate, stage3_lateral_features * lateral_spatial_gate)
        stage4_ap_features = self.layer4(stage3_ap_features)
        stage4_lateral_features = self.layer4(stage3_lateral_features)
        stage4_ap_features, stage4_lateral_features, _, _ = self.dvca_stage4(stage4_ap_features * ap_spatial_gate, stage4_lateral_features * lateral_spatial_gate)
        fused_3d_features = self.cafm(stage4_ap_features, stage4_lateral_features)
        return {
            "view_features": {"stage4_ap_features": stage4_ap_features, "stage4_lateral_features": stage4_lateral_features},
            "fused_3d_features": fused_3d_features,
        }
