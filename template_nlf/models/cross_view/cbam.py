# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels),
        )

    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        max_out = self.mlp(F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))).unsqueeze(-1).unsqueeze(-1)
        scale = torch.sigmoid(avg_out + max_out).expand_as(x)
        return x * scale


class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x_compress = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        scale = torch.sigmoid(self.spatial(x_compress))
        return x * scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.channel_gate = ChannelGate(gate_channels, reduction_ratio)
        self.spatial_gate = SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x
