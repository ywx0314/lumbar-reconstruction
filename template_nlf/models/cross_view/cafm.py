# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
# Image-encoder design reference: https://github.com/ShilongHong/DAGNet
# Project copyright and SPDX notice cover project-owned contributions only.
# Architecture reference only; DAGNet source code is not bundled.
# See THIRD_PARTY_NOTICES.md for the architectural acknowledgment.
import torch
import torch.nn as nn
from .convolution import ConvNormLayer, DWConv
from .cbam import CBAM


class CSPRepLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, expansion=1.0, bias=None, act="silu"):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = DWConv(in_channels, hidden_channels)
        self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act) if hidden_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.conv3(self.bottlenecks(x) + self.conv2(x))


class CAFM(nn.Module):
    def __init__(self, int_c, out_c, act="silu", expansion=1):
        super().__init__()
        self.attn = CBAM(int_c // 2)
        self.csp = CSPRepLayer(int_c, out_c, 1, act=act, expansion=expansion)

    def forward(self, h_x, l_x):
        h_x = self.attn(h_x)
        l_x = self.attn(l_x)
        return self.csp(torch.cat((h_x, l_x), dim=1).contiguous())
