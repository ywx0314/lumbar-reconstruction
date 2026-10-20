# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import torch.nn as nn
from .activations import get_activation


class ConvNormLayer(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in,
            ch_out,
            kernel_size,
            stride,
            padding=(kernel_size - 1) // 2 if padding is None else padding,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DWConv(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size=3, stride=1, padding=None, bias=False, act=None):
        super().__init__()
        padding = (kernel_size - 1) // 2 if padding is None else padding
        self.dconv = nn.Conv2d(ch_in, ch_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=ch_in, bias=bias)
        self.wconv = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act)

    def forward(self, x):
        x = self.dconv(x)
        x = self.wconv(x)
        x = self.norm(x)
        x = self.act(x)
        return x
