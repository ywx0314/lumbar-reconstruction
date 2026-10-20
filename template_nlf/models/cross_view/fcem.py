# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
# Image-encoder design reference: https://github.com/ShilongHong/DAGNet
# Project copyright and SPDX notice cover project-owned contributions only.
# Architecture reference only; DAGNet source code is not bundled.
# See THIRD_PARTY_NOTICES.md for the architectural acknowledgment.
import torch
import torch.nn as nn
def to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0, scale_learnable=True, bias_learnable=True, mode=None, inplace=False):
        super().__init__()
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1), requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1), requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.0, bias=False, **kwargs):
        super().__init__()
        out_features = out_features or dim
        hidden_features = int(mlp_ratio * dim)
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(dim, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class DynamicFilterV2(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=0.25, act1_layer=StarReLU, act2_layer=nn.Identity, bias=False, num_filters=4, size=14, weight_resize=False, **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.med_channels = int(expansion_ratio * dim)
        self.pwconv0 = nn.Linear(dim, dim, bias=bias)
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(torch.randn(self.size, self.filter_size, num_filters, 2, dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def _apply_frequency_enhancement(self, input_tensor, weight, h, w):
        freq = self.pwconv1(input_tensor)
        freq = self.act1(freq).to(torch.float32)
        freq = torch.fft.rfft2(freq, dim=(1, 2), norm="ortho")
        freq = freq * weight
        freq = torch.fft.irfft2(freq, s=(h, w), dim=(1, 2), norm="ortho")
        freq = self.act2(freq)
        return self.pwconv2(freq).permute(0, 3, 1, 2)

    def forward(self, x, y):
        x = x.permute(0, 2, 3, 1)
        y = y.permute(0, 2, 3, 1)
        b, h, w, _ = x.shape
        x_mean = x.mean(dim=(1, 2))
        y_mean = y.mean(dim=(1, 2))
        x_mean = x_mean * torch.sigmoid(self.pwconv0(y_mean)) + x_mean
        routing_weights = self.reweight(x_mean).view(b, self.num_filters, -1).softmax(dim=1)
        complex_weights = torch.view_as_complex(self.complex_weights)
        weight = torch.einsum("bfc,hwf->bhwc", routing_weights.to(torch.complex64), complex_weights)
        weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        return self._apply_frequency_enhancement(x, weight, h, w)


class FCEM(nn.Module):
    def __init__(self, channel, size=64):
        super().__init__()
        self.freq_enhance = DynamicFilterV2(channel, size=size)

    def forward(self, x, y):
        return self.freq_enhance(x, y) + x, self.freq_enhance(y, x) + y
