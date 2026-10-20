# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Feature-domain classifier and gradient reversal from the Methods."""
import math
import torch
from torch import nn

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient):
        ctx.coefficient = coefficient
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None

def gradient_reversal_coefficient(progress):
    return 2 / (1 + math.exp(-10 * progress)) - 1

class FeatureDomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(256, 256, 3, 2, 1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, 3, 2, 1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, 2)

    def forward(self, features):
        return self.classifier(self.features(features).flatten(1))
