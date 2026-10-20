# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import torch
from typing import Optional


def heatmap_to_image(coords: torch.Tensor, image_side: int, stride: int, centered_stride: bool):
    # stride = MODEL_SETTINGS.stride_train if is_training else MODEL_SETTINGS.stride_test

    last_image_pixel = image_side - 1
    last_receptive_center = last_image_pixel - (last_image_pixel % stride)
    # 允许坐标映射到最后一个感受野之外一个 stride，用于覆盖图像外的 GT
    coord_max = last_receptive_center + stride
    coords_out = coords * coord_max

    if centered_stride:
        coords_out = coords_out + stride // 2

    return coords_out


def heatmap_to_metric(
    coords: torch.Tensor, image_side: int, stride: int, centered_stride: bool, projection_box_size_mm: float
):
    # Map normalized expectations to [-50, 50] mm per axis around the centroid.
    # projection_box_size_mm is a separate 2D loss scale, not this offset range.
    coords3d = (coords - 0.5) * 100
    return coords3d
