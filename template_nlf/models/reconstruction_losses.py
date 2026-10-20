# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
from __future__ import annotations

from typing import Tuple

import torch

from models import tensor_ops, cone_beam_geometry


def compute_spatial_vertex_loss(
    verts_gt: torch.Tensor, verts_pred: torch.Tensor, *, epsilon: float = 2e-2
) -> torch.Tensor:
    """
    Absolute 3D loss from the manuscript:
    mean(charbonnier(diff, epsilon=2e-2, dim=-1))
    """

    diff = verts_gt - verts_pred
    return torch.mean(tensor_ops.charbonnier(diff, epsilon=epsilon, dim=-1))


def compute_projection_loss(
    verts_pred: torch.Tensor,
    verts_gt: torch.Tensor,
    image_plane_vertices: torch.Tensor,
    light_source: torch.Tensor,
    *,
    image_side: int,
    projection_box_size_mm: float,
    epsilon: float = 2e-2,
) -> torch.Tensor:
    """
    Dual-view projected surface loss:
    - 两个视角投影到像素坐标
    - mean(charbonnier((coords2d_true - coords2d_pred) * scale_2d, epsilon=2e-2, dim=-1))
    - 两视角平均
    """

    # scale_2d 同 nlf_trainer：将像素差映射到毫米
    scale_2d = 1.0 / float(image_side) * float(projection_box_size_mm)

    loss2d_list = []
    image_size: Tuple[int, int] = (image_side, image_side)

    for cam_idx in range(2):
        vertices_cam = image_plane_vertices[:, cam_idx, :, :]  # [B, 4, 3]
        light_cam = light_source[:, cam_idx, :]  # [B, 3]

        coords2d_pred_cam = cone_beam_geometry.project_cone_beam_vectorized(
            coords3d=verts_pred,
            image_plane_vertices=vertices_cam,
            light_source=light_cam,
            image_size=image_size,
        )  # [B, N, 2]
        coords2d_true_cam = cone_beam_geometry.project_cone_beam_vectorized(
            coords3d=verts_gt,
            image_plane_vertices=vertices_cam,
            light_source=light_cam,
            image_size=image_size,
        )  # [B, N, 2]

        loss2d_cam = torch.mean(
            tensor_ops.charbonnier(
                (coords2d_true_cam - coords2d_pred_cam) * scale_2d,
                epsilon=epsilon,
                dim=-1,
            )
        )
        loss2d_list.append(loss2d_cam)

    return torch.stack(loss2d_list, dim=0).mean(dim=0)
