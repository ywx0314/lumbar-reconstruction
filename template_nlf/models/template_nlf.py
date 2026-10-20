# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models.localization_head import LocalizationFieldBase
from models import spectral_kernel_field
from models.biplanar_encoder import BiplanarImageEncoder
from models import cone_beam_geometry
from models.runtime import MODEL_SETTINGS


def configure_localization_field(cfg: Any) -> None:
    """Configure the single-level localization field."""
    settings = MODEL_SETTINGS
    settings.data_root = str(cfg.data_root)
    settings.image_side = int(cfg.image_side)
    settings.stride_train = int(cfg.stride_train)
    settings.stride_test = int(cfg.stride_test)
    settings.centered_stride = bool(cfg.centered_stride)
    settings.dtype = str(cfg.dtype)
    settings.depth_bins = int(getattr(cfg, "depth_bins", 16))
    settings.projection_box_size_mm = float(getattr(cfg, "projection_box_size_mm", 427.0))
    settings.feature_channels = int(getattr(cfg, "feature_channels", 512))
    settings.inference_fusion_weight = float(getattr(cfg, "inference_fusion_weight", 0.5))
    settings.kernel_mlp_hidden_dim = int(getattr(cfg, "kernel_mlp_hidden_dim", 384))
    settings.kernel_mlp_hidden_layers = int(getattr(cfg, "kernel_mlp_hidden_layers", 1))
    settings.spectral_modes = int(getattr(cfg, "spectral_modes", 1024))
    settings.kernel_regularization_weight = float(getattr(cfg, "kernel_regularization_weight", 0.01))
    settings.spatial_loss_weight = float(getattr(cfg, "spatial_loss_weight", 1.0))
    settings.absloss_start_step = int(getattr(cfg, "absloss_start_step", 1))
    settings.surface_loss_weight = float(getattr(cfg, "surface_loss_weight", 1.0))



def compute_kernel_weight_regularization(weight_field: nn.Module) -> torch.Tensor:
    """Mean column-wise squared norm of the first MLP weight matrix."""
    w = weight_field.pred_mlp[0].weight
    return w.square().sum(dim=0).mean()


class TemplateNeuralLocalizationField(LocalizationFieldBase):
    """Shared dual-view encoder, fixed template field and geometric decoder."""

    def __init__(self, cfg: Any, nodes: np.ndarray):
        configure_localization_field(cfg)
        self.num_vertices = int(nodes.shape[0])
        MODEL_SETTINGS.vertebra_point_counts = [self.num_vertices]

        backbone = BiplanarImageEncoder()
        weight_field = spectral_kernel_field.build_spectral_kernel_field(
            vertebrae_names=[str(cfg.vertebra_level)],
        )
        super().__init__(backbone, weight_field, nn.BatchNorm2d, backbone_channels=backbone.out_channels)

    def _fuse_geometric_and_spatial_vertices(
        self,
        coords2d_view1: torch.Tensor,
        coords2d_view2: torch.Tensor,
        relative_vertex_coordinates: torch.Tensor,
        image_plane_vertices1: torch.Tensor,
        image_plane_vertices2: torch.Tensor,
        light_source1: torch.Tensor,
        light_source2: torch.Tensor,
        fusion_weights: torch.Tensor,
    ) -> torch.Tensor:
        device = coords2d_view1.device
        image_plane_vertices1 = image_plane_vertices1.to(device=device)
        image_plane_vertices2 = image_plane_vertices2.to(device=device)
        light_source1 = light_source1.to(device=device)
        light_source2 = light_source2.to(device=device)

        triangulated_vertices = cone_beam_geometry.triangulate_from_two_views(
            coords2d_view1,
            coords2d_view2,
            image_plane_vertices1,
            image_plane_vertices2,
            light_source1,
            light_source2,
            (MODEL_SETTINGS.image_side, MODEL_SETTINGS.image_side),
        )

        batch_size = triangulated_vertices.shape[0]
        fusion_weight = fusion_weights.to(device=device, dtype=relative_vertex_coordinates.dtype)
        if fusion_weight.dim() == 0:
            fusion_weight = fusion_weight.view(1, 1, 1).expand(batch_size, 1, 1)
        elif fusion_weight.dim() == 1:
            fusion_weight = fusion_weight.view(-1, 1, 1)

        if torch.any(fusion_weight > 0):
            triangulated_centroid = torch.mean(triangulated_vertices, dim=1, keepdim=True)
            compensated_vertices = relative_vertex_coordinates + triangulated_centroid
            broadcast_fusion_weight = fusion_weight.view(batch_size, 1, 1)
            return (
                broadcast_fusion_weight * compensated_vertices
                + (1 - broadcast_fusion_weight) * triangulated_vertices
            )
        return triangulated_vertices

    def forward(
        self,
        image: torch.Tensor,
        image_plane_vertices: torch.Tensor,
        light_source: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.backbone.parameters()).device
        image = image.to(device=device, dtype=next(self.backbone.parameters()).dtype, non_blocking=True)

        features_2d, features_3d = self.backbone(image)
        vpc = torch.tensor([self.num_vertices], device=device, dtype=torch.long)
        view_coordinates, relative_vertex_coordinates = self.heatmap_head(
            features_2d,
            features_3d,
            self.num_vertices,
            vertebra_point_counts=vpc,
        )

        ap_detector_corners = image_plane_vertices[:, 0, :, :]
        lateral_detector_corners = image_plane_vertices[:, 1, :, :]
        ap_source_position = light_source[:, 0, :]
        lateral_source_position = light_source[:, 1, :]
        ap_coordinates = view_coordinates[:, 0, :, :]
        lateral_coordinates = view_coordinates[:, 1, :, :]

        batch_size = relative_vertex_coordinates.shape[0]
        if self.training:
            fusion_weights = torch.rand([batch_size, 1, 1], device=device, dtype=relative_vertex_coordinates.dtype)
        else:
            inference_weight = float(MODEL_SETTINGS.inference_fusion_weight)
            fusion_weights = torch.full((batch_size, 1, 1), inference_weight, device=device, dtype=relative_vertex_coordinates.dtype)

        return self._fuse_geometric_and_spatial_vertices(
            ap_coordinates, lateral_coordinates, relative_vertex_coordinates, ap_detector_corners, lateral_detector_corners, ap_source_position, lateral_source_position, fusion_weights
        )
