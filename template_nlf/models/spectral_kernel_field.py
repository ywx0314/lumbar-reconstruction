# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Fixed cotangent-LBO spectral spectral_encoding and shared dynamic kernel MLP."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from models.runtime import MODEL_SETTINGS

class SpectralKernelField(nn.Module):
    def __init__(self, vertebra_level):
        super().__init__()
        canonical = Path(MODEL_SETTINGS.data_root) / 'canonical'
        eigenvalues = np.load(canonical / f'{vertebra_level}_eigval3.npy', allow_pickle=False)
        eigenvectors = np.load(canonical / f'{vertebra_level}_eigvec3.npy', allow_pickle=False)
        num_spectral_modes = MODEL_SETTINGS.spectral_modes
        if eigenvalues.ndim != 1 or len(eigenvalues) < num_spectral_modes + 1 or eigenvectors.shape != (MODEL_SETTINGS.vertebra_point_counts[0], len(eigenvalues)):
            raise ValueError('Expected matching eigenvalues/eigenvectors including constant mode, in template vertex order.')
        if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)) or np.any(eigenvalues[1:num_spectral_modes+1] <= 0):
            raise ValueError('Spectral spectral_encoding must be finite with positive nonconstant eigenvalues.')
        spectral_encoding = 0.1 * eigenvectors[:, 1:num_spectral_modes+1] / np.sqrt(eigenvalues[None, 1:num_spectral_modes+1])
        self.register_buffer('fixed_posenc', torch.tensor(spectral_encoding, dtype=torch.float32))
        layer_dimensions = [num_spectral_modes] + [MODEL_SETTINGS.kernel_mlp_hidden_dim] * MODEL_SETTINGS.kernel_mlp_hidden_layers + [(MODEL_SETTINGS.feature_channels + 1) * MODEL_SETTINGS.depth_bins]
        layers = []
        for index, (a, b) in enumerate(zip(layer_dimensions[:-1], layer_dimensions[1:])):
            layers.append(nn.Linear(a, b))
            if index < len(layer_dimensions) - 2:
                layers.append(nn.GELU())
        self.pred_mlp = nn.Sequential(*layers)

    def forward(self, num_points, batch_size=1, device=None):
        if num_points != len(self.fixed_posenc):
            raise ValueError('Queries must include every template vertex in the original order.')
        return self.pred_mlp(self.fixed_posenc)

def build_spectral_kernel_field(vertebrae_names, **kwargs):
    if len(vertebrae_names) != 1:
        raise ValueError('Train a separate model for each lumbar level.')
    return SpectralKernelField(vertebrae_names[0])
