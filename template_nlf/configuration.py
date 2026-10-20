# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Read repository-relative TOML configuration; one model per lumbar level."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

ROOT = Path(__file__).resolve().parent

LEGACY_SETTING_NAMES = {
    'vertebra': 'vertebra_level', 'data_root_vertebrae': 'data_root',
    'proc_side': 'image_side', 'depth': 'depth_bins',
    'box_size_m': 'projection_box_size_mm',
    'backbone_link_dim': 'feature_channels',
    'mix_3d_inside_fov': 'inference_fusion_weight',
    'field_hidden_size': 'kernel_mlp_hidden_dim',
    'field_hidden_layers': 'kernel_mlp_hidden_layers',
    'field_posenc_dim': 'spectral_modes',
    'lbo_eigval_regul': 'kernel_regularization_weight',
    'absloss_factor': 'spatial_loss_weight',
    'loss_factor_vertebrae': 'surface_loss_weight',
}

def load_reconstruction_config(path):
    with Path(path).open('rb') as stream:
        settings = tomllib.load(stream)
    for old_name, new_name in LEGACY_SETTING_NAMES.items():
        if old_name in settings:
            if new_name in settings and settings[new_name] != settings[old_name]:
                raise ValueError(f'Conflicting configuration values: {old_name}, {new_name}')
            settings[new_name] = settings.pop(old_name)
    cfg = SimpleNamespace(**settings)
    for key in ('data_root', 'output_dir'):
        setattr(cfg, key, str((ROOT / getattr(cfg, key)).resolve()))
    for key in ('id_list', 'checkpoint', 'images_dir', 'camera_dir', 'gt_dir'):
        if getattr(cfg, key, ''):
            setattr(cfg, key, str((ROOT / getattr(cfg, key)).resolve()))
    if cfg.vertebra_level not in ('L1', 'L2', 'L3', 'L4', 'L5'):
        raise ValueError('vertebra_level must be L1--L5')
    return cfg

def compute_learning_rate(epoch, *, epochs=300, base_lr=2.121e-4):
    """10 epoch linear warmup, main decay to 1/3, final 20 epoch decay."""
    if epochs <= 30:
        raise ValueError('The three-stage schedule requires more than 30 epochs.')
    if epoch < 10:
        return base_lr * (epoch + 1) / 10
    if epoch < epochs - 20:
        return base_lr * (1 / 3) ** ((epoch - 10) / (epochs - 30))
    return base_lr / 30 * 0.3 ** ((epoch - (epochs - 20)) / 20)
