# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch


def _as_float32_array(x: np.ndarray, *, expected_last_dim: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.shape[-1] != expected_last_dim:
        raise ValueError(f"参数 shape 异常: {arr.shape}, expected_last_dim={expected_last_dim}")
    return arr


def _load_sample_dual_camera_from_npz(npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    if not npz_path.is_file():
        raise FileNotFoundError(f"找不到样本投影参数: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    keys = set(data.files)

    # 新格式（推荐）：camera1_* / camera2_*
    if {
        "camera1_image_plane_vertices",
        "camera1_light_source",
        "camera2_image_plane_vertices",
        "camera2_light_source",
    }.issubset(keys):
        plane1 = _as_float32_array(data["camera1_image_plane_vertices"], expected_last_dim=3)
        plane2 = _as_float32_array(data["camera2_image_plane_vertices"], expected_last_dim=3)
        light1 = _as_float32_array(data["camera1_light_source"], expected_last_dim=3)
        light2 = _as_float32_array(data["camera2_light_source"], expected_last_dim=3)
    # 兼容格式：直接存 image_plane_vertices / light_source
    elif {"image_plane_vertices", "light_source"}.issubset(keys):
        plane = _as_float32_array(data["image_plane_vertices"], expected_last_dim=3)
        light = _as_float32_array(data["light_source"], expected_last_dim=3)
        if plane.ndim == 3 and plane.shape[0] == 2:
            plane1, plane2 = plane[0], plane[1]
        else:
            raise ValueError(f"{npz_path} image_plane_vertices shape 应为 [2,4,3]，实际 {plane.shape}")
        if light.ndim == 2 and light.shape[0] == 2:
            light1, light2 = light[0], light[1]
        else:
            raise ValueError(f"{npz_path} light_source shape 应为 [2,3]，实际 {light.shape}")
    else:
        raise KeyError(f"{npz_path} 缺少相机参数键，当前 keys={sorted(keys)}")

    if plane1.shape != (4, 3) or plane2.shape != (4, 3):
        raise ValueError(f"{npz_path} image_plane_vertices shape 异常: {plane1.shape}, {plane2.shape}")
    if light1.shape != (3,) or light2.shape != (3,):
        raise ValueError(f"{npz_path} light_source shape 异常: {light1.shape}, {light2.shape}")

    image_plane_vertices = torch.from_numpy(np.stack([plane1, plane2], axis=0))  # [2,4,3]
    light_source = torch.from_numpy(np.stack([light1, light2], axis=0))  # [2,3]
    return image_plane_vertices, light_source


def load_sample_camera_parameters(
    data_root_path: Path,
    *,
    vertebra_level: str,
    sample_ids: Iterable[str],
    params_dir: str | Path | None = None,
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    params_dir = Path(params_dir) if params_dir is not None else (data_root_path / f"{vertebra_level}_proj_params")
    if not params_dir.is_dir():
        raise FileNotFoundError(f"样本相机参数目录不存在: {params_dir}")

    params_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
    for sid in sample_ids:
        npz_path = params_dir / f"{sid}.npz"
        params_map[sid] = _load_sample_dual_camera_from_npz(npz_path)
    return params_map


def build_camera_batch(
    sample_ids: Iterable[str],
    params_map: Dict[str, Tuple[torch.Tensor, torch.Tensor]],
    *,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    planes = []
    lights = []
    for sid in sample_ids:
        if sid not in params_map:
            raise KeyError(f"样本 {sid} 缺少相机参数，请检查对应 npz 是否存在且已加载。")
        plane_i, light_i = params_map[sid]
        planes.append(plane_i)
        lights.append(light_i)
    image_plane_vertices = torch.stack(planes, dim=0).to(device=device, non_blocking=True)  # [B,2,4,3]
    light_source = torch.stack(lights, dim=0).to(device=device, non_blocking=True)  # [B,2,3]
    return image_plane_vertices, light_source
