# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_training_id_list(data_root: Path, vertebra_level: str = "L1", seed: int = 42) -> Path:
    imgs_path = data_root / f"{vertebra_level}_imgs"
    stl_path = data_root / f"{vertebra_level}_stl"
    if not imgs_path.exists():
        raise FileNotFoundError(f"图像目录不存在: {imgs_path}")
    if not stl_path.exists():
        raise FileNotFoundError(f"STL 目录不存在: {stl_path}")

    ap_images = list(imgs_path.glob("*_AP.png"))
    sample_ids: set[str] = set()
    for ap in ap_images:
        stem = ap.stem
        if not stem.endswith("_AP"):
            continue
        sid = stem[:-3]
        if (imgs_path / f"{sid}_ML.png").exists() and (stl_path / f"{sid}.npy").exists():
            sample_ids.add(sid)

    if not sample_ids:
        raise RuntimeError(f"未找到有效样本: {imgs_path} / {stl_path}")

    set_random_seed(seed)
    train_txt = data_root / f"{vertebra_level.lower()}_train.txt"
    with open(train_txt, "w") as f:
        for sid in sorted(sample_ids):
            f.write(sid + "\n")
    return train_txt


class SyntheticVertebraDataset(Dataset):
    def __init__(self, data_root: str | Path, id_txt: str | Path, *, image_side: int = 512, vertebra_level: str = "L1"):
        super().__init__()
        self.data_root = Path(data_root)
        self.image_side = int(image_side)
        self.vertebra_level = vertebra_level
        self.imgs_dir = self.data_root / f"{self.vertebra_level}_imgs"
        self.stl_dir = self.data_root / f"{self.vertebra_level}_stl"

        id_txt = Path(id_txt)
        if not id_txt.exists():
            raise FileNotFoundError(f"ID 列表 txt 不存在: {id_txt}")
        with open(id_txt, "r") as f:
            self.sample_ids = [line.strip() for line in f if line.strip()]
        if not self.sample_ids:
            raise RuntimeError(f"{id_txt} 内没有有效的样本 id")

        nodes_path = self.data_root / "canonical" / f"2000{self.vertebra_level}_nodes3.npy"
        if not nodes_path.exists():
            raise FileNotFoundError(f"找不到参考网格 npy: {nodes_path}")
        self.nodes = np.load(nodes_path).astype(np.float32)
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"参考网格形状异常: nodes {self.nodes.shape}")

    def __len__(self) -> int:
        return len(self.sample_ids)

    def _load_image_pair(self, sid: str) -> np.ndarray:
        from PIL import Image

        ap_path = self.imgs_dir / f"{sid}_AP.png"
        ml_path = self.imgs_dir / f"{sid}_ML.png"
        with Image.open(ap_path) as image:
            im_ap = np.asarray(image.convert('L'), dtype=np.float32)
        with Image.open(ml_path) as image:
            im_ml = np.asarray(image.convert('L'), dtype=np.float32)

        im_ap = im_ap.astype(np.float32) / 255.0
        im_ml = im_ml.astype(np.float32) / 255.0
        if im_ap.shape != (self.image_side, self.image_side) or im_ml.shape != (self.image_side, self.image_side):
            raise ValueError('Prepare images and matching camera geometry at image_side before training.')
        return np.stack([im_ap, im_ml], axis=0)

    def _load_stl_vertices(self, sid: str) -> np.ndarray:
        # NPY is mandatory: STL has no stable indexed vertex correspondence.
        return np.load(self.stl_dir / f"{sid}.npy", allow_pickle=False).astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sid = self.sample_ids[idx]
        image = self._load_image_pair(sid)
        verts_gt = self._load_stl_vertices(sid)
        if verts_gt.shape != self.nodes.shape:
            raise ValueError(f"GT 顶点形状 {verts_gt.shape} 与 canonical nodes {self.nodes.shape} 不一致。")
        if not np.all(np.isfinite(verts_gt)):
            raise ValueError(f'Nonfinite ground-truth coordinates: {sid}')
        return {
            "image": torch.from_numpy(image).float(),
            "verts_gt": torch.from_numpy(verts_gt).float(),
            "sample_id": sid,
        }
