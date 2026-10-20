# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Single-view source and unlabeled target datasets. Intensities are in [0, 1]."""
import pathlib
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
class SourceExtractionDataset(Dataset):
    def __init__(
        self,
        txt_file: str | pathlib.Path,
        data_root: str | pathlib.Path,
        image_size: int | tuple[int, int],
        plane: str,
        use_augmentation: bool = False,
        preload: bool = False,
        n_worker: int = 0,
        debug: bool = False,
        edge_occlusion_prob: float = 0.0,
        intensity_aug_prob: float = 0.0,
        brightness_shift: float = 0.1,
        contrast_range: tuple[float, float] = (0.8, 1.2),
        gamma_range: tuple[float, float] = (0.8, 1.5),
        noise_std_range: tuple[float, float] = (0.0, 0.02),
    ):
        super().__init__()
        self._id_list_path = pathlib.Path(txt_file)
        self._data_root = pathlib.Path(data_root)
        self._image_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self._plane = plane.upper()
        if self._plane not in {"AP", "ML"}:
            raise ValueError(f"plane 只能是 AP/ML，收到: {plane}")

        self._use_augmentation = use_augmentation
        self._preload = preload
        self._edge_occlusion_prob = edge_occlusion_prob if use_augmentation else 0.0
        self._intensity_aug_prob = intensity_aug_prob if use_augmentation else 0.0
        self._brightness_shift = brightness_shift
        self._contrast_range = contrast_range
        self._gamma_range = gamma_range
        self._noise_std_range = noise_std_range

        self._radiograph_directory = self._data_root / "Xray"
        self._vertebra_image_directories = [self._data_root / f"Mask_L{i}" for i in range(1, 6)]
        self._mask_index = 0 if self._plane == "AP" else 1

        names = []
        with self._id_list_path.open("r", encoding="utf-8") as f:
            for line in f:
                n = line.strip()
                if n:
                    names.append(n)

        self._sample_ids: list[str] = []
        self._img_paths: list[pathlib.Path] = []
        self._vertebra_image_paths: list[list[pathlib.Path]] = []
        for n in names:
            img_p = self._radiograph_directory / f"{n}{self._plane}.jpg"
            masks = [d / f"{n}{self._plane}.jpg" for d in self._vertebra_image_directories]
            if img_p.exists() and all(m.exists() for m in masks):
                self._sample_ids.append(n)
                self._img_paths.append(img_p)
                self._vertebra_image_paths.append(masks)

        if not self._sample_ids:
            raise ValueError(f"No complete samples found in {self._id_list_path}")
        if len(self._sample_ids) != len(names):
            raise FileNotFoundError(f"Missing image or label files listed in {self._id_list_path}")
        if debug:
            self._sample_ids = self._sample_ids[:10]
            self._img_paths = self._img_paths[:10]
            self._vertebra_image_paths = self._vertebra_image_paths[:10]

        self._img_cache = None
        self._vertebra_image_cache = None
        if self._preload:
            self._img_cache = [self._load_jpg(p) for p in tqdm(self._img_paths, desc=f"Preload {self._plane}", mininterval=10)]
            self._vertebra_image_cache = []
            for mp in tqdm(self._vertebra_image_paths, desc=f"Preload mask {self._plane}", mininterval=10):
                self._vertebra_image_cache.append([self._load_jpg(x) for x in mp])

    @staticmethod
    def _load_jpg(path: pathlib.Path) -> np.ndarray:
        return np.array(Image.open(path).convert("L"), dtype=np.float32)

    @staticmethod
    def _bounding_box_from_labels(mask: np.ndarray) -> tuple[int, int, int, int]:
        ys, xs = np.where(mask > 1e-6)
        if len(xs) == 0 or len(ys) == 0:
            h, w = mask.shape
            return 0, h, 0, w
        return int(ys.min()), int(ys.max() + 1), int(xs.min()), int(xs.max() + 1)

    @staticmethod
    def _crop_or_pad_image(img: np.ndarray, crop: tuple[int, int, int, int], out_hw: tuple[int, int]) -> np.ndarray:
        y0, y1, x0, x1 = crop
        patch = img[y0:y1, x0:x1]
        out_h, out_w = out_hw
        h, w = patch.shape
        if h > out_h:
            s = (h - out_h) // 2
            patch = patch[s:s + out_h, :]
            h = out_h
        if w > out_w:
            s = (w - out_w) // 2
            patch = patch[:, s:s + out_w]
            w = out_w
        canvas = np.zeros((out_h, out_w), dtype=np.float32)
        sy = (out_h - h) // 2
        sx = (out_w - w) // 2
        canvas[sy:sy + h, sx:sx + w] = patch
        return canvas

    def _compute_lumbar_crop(self, vertebra_images: np.ndarray) -> tuple[int, int, int, int]:
        combined = vertebra_images.sum(axis=0)
        y0, y1, x0, x1 = self._bounding_box_from_labels(combined)
        h, w = self._image_size
        cy = (y0 + y1) // 2
        cx = (x0 + x1) // 2
        sy = max(0, cy - h // 2)
        sx = max(0, cx - w // 2)
        ey = sy + h
        ex = sx + w
        img_h, img_w = combined.shape
        if ey > img_h:
            ey = img_h
            sy = max(0, ey - h)
        if ex > img_w:
            ex = img_w
            sx = max(0, ex - w)
        return sy, ey, sx, ex

    def _augment_intensity(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() >= self._intensity_aug_prob:
            return x
        mu = float(x.mean())
        contrast = np.random.uniform(*self._contrast_range)
        gamma = np.random.uniform(*self._gamma_range)
        shift = np.random.uniform(-self._brightness_shift, self._brightness_shift)
        noise_std = np.random.uniform(*self._noise_std_range)
        x = (x - mu) * contrast + mu
        x = np.clip(x + shift, 0.0, 1.0)
        x = np.power(np.clip(x, 1e-6, 1.0), gamma).astype(np.float32)
        if noise_std > 0:
            x = x + np.random.normal(0.0, noise_std, size=x.shape).astype(np.float32)
        return np.clip(x, 0.0, 1.0)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self._preload:
            img = self._img_cache[idx]
            masks = self._vertebra_image_cache[idx]
        else:
            img = self._load_jpg(self._img_paths[idx])
            masks = [self._load_jpg(p) for p in self._vertebra_image_paths[idx]]

        vertebra_images = np.stack(masks, axis=0) / 255.0
        crop = self._compute_lumbar_crop(vertebra_images)
        img = self._crop_or_pad_image(img, crop, self._image_size) / 255.0
        vertebra_images = np.stack([self._crop_or_pad_image(m, crop, self._image_size) / 255.0 for m in masks], axis=0)
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        vertebra_images = np.clip(vertebra_images, 0.0, 1.0).astype(np.float32)

        if self._use_augmentation:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1)
                vertebra_images = np.flip(vertebra_images, axis=2)

            if np.random.rand() < self._edge_occlusion_prob:
                w = img.shape[1]
                occ = int(np.random.uniform(0.0, 0.15) * w)
                if occ > 0:
                    img[:, :occ] = 0.0
                    img[:, -occ:] = 0.0
                    vertebra_images[:, :, :occ] = 0.0
                    vertebra_images[:, :, -occ:] = 0.0

            img = self._augment_intensity(img)

        img = np.ascontiguousarray(img[None, ...])  # (1, H, W)
        vertebra_images = np.ascontiguousarray(vertebra_images)  # (5, H, W)

        return {
            "image_id": self._sample_ids[idx],
            "image": torch.from_numpy(img),
            "mask": torch.from_numpy(vertebra_images),
        }

class TargetRadiographDataset(Dataset):
    """Prepared target-domain radiographs; validate shape without random augmentation."""

    def __init__(
        self,
        txt_file: str | pathlib.Path,
        data_root: str | pathlib.Path,
        image_size: int | tuple[int, int],
        plane: str,
        preload: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        self._id_list_path = pathlib.Path(txt_file)
        self._data_root = pathlib.Path(data_root)
        self._image_size = tuple(image_size) if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self._plane = plane.upper()
        self._preload = preload
        self._radiograph_directory = self._data_root / "Xray"

        names = []
        with self._id_list_path.open("r", encoding="utf-8") as f:
            for line in f:
                n = line.strip()
                if n:
                    names.append(n)

        self._sample_ids: list[str] = []
        self._img_paths: list[pathlib.Path] = []
        for n in names:
            p = self._radiograph_directory / f"{n}{self._plane}.jpg"
            if p.exists():
                self._sample_ids.append(n)
                self._img_paths.append(p)

        if not self._sample_ids:
            raise ValueError(f"No complete samples found in {self._id_list_path}")
        if len(self._sample_ids) != len(names):
            raise FileNotFoundError(f"Missing image or label files listed in {self._id_list_path}")
        if debug:
            self._sample_ids = self._sample_ids[:10]
            self._img_paths = self._img_paths[:10]

        self._img_cache = None
        if self._preload:
            self._img_cache = [np.array(Image.open(p).convert("L"), dtype=np.float32) for p in tqdm(self._img_paths, desc=f"Preload target {self._plane}", mininterval=10)]

    @staticmethod
    def _validate_image_shape(image: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
        if image.shape != out_hw:
            raise ValueError('Prepare target radiographs with a crop/pad around L1--L5 at image_size.')
        return image.astype(np.float32)

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        if self._preload:
            img = self._img_cache[idx]
        else:
            img = np.array(Image.open(self._img_paths[idx]).convert("L"), dtype=np.float32)
        img = self._validate_image_shape(img, self._image_size) / 255.0
        img = np.clip(img, 0.0, 1.0).astype(np.float32)
        img = np.ascontiguousarray(img[None, ...])
        return {"image_id": self._sample_ids[idx], "image": torch.from_numpy(img)}
