# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Reconstruct indexed template vertices; optionally evaluate symmetric surface distances."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
import trimesh
from configuration import load_reconstruction_config
from datasets.synthetic_dataset import set_random_seed
from models.template_nlf import TemplateNeuralLocalizationField
from models.camera_parameters import load_sample_camera_parameters, build_camera_batch

def compute_surface_distance_metrics(predicted, reference):
    """Vertex-to-triangle ASSD and symmetric HD, without registration.

    This is a discrete vertex-sampled surface estimate, not continuous mesh HD.
    """
    def distances(source, target):
        result = []
        for start in range(0, len(source.vertices), 128):
            _, distance, _ = trimesh.proximity.closest_point(target, source.vertices[start:start + 128])
            result.extend(distance)
        return np.asarray(result)
    forward, backward = distances(predicted, reference), distances(reference, predicted)
    return 0.5 * (forward.mean() + backward.mean()), max(forward.max(), backward.max())

@torch.inference_mode()
def main():
    from PIL import Image
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/predict.toml')
    args = p.parse_args()
    cfg = load_reconstruction_config(args.config)
    set_random_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    canonical = Path(cfg.data_root) / 'canonical'
    nodes = np.load(canonical / f'2000{cfg.vertebra_level}_nodes3.npy', allow_pickle=False)
    faces = np.load(canonical / f'{cfg.vertebra_level}_faces.npy', allow_pickle=False)
    ids = [line.strip() for line in Path(cfg.id_list).read_text(encoding='utf-8').splitlines() if line.strip()]
    if not ids:
        raise ValueError('Empty inference ID list.')
    cameras = load_sample_camera_parameters(Path(cfg.data_root), vertebra_level=cfg.vertebra_level,
                                         sample_ids=ids, params_dir=cfg.camera_dir)
    model = TemplateNeuralLocalizationField(cfg, nodes).to(device)
    checkpoint = torch.load(cfg.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.eval()
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    for sid in ids:
        images = []
        for view in ('AP', 'ML'):
            path = Path(cfg.images_dir) / f'{sid}{view}.png'
            with Image.open(path) as source:
                image = np.asarray(source.convert('L'), dtype=np.float32)
            if image.shape != (cfg.image_side, cfg.image_side):
                raise ValueError(f'Expected prepared {cfg.image_side} x {cfg.image_side} image: {path}')
            images.append(image.astype(np.float32) / 255)
        planes, lights = build_camera_batch([sid], cameras, device=device)
        pred = model(torch.from_numpy(np.stack(images))[None].to(device), planes, lights)[0].cpu().numpy()
        np.save(output / f'{sid}.npy', pred)
        mesh = trimesh.Trimesh(vertices=pred, faces=faces, process=False)
        mesh.export(output / f'{sid}.ply')
        if getattr(cfg, 'gt_dir', ''):
            reference = trimesh.load(Path(cfg.gt_dir) / sid / f'{cfg.vertebra_level}.stl')
            assd, hd = compute_surface_distance_metrics(mesh, reference)
            metrics.append((sid, assd, hd))
    if metrics:
        with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(('sample_id', 'ASSD_mm', 'HD_mm'))
            writer.writerows(metrics)

if __name__ == '__main__':
    main()
