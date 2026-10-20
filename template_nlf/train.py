# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Train a template neural localization field using synthetic paired data."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from configuration import load_reconstruction_config, compute_learning_rate
from datasets.synthetic_dataset import SyntheticVertebraDataset, set_random_seed
from models.template_nlf import TemplateNeuralLocalizationField, compute_kernel_weight_regularization
from models.camera_parameters import load_sample_camera_parameters, build_camera_batch
from models.reconstruction_losses import compute_spatial_vertex_loss, compute_projection_loss

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.toml')
    args = parser.parse_args()
    cfg = load_reconstruction_config(args.config)
    set_random_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = SyntheticVertebraDataset(cfg.data_root, cfg.id_list, image_side=cfg.image_side, vertebra_level=cfg.vertebra_level)
    if ds.nodes.shape != (2000, 3):
        raise ValueError('The manuscript uses 2000 template vertices.')
    if len(ds) != 10000:
        print(f'Note: {len(ds)} samples supplied; the manuscript reports 10000 per lumbar level.')
    cameras = load_sample_camera_parameters(Path(cfg.data_root), vertebra_level=cfg.vertebra_level, sample_ids=ds.sample_ids)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    model = TemplateNeuralLocalizationField(cfg, ds.nodes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.base_learning_rate,
                                  weight_decay=cfg.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(vars(cfg), indent=2), encoding='utf-8')
    for epoch in range(cfg.epochs):
        model.train()
        lr = compute_learning_rate(epoch, epochs=cfg.epochs, base_lr=cfg.base_learning_rate)
        for group in optimizer.param_groups:
            group['lr'] = lr
        total = 0.0
        for batch in loader:
            image = batch['image'].to(device)
            gt = batch['verts_gt'].to(device)
            planes, lights = build_camera_batch(batch['sample_id'], cameras, device=device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(image, planes, lights)
            projection_loss = compute_projection_loss(pred, gt, planes, lights,
                                             image_side=cfg.image_side, projection_box_size_mm=cfg.projection_box_size_mm)
            loss3d = compute_spatial_vertex_loss(gt, pred)
            data_loss = torch.nan_to_num(projection_loss + cfg.spatial_loss_weight * loss3d, nan=0., posinf=0., neginf=0.)
            regularization = compute_kernel_weight_regularization(model.heatmap_head.weight_field)
            loss = cfg.surface_loss_weight * data_loss + cfg.kernel_regularization_weight * regularization
            loss.backward()
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite gradients: check input geometry and template encoding.')
            optimizer.step()
            total += loss.item()
        print(f'epoch={epoch + 1} lr={lr:.8g} loss={total / len(loader):.6f}')
        if (epoch + 1) % cfg.save_every == 0 or epoch + 1 == cfg.epochs:
            torch.save({'model_state_dict': model.state_dict(), 'config': vars(cfg), 'epoch': epoch + 1},
                       output / f'epoch_{epoch + 1:03d}.pt')

if __name__ == '__main__':
    main()
