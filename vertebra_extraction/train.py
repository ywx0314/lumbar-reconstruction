# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Train the five-channel, single-radiograph extraction network."""
import argparse
import json
from pathlib import Path
import tomllib
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from datasets import SourceExtractionDataset, TargetRadiographDataset
from models.hrnet.extraction_hrnet import build_extraction_network
from models.domain_adaptation import FeatureDomainDiscriminator, GradientReversal, gradient_reversal_coefficient
from models.losses.ncc_loss import NCCLoss

def compute_extraction_loss(prediction, target):
    # Updated Methods: pixel MSE, averaged equally over samples and L1--L5.
    return 60 * NCCLoss()(prediction, target) + 15000 * F.mse_loss(prediction, target)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/train.toml')
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open('rb') as f:
        cfg = tomllib.load(f)
    root = config_path.parent.parent
    resolve = lambda p: str((root / p).resolve())
    torch.manual_seed(cfg['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    source = SourceExtractionDataset(resolve(cfg['source_list']), resolve(cfg['data_root']),
        cfg['image_size'], cfg['plane'], use_augmentation=True,
        edge_occlusion_prob=0.5, intensity_aug_prob=0.8,
        brightness_shift=0.2, noise_std_range=(0, 0.03))
    target = TargetRadiographDataset(resolve(cfg['target_list']), resolve(cfg['data_root']),
        cfg['image_size'], cfg['plane'])
    source_loader = DataLoader(source, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['workers'])
    target_loader = DataLoader(target, batch_size=cfg['batch_size'], shuffle=True, num_workers=cfg['workers'])
    generator, discriminator = build_extraction_network().to(device), FeatureDomainDiscriminator().to(device)
    generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=1e-3, weight_decay=1e-3)
    domain_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=1e-4, weight_decay=1e-3)
    schedule = torch.optim.lr_scheduler.StepLR(generator_optimizer, step_size=20, gamma=0.9)
    output = Path(resolve(cfg['output_dir']))
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(cfg, indent=2), encoding='utf-8')
    total_steps = cfg['epochs'] * len(source_loader)
    for epoch in range(cfg['epochs']):
        generator.train()
        discriminator.train()
        target_iter = iter(target_loader)
        total = 0.0
        for batch_index, batch in enumerate(source_loader):
            try:
                unlabeled = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                unlabeled = next(target_iter)
            generator_optimizer.zero_grad(set_to_none=True)
            domain_optimizer.zero_grad(set_to_none=True)
            prediction, source_features = generator(batch['image'].to(device), return_features=True)
            _, target_features = generator(unlabeled['image'].to(device), return_features=True)
            progress = (epoch * len(source_loader) + batch_index) / max(total_steps - 1, 1)
            coefficient = gradient_reversal_coefficient(progress)
            # A single CE objective: GRL applies lambda once to generator gradients;
            # discriminator gradients retain full-strength classification supervision.
            source_domain_logits = discriminator(GradientReversal.apply(source_features, coefficient))
            target_domain_logits = discriminator(GradientReversal.apply(target_features, coefficient))
            domain = 0.5 * (F.cross_entropy(source_domain_logits, torch.zeros(len(source_features), dtype=torch.long, device=device))
                          + F.cross_entropy(target_domain_logits, torch.ones(len(target_features), dtype=torch.long, device=device)))
            # lambda_adv = 1; lambda_GRL is applied only by GradientReversal.
            loss = compute_extraction_loss(prediction, batch['mask'].to(device)) + domain
            loss.backward()
            generator_optimizer.step()
            domain_optimizer.step()
            total += loss.item()
        schedule.step()
        print(f"epoch={epoch + 1} loss={total / len(source_loader):.6f}")
        if (epoch + 1) % cfg['save_every'] == 0 or epoch + 1 == cfg['epochs']:
            torch.save({'generator': generator.state_dict(), 'domain': discriminator.state_dict(),
                        'epoch': epoch + 1, 'config': cfg}, output / f'epoch_{epoch + 1:03d}.pt')

if __name__ == '__main__':
    main()
