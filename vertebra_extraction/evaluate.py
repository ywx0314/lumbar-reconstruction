# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Load an extraction checkpoint and evaluate every labeled test radiograph."""
import argparse
import csv
import json
from pathlib import Path
import tomllib
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import SourceExtractionDataset
from evaluate_images import compute_extraction_metrics
from predict import load_extraction_model, save_vertebra_images


@torch.inference_mode()
def evaluate_test_set(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open('rb') as stream:
        settings = tomllib.load(stream)
    component_root = Path(__file__).resolve().parent
    resolve = lambda name: (component_root / settings[name]).resolve()
    dataset = SourceExtractionDataset(resolve('id_list'), resolve('data_root'),
                                      settings['image_size'], settings['plane'], use_augmentation=False)
    loader = DataLoader(dataset, batch_size=settings.get('batch_size', 1), shuffle=False,
                        num_workers=settings.get('workers', 0))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_extraction_model(resolve('checkpoint'), device)
    output = resolve('output_dir')
    output.mkdir(parents=True, exist_ok=True)
    threshold = float(settings.get('dice_threshold', 0.005))
    rows = []
    for batch in loader:
        predictions = model(batch['image'].to(device)).cpu().numpy()
        references = batch['mask'].numpy()
        for sample_id, prediction, reference in zip(batch['image_id'], predictions, references):
            channel_scores = [compute_extraction_metrics(p, r, threshold)
                              for p, r in zip(prediction, reference)]
            dice, psnr, ssim = np.mean(channel_scores, axis=0)
            rows.append((sample_id, settings['plane'], float(dice), float(psnr), float(ssim)))
            if settings.get('save_predictions', True):
                save_vertebra_images(prediction, output / 'extracted' / sample_id)
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(('sample_id', 'view', 'dice', 'psnr', 'ssim'))
        writer.writerows(rows)
    means = np.mean([row[2:] for row in rows], axis=0)
    summary = {'samples': len(rows), 'view': settings['plane'], 'dice_threshold': threshold}
    for name, value in zip(('dice', 'psnr', 'ssim'), means):
        summary[name] = float(value) if np.isfinite(value) else str(float(value))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/evaluate.toml')
    args = parser.parse_args()
    evaluate_test_set(args.config)


if __name__ == '__main__':
    main()
