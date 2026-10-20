# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Evaluate intensity extraction using per-image, per-channel Dice, PSNR and SSIM."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def compute_extraction_metrics(prediction, reference, threshold=0.005):
    pred_mask, true_mask = prediction > threshold, reference > threshold
    denominator = pred_mask.sum() + true_mask.sum()
    dice = 1.0 if denominator == 0 else 2 * np.logical_and(pred_mask, true_mask).sum() / denominator
    return dice, peak_signal_noise_ratio(reference, prediction, data_range=1), structural_similarity(reference, prediction, data_range=1)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prediction', required=True, help='Directory with L1.png--L5.png')
    p.add_argument('--reference', required=True, help='Directory with L1.png--L5.png')
    p.add_argument('--threshold', type=float, default=0.005, help='Evaluation choice; not specified in Methods')
    args = p.parse_args()
    scores = []
    for level in range(1, 6):
        arrays = [np.asarray(Image.open(Path(folder) / f'L{level}.png').convert('L'), dtype=float) / 255
                  for folder in (args.prediction, args.reference)]
        scores.append(compute_extraction_metrics(*arrays, args.threshold))
    for name, value in zip(('Dice', 'PSNR', 'SSIM'), np.mean(scores, axis=0)):
        print(f'{name}: {value:.6f}')

if __name__ == '__main__':
    main()
