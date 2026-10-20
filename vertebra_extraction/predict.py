# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
"""Extract L1--L5 intensity images from a prepared grayscale radiograph."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models.hrnet.extraction_hrnet import build_extraction_network

def load_extraction_model(checkpoint_path, device):
    """Load the same generator state dictionary written by train.py."""
    model = build_extraction_network().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['generator'], strict=True)
    return model.eval()


def save_vertebra_images(prediction, output_directory):
    """Write five normalized intensity channels in template lumbar order."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for level, channel in enumerate(prediction, 1):
        Image.fromarray((channel.clip(0, 1) * 255).round().astype('uint8')).save(output / f'L{level}.png')


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    image = np.asarray(Image.open(args.image).convert('L'), dtype=np.float32) / 255
    if image.shape != (512, 512):
        raise ValueError('Prepare a 512 x 512 crop/padded image around L1--L5 before inference.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_extraction_model(args.checkpoint, device)
    result = model(torch.from_numpy(image.copy())[None, None].to(device))[0].cpu().numpy()
    save_vertebra_images(result, args.output)

if __name__ == '__main__':
    main()
