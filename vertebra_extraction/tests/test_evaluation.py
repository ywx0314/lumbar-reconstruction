# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import evaluate_test_set


class IdentityExtractionModel(torch.nn.Module):
    def forward(self, images):
        return images.repeat(1, 5, 1, 1)


class DatasetEvaluationTests(unittest.TestCase):
    def test_all_test_cases_are_evaluated_and_exported(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            root = Path(directory)
            ids = ['synthetic_a', 'synthetic_b', 'synthetic_c']
            for folder in ['Xray'] + [f'Mask_L{level}' for level in range(1, 6)]:
                (root / 'data' / folder).mkdir(parents=True)
                for sample_id in ids:
                    Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).save(root / 'data' / folder / f'{sample_id}AP.jpg')
            (root / 'ids.txt').write_text('\n'.join(ids), encoding='utf-8')
            settings = '\n'.join([
                'plane = "AP"', 'image_size = [32, 32]', 'batch_size = 2', 'workers = 0',
                f'data_root = "{(root / "data").as_posix()}"',
                f'id_list = "{(root / "ids.txt").as_posix()}"',
                f'checkpoint = "{(root / "not_used.pt").as_posix()}"',
                f'output_dir = "{(root / "evaluation").as_posix()}"',
            ])
            (root / 'test.toml').write_text(settings, encoding='utf-8')
            with patch('evaluate.load_extraction_model', return_value=IdentityExtractionModel()):
                summary = evaluate_test_set(root / 'test.toml')
            self.assertEqual(summary['samples'], 3)
            self.assertAlmostEqual(summary['dice'], 1)
            self.assertAlmostEqual(summary['ssim'], 1)
            self.assertEqual(summary['psnr'], 'inf')
            with (root / 'evaluation/metrics.csv').open(encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['sample_id'] for row in rows], ids)
            for sample_id in ids:
                self.assertEqual(len(list((root / 'evaluation/extracted' / sample_id).glob('L*.png'))), 5)


if __name__ == '__main__':
    unittest.main()
