# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.domain_adaptation import GradientReversal, FeatureDomainDiscriminator, gradient_reversal_coefficient
from models.hrnet.extraction_hrnet import build_extraction_network
from train import compute_extraction_loss
from datasets import SourceExtractionDataset

torch.set_num_threads(2)

class MethodsTests(unittest.TestCase):
    def test_grl_scales_feature_gradient_once(self):
        x = torch.tensor([2.0], requires_grad=True)
        weight = torch.tensor([3.0], requires_grad=True)
        (GradientReversal.apply(x, 0.4) * weight).sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([-1.2]))
        torch.testing.assert_close(weight.grad, torch.tensor([2.0]))
        self.assertEqual(gradient_reversal_coefficient(0), 0)
        self.assertGreater(gradient_reversal_coefficient(1), 0.99)

    def test_network_and_domain_backward(self):
        model = build_extraction_network()
        image = torch.rand(1, 1, 64, 64)
        prediction, features = model(image, return_features=True)
        self.assertEqual(tuple(prediction.shape), (1, 5, 64, 64))
        self.assertEqual(features.shape[1], 256)
        domain = FeatureDomainDiscriminator()(GradientReversal.apply(features, 0.5))
        loss = compute_extraction_loss(prediction, torch.rand_like(prediction)) + domain.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_extraction_mse_is_independent_of_image_area(self):
        prediction = torch.ones(1, 5, 2, 2)
        target = torch.zeros_like(prediction)
        self.assertEqual(compute_extraction_loss(prediction, target).item(), 60 + 15000)
        large_prediction = torch.ones(2, 5, 8, 8)
        self.assertEqual(compute_extraction_loss(large_prediction, torch.zeros_like(large_prediction)).item(), 60 + 15000)

    def test_crop_padding_keeps_image_and_label_aligned(self):
        image = np.arange(12, dtype=np.float32).reshape(3, 4)
        cropped = SourceExtractionDataset._crop_or_pad_image(image, (0, 3, 0, 4), (8, 8))
        np.testing.assert_array_equal(cropped[2:5, 2:6], image)
        self.assertEqual(cropped.shape, (8, 8))

if __name__ == '__main__':
    unittest.main()
