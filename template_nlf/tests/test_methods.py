# Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn)
# SPDX-License-Identifier: LicenseRef-Research-Only-Noncommercial
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tensor_ops import charbonnier
from models.cone_beam_geometry import project_cone_beam_vectorized, triangulate_from_two_views
from models.template_nlf import TemplateNeuralLocalizationField
from models.reconstruction_losses import compute_projection_loss, compute_spatial_vertex_loss
from configuration import compute_learning_rate
from preprocessing.compute_spectral_encoding import cotangent_matrices
from predict import compute_surface_distance_metrics
import trimesh

torch.set_num_threads(2)

def cameras():
    planes = torch.tensor([[[[-50., -50., 100.], [-50., 50., 100.], [50., 50., 100.], [50., -50., 100.]],
                            [[100., -50., -50.], [100., 50., -50.], [100., 50., 50.], [100., -50., 50.]]]])
    lights = torch.tensor([[[0., 0., -100.], [-100., 0., 0.]]])
    return planes, lights

class MethodsTests(unittest.TestCase):
    def test_legacy_flat_configuration_names(self):
        from configuration import load_reconstruction_config
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory) / 'legacy.toml'
            path.write_text('vertebra = "L1"\ndata_root_vertebrae = "data"\noutput_dir = "outputs/L1"\nproc_side = 512\nbox_size_m = 427.0\n', encoding='utf-8')
            cfg = load_reconstruction_config(path)
            self.assertEqual(cfg.vertebra_level, 'L1')
            self.assertEqual(cfg.image_side, 512)
            self.assertEqual(cfg.projection_box_size_mm, 427)

    def test_charbonnier_uses_mean_squared_components(self):
        residual = torch.tensor([[3., 4., 0.]])
        torch.testing.assert_close(charbonnier(residual, epsilon=0.02), torch.tensor([((25 / 3 + 0.0004) ** 0.5) - 0.02]))
        # Replicating coordinates preserves RMS magnitude, unlike the L2 norm.
        torch.testing.assert_close(charbonnier(torch.ones(1, 2), epsilon=0.02),
                                   charbonnier(torch.ones(1, 3), epsilon=0.02))

    def test_original_attention_and_spatial_gate_gradients(self):
        from models.cross_view.dvca import DVCA
        block = DVCA(8, 16).eval()
        a, b = torch.randn(1, 8, 4, 4), torch.randn(1, 8, 4, 4)
        fa, fb, sa, sb = block(a, b)
        self.assertEqual(tuple(sa.shape), (1, 1, 2, 2))
        self.assertTrue(torch.all((sa > 0) & (sa < 1)))
        next_a, next_b = torch.randn(1, 16, 2, 2), torch.randn(1, 16, 2, 2)
        loss = (next_a * sa).square().mean() + (next_b * sb).square().mean()
        loss.backward()
        self.assertGreater(block.spatial.weight.grad.abs().sum().item(), 0)
        self.assertGreater(block.ffn.dconv.weight.grad.abs().sum().item(), 0)

    def test_2d_decoder_sums_logits_before_softmax(self):
        from models.runtime import MODEL_SETTINGS
        from models.localization_head import DynamicLocalizationHead
        for key, value in dict(depth_bins=2, stride_test=32, centered_stride=True,
                               projection_box_size_mm=427., image_side=512, feature_channels=1).items():
            setattr(MODEL_SETTINGS, key, value)
        head = DynamicLocalizationHead(torch.nn.Identity(), torch.nn.BatchNorm2d, 1, 1)
        features = torch.tensor([[[[0., 2.]]]])
        # Opposite depth_bins logits cancel at each pixel, giving a uniform 2D map.
        weights = torch.tensor([[1., -1., 0., 0.]])
        coords2d, _ = head.apply_weights3d(features[:, None].expand(-1, 2, -1, -1, -1),
                                          features, weights, 2)
        torch.testing.assert_close(coords2d[..., 0], torch.full((1, 2, 1), 0.5))

    def test_projection_triangulation_and_losses(self):
        planes, lights = cameras()
        points = torch.tensor([[[0., 0., 0.], [8., -6., 3.]]])
        pixels = [project_cone_beam_vectorized(points, planes[:, c], lights[:, c], (512, 512)) for c in range(2)]
        reconstructed = triangulate_from_two_views(*pixels, planes[:, 0], planes[:, 1], lights[:, 0], lights[:, 1], (512, 512))
        torch.testing.assert_close(points, reconstructed, atol=3e-5, rtol=1e-4)
        self.assertEqual(compute_spatial_vertex_loss(points, points).item(), 0)
        self.assertEqual(compute_projection_loss(points, points, planes, lights, image_side=512, projection_box_size_mm=427).item(), 0)

    def test_cotangent_operator_and_schedule(self):
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]])
        stiffness, mass = cotangent_matrices(vertices, faces)
        np.testing.assert_allclose(stiffness.toarray(), stiffness.toarray().T)
        np.testing.assert_allclose(stiffness @ np.ones(4), 0, atol=1e-14)
        self.assertTrue(np.all(mass.diagonal() > 0))
        self.assertAlmostEqual(compute_learning_rate(9), 2.121e-4)
        self.assertAlmostEqual(compute_learning_rate(280), 2.121e-4 / 30)

    def test_symmetric_metrics(self):
        mesh = trimesh.creation.box()
        assd, hd = compute_surface_distance_metrics(mesh, mesh)
        self.assertEqual(assd, 0)
        self.assertEqual(hd, 0)
        shifted = mesh.copy()
        shifted.apply_translation([2, 0, 0])
        np.testing.assert_allclose(compute_surface_distance_metrics(mesh, shifted), compute_surface_distance_metrics(shifted, mesh))

    def test_full_network_forward_backward_and_checkpoint(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            canonical = Path(directory) / 'canonical'
            canonical.mkdir()
            np.save(canonical / 'L1_eigval3.npy', np.array([0., 1, 2, 3, 4]))
            np.save(canonical / 'L1_eigvec3.npy', np.random.default_rng(1).normal(size=(6, 5)))
            cfg = SimpleNamespace(data_root=directory, vertebra_level='L1', image_side=512,
                                  stride_train=32, stride_test=32, centered_stride=True,
                                  dtype='float32', spectral_modes=4, kernel_mlp_hidden_dim=8)
            model = TemplateNeuralLocalizationField(cfg, np.zeros((6, 3)))
            planes, lights = cameras()
            images = torch.rand(1, 2, 512, 512)
            model.train()
            prediction = model(images, planes, lights)
            self.assertEqual(tuple(prediction.shape), (1, 6, 3))
            prediction.square().mean().backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertIsNotNone(model.heatmap_head.weight_field.pred_mlp[0].weight.grad)
            for block in (model.backbone.encoder.cfe2, model.backbone.encoder.cfe3):
                self.assertIsNotNone(block.spatial.weight.grad)
                self.assertGreater(block.spatial.weight.grad.abs().sum().item(), 0)
            model.eval()
            with torch.no_grad():
                first = model(images, planes, lights)
                second = model(images, planes, lights)
            torch.testing.assert_close(first, second)
            state = model.state_dict()
            self.assertIn('heatmap_head.weight_field.fixed_posenc', state)
            clone = TemplateNeuralLocalizationField(cfg, np.zeros((6, 3)))
            clone.load_state_dict(state, strict=True)

if __name__ == '__main__':
    unittest.main()
