# template_nlf: template neural localization fields

Reconstruct template-corresponding vertebral surface coordinates from paired AP/lateral intensity images and calibrated cone-beam geometry. Train a separate model for each lumbar level.

## Image-encoder acknowledgment

The image-encoder design references [DAGNet](https://github.com/ShilongHong/DAGNet) by ShilongHong, including FCEM, DVCA and CAFM. The implementation was independently written for paired vertebral images and template neural localization fields; DAGNet source code is not bundled. The encoder and related modules are in `models/image_encoder.py`, `models/biplanar_encoder.py` and `models/cross_view/`. See the [DAGNet architectural acknowledgment](../THIRD_PARTY_NOTICES.md) and [citation information](../CITATION.md).

## Setup

Use Python 3.11 or newer. All commands below run from this directory:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Training data

```text
data/
  L1_train.txt
  L1_imgs/<id>_AP.png
  L1_imgs/<id>_ML.png
  L1_stl/<id>.npy
  L1_proj_params/<id>.npz
  canonical/
    2000L1_nodes3.npy
    L1_faces.npy
    L1_eigval3.npy
    L1_eigvec3.npy
```

The ID list contains one anonymous sample ID per line. Images are 512×512 grayscale and normalized to [0,1]. Coordinates are `[2000,3]` arrays in mm, with the exact vertex ordering of the canonical template. The `L1_stl` folder contains NPY coordinate arrays, not STL files.

Export coordinates from the correspondence-aware mesh source. A matching vertex count alone does not establish correspondence, and automatic STL processing can change vertex order. Image dimensions are checked rather than resized automatically.

The template nodes have shape `[2000,3]`. Faces are zero-based integer indices with shape `[F,3]`. Eigenvalues include the constant mode at index zero, with at least 1025 entries; eigenvectors have shape `[2000,len(eigenvalues)]` in the same vertex order.

## Camera data

Each sample NPZ contains numeric arrays:

| Key | Shape | Meaning |
|---|---|---|
| `image_plane_vertices` | `[2,4,3]` | AP/ML detector corners in mm |
| `light_source` | `[2,3]` | AP/ML source positions in mm |

Detector corners are ordered top-left, bottom-left, bottom-right, top-right. View zero is AP; view one is ML, the lateral view. The alternative keys `camera1_image_plane_vertices`, `camera1_light_source`, `camera2_image_plane_vertices` and `camera2_light_source` are also accepted.

Geometry and mesh coordinates must share a common coordinate frame. The source/corner representation defines calibrated cone-beam projection; direct 3×4 projection matrix input is not implemented. Cropping, padding, resizing and horizontal flipping must be accounted for in calibration.

## Spectral assets

Given indexed template vertices and triangular faces:

```sh
python preprocessing/compute_spectral_encoding.py --vertices template_vertices.npy --faces template_faces.npy --vertebra-level L1 --output data/canonical
```

The utility solves the cotangent-stiffness / lumped-mass generalized eigenproblem. Stored arrays include the constant mode; the field skips that mode and uses `0.1 * phi / sqrt(lambda)` for the first 1024 nontrivial modes.

Keep the exact template and spectral assets with a checkpoint. Eigenvector signs and repeated-eigenvalue bases can differ when recomputed. The fixed encoding is also included in new checkpoints.

## Training

```sh
python train.py --config configs/train.toml
```

Configuration paths are resolved relative to this component. Update `vertebra_level`, ID list and output directory for L2–L5. Defaults are batch size 128, 300 epochs and AdamW with betas (0.9,0.999), epsilon 1e-8 and weight decay 0.003 applied to all parameters.

The base learning rate is 2.121e-4. The first 10 epochs use linear warmup; the main phase decays toward one-third of the base rate. The final 20 epochs start at base rate/30 and decay exponentially with a phase factor of 0.3. Each epoch reshuffles the training samples. A sample count other than 10000 is reported; training enforces 2000 template vertices.

The trainer saves model weights, epoch and configuration every 20 epochs by default. It does not implement optimizer-state resume, validation during training or TensorBoard logging.

## Architecture and losses

- The shared image encoder has 64/128/256/512-channel stages. FCEM follows Stage 1; DVCA follows Stages 2–4. Directional attention derives Q/K from one view and V from the other, with value-view residual connections. Spatial gates after Stages 2 and 3 modulate the corresponding next-stage features.
- Stage-4 view features feed the 2D branches. CAFM uses channel–spatial attention and convolutional fusion to generate the 3D branch features.
- A shared MLP, with a default hidden width of 384 and GELU activation, maps 1024 spectral features to depth-16 dynamic kernels and biases. The 2D branches sum logits over depth before spatial softmax and coordinate expectation decoding. The 3D branch uses joint depth/spatial softmax.
- Predicted 2D coordinates are triangulated. Relative 3D offsets are centered on the triangulated cloud centroid, then fused with triangulated coordinates. Fusion alpha is sampled independently per training sample from U(0,1); inference uses 0.5 by default.
- At epsilon=0.02, Charbonnier is `sqrt(mean(residual**2)+epsilon**2)-epsilon`. The mean inside the square root is over coordinate components: two for projected residuals, three for spatial residuals. Both losses supervise the fused surface; 2D residuals are scaled by `427/H` mm/pixel.
- Total loss is `safe(loss2d + loss3d) + 0.01 * first_layer_weight.square().sum(0).mean()`. Nonfinite scalar data losses are replaced by zero; nonfinite gradients stop training.

The relative 3D decoder maps normalized coordinates to [-50,50] mm per axis. With the default 512-pixel image size, stride32 and centered mapping, the 2D coordinate decoder spans [16,528]. These conventions must be kept consistent with checkpoint training and calibration. `projection_box_size_mm` supplies the 427 mm projection-loss scale.

## Inference and evaluation

```sh
python predict.py --config configs/predict.toml
```

Set `id_list`, `images_dir`, `camera_dir`, `checkpoint` and `output_dir`. The image folder contains `<id>AP.png` and `<id>ML.png`; the camera folder contains `<id>.npz`. Ground truth is optional. Inference exports `[2000,3]` NPY coordinates and indexed PLY meshes using the canonical faces.

To evaluate, set `gt_dir` to a folder containing `<id>/L1.stl`, changing the level as appropriate. Evaluation uses the common imaging frame without additional registration. ASSD averages the two directed vertex-to-triangle mean distances; HD is the maximum of both directed maxima. Results are written to `metrics.csv` in mm. These are vertex-sampled estimates, not continuous mesh Hausdorff distances or area-weighted integrals.

## Validation and assets

Tests use synthetic inputs and include a 512×512 forward/backward pass with a small synthetic template. No dataset, trained weights or template assets are bundled. Checkpoints require compatible model structure, spectral assets and vertex ordering.

## License

See the repository [Research-Only Noncommercial License](../LICENSE) and [third-party notices](../THIRD_PARTY_NOTICES.md).

## Configuration and checkpoint compatibility

The configuration uses method-specific names such as `vertebra_level`, `image_side`, `depth_bins`, `spectral_modes`, `projection_box_size_mm` and `inference_fusion_weight`. Use the supplied configuration files as examples. The loader also accepts legacy flat TOML setting names and rejects conflicting old/new values.

Checkpoints must match the network architecture, spectral assets and template vertex ordering used for training.

Project-owned code and modifications are limited to noncommercial scientific research. Publicly disseminated research results must include the manuscript reference in [CITATION.md](../CITATION.md). Commercial use is not permitted under the project license.
