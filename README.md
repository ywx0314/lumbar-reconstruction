# Lumbar Reconstruction

Domain-adapted vertebra extraction and template neural localization fields for lumbar surface reconstruction from biplanar radiographs.

The repository contains two independently runnable components:

| Component | Input | Output | Documentation |
|---|---|---|---|
| vertebra_extraction | A prepared AP or lateral radiograph | Five vertebral intensity images, L1–L5 | [Extraction guide](vertebra_extraction/README.md) |
| template_nlf | AP/lateral vertebral images, calibrated geometry and template assets | Template-corresponding 3D vertex coordinates and meshes | [Reconstruction guide](template_nlf/README.md) |

AP denotes anteroposterior. The filename suffix `ML` denotes the lateral view, also called LAT in the method description. Reconstruction uses a separate trained model for each lumbar level.

## Interactive demo

**[Explore the online demo ↗](https://ywx0314.github.io/lumbar-reconstruction/)** · [Reconstruction explorer](https://ywx0314.github.io/lumbar-reconstruction/#demo) · [Point mapping](https://ywx0314.github.io/lumbar-reconstruction/#mapping)

Explore the two-stage workflow from paired AP and lateral radiographs to vertebra extraction and anatomically corresponding L1–L5 surfaces. The reconstruction explorer presents five examples selected by surface agreement with CT-derived reference surfaces. Switch between original and extracted images, inspect calibrated projections, rotate the reconstruction, isolate individual lumbar levels, and overlay the CT reference.

[![Reconstruction explorer showing biplanar images, L1–L5 surfaces and a CT reference overlay](https://github.com/user-attachments/assets/befe8ba0-956f-4d8f-8690-368cfa757af3)](https://ywx0314.github.io/lumbar-reconstruction/#demo)

The point-mapping workspace uses the supplied L1–L5 statistical shape model templates as **Case 000**. Select a lumbar level and target case, then place surface or interior points on the template to explore their corresponding locations in the reconstruction. Linked cameras and matching point colors support side-by-side inspection; interior-point depth, shell opacity and coordinate readouts make the transfer visible. Interior points are transferred using mean value coordinates of the corresponding boundary.

[![Point-mapping workspace showing a template and reconstruction with a corresponding interior point](https://github.com/user-attachments/assets/79a7f160-96e7-461b-ac93-742aba5e6ccd)](https://ywx0314.github.io/lumbar-reconstruction/#mapping)

*Click either preview to open its interactive workspace.*

## Installation

Python 3.11 or newer is required. Local verification used Python 3.12 and PyTorch 2.11.0 on CPU. Create and activate a virtual environment, then install either or both components from the repository root:

```sh
python -m pip install -r vertebra_extraction/requirements.txt
python -m pip install -r template_nlf/requirements.txt
```

For GPU training, install a PyTorch build suitable for your GPU environment before installing the requirements. Both components use the same PyTorch version constraint and can share an environment. CPU checks install PyTorch from the CPU wheel index. Only packages used by the published code are listed; the dependency ranges are not a fully pinned environment, and their minimum versions have not all been tested.

## Extraction

Prepare data and anonymous ID lists as described in the [extraction guide](vertebra_extraction/README.md), then run from `vertebra_extraction/`:

```sh
python train.py --config configs/train.toml
python predict.py --checkpoint outputs/AP/epoch_200.pt --image prepared_AP.png --output results/case001/AP
python evaluate.py --config configs/evaluate.toml
```

Configure `plane = "ML"` and a separate output directory for lateral training. Inference writes `L1.png` through `L5.png` for each view.

## Reconstruction

Prepare paired vertebral images, template-corresponding coordinates, spectral assets and per-sample camera geometry as described in the [reconstruction guide](template_nlf/README.md). Run from `template_nlf/`:

```sh
python train.py --config configs/train.toml
python predict.py --config configs/predict.toml
```

Edit the level and paths in the configuration for L1–L5. Inference requires a compatible trained checkpoint and writes NPY coordinates and indexed PLY meshes.

## Connecting the stages

For a given vertebra, place the two extracted images in the reconstruction image directory as `<id>AP.png` and `<id>ML.png`. Both must be 512×512 and correspond to the same subject and calibrated acquisition.

Camera geometry must describe the exact image coordinate system used by reconstruction. Any cropping, padding, resizing or horizontal flipping must be reflected in that geometry. The extraction and reconstruction scripts do not automatically propagate image transforms into calibration. Select image orientation from calibration, rather than applying an unconditional flip.

## Tests

Run the components in separate processes to keep their local imports isolated:

```sh
cd vertebra_extraction
python -m unittest discover -s tests -v
cd ../template_nlf
python -m unittest discover -s tests -v
```

Tests use synthetic inputs and cover losses, gradient reversal, spatial gates, decoding, geometry and network forward/backward passes. GitHub Actions runs CPU tests separately for each component.

## Included assets

The model source does not bundle training/evaluation datasets, private sample lists, canonical template/spectral assets or trained weights. The separate [demo-site.zip](demo-site.zip) contains the precomputed visualization examples used by the online viewer.

Use compatible template assets and checkpoints; do not assume a checkpoint from a different implementation or template ordering is interchangeable.

## Acknowledgments

The reconstruction image-encoder design references [ShilongHong/DAGNet](https://github.com/ShilongHong/DAGNet), including FCEM, DVCA and CAFM. See the [reconstruction guide](template_nlf/README.md), [citation information](CITATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and upstream license information. The extraction network uses Microsoft HRNet, acknowledged in the third-party notices.

## Author

Wanxin Yu (yuwanxin@sjtu.edu.cn).

## License

Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn).

The project-owned code and modifications are available under the custom [Research-Only Noncommercial License](LICENSE): noncommercial scientific research only, with mandatory citation for publicly disseminated research results. Commercial use is not permitted under this license. Redistribution must preserve these conditions and all applicable notices. See [CITATION.md](CITATION.md) for the required manuscript reference.

This is a source-available research release, not an MIT-licensed or OSI-approved open-source project. Original Microsoft HRNet code retains its MIT license; see [third-party notices](THIRD_PARTY_NOTICES.md).
