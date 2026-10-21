# Lumbar Reconstruction

Domain-adapted vertebra extraction and template neural localization fields for lumbar surface reconstruction from biplanar radiographs.

The repository contains two independently runnable components:

| Component | Input | Output | Documentation |
|---|---|---|---|
| vertebra_extraction | A prepared AP or lateral radiograph | Five vertebral intensity images, L1–L5 | [Extraction guide](vertebra_extraction/README.md) |
| template_nlf | AP/lateral vertebral images, calibrated geometry and template assets | Template-corresponding 3D vertex coordinates and meshes | [Reconstruction guide](template_nlf/README.md) |

AP denotes anteroposterior. The filename suffix `ML` denotes the lateral view, also called LAT in the method description. Reconstruction uses a separate trained model for each lumbar level.

## Interactive demo

**[Open the online demo ↗](https://ywx0314.github.io/lumbar-reconstruction/)** · [Demo guide](https://ywx0314.github.io/lumbar-reconstruction/README.md)

Explore 12 precomputed cases with paired original/extracted AP and lateral radiographs, calibrated projection overlays, freely rotatable L1–L5 surfaces and optional CT reference overlays. Switch cases, select lumbar levels, and adjust image/surface opacity in the browser.

[![Interactive lumbar reconstruction demo with biplanar images and L1–L5 surfaces](https://github.com/user-attachments/assets/b0e335c3-074b-4b4a-80c4-ae4250d7812f)](https://ywx0314.github.io/lumbar-reconstruction/)

*Preview of the interactive workspace. Click the image to open the demo.*

The demo displays precomputed reconstruction results; it does not run network inference in the browser. No model installation or GPU is required to explore the examples.

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

## Included assets and validation limits

The model source does not bundle training/evaluation datasets, private sample lists, canonical template/spectral assets or trained weights. The separate [demo-site.zip](demo-site.zip) contains the precomputed visualization examples used by the online viewer. The complete SSM, local-deformation and DRR data-generation pipeline is not included.

Full GPU training, published cohort metrics and inference timings have not been reproduced by the included source tests. Use compatible template assets and checkpoints; do not assume a checkpoint from a different implementation or template ordering is interchangeable.

## Acknowledgments

The reconstruction image-encoder design references [ShilongHong/DAGNet](https://github.com/ShilongHong/DAGNet), including FCEM, DVCA and CAFM. See the [reconstruction guide](template_nlf/README.md), [citation information](CITATION.md) and [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and upstream license information. The extraction network uses Microsoft HRNet, acknowledged in the third-party notices.

## Author

Wanxin Yu (yuwanxin@sjtu.edu.cn).

## License

Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn).

The project-owned code and modifications are available under the custom [Research-Only Noncommercial License](LICENSE): noncommercial scientific research only, with mandatory citation for publicly disseminated research results. Commercial use is not permitted under this license. Redistribution must preserve these conditions and all applicable notices. See [CITATION.md](CITATION.md) for the required manuscript reference.

This is a source-available research release, not an MIT-licensed or OSI-approved open-source project. Original Microsoft HRNet code retains its MIT license; see [third-party notices](THIRD_PARTY_NOTICES.md).
