# vertebra_extraction: domain-adapted vertebra extraction

An HRNet-W48 encoder and progressive extraction decoder map one grayscale radiograph to five intensity images for L1–L5. A two-class feature discriminator is trained through gradient reversal using labeled synthetic DRRs and unlabeled target radiographs.

## Setup

Use Python 3.11 or newer. All commands below run from this directory:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

No data or trained weights are bundled.

## Data layout

```text
data/
  Xray/<id>AP.jpg
  Xray/<id>ML.jpg
  Mask_L1/<id>AP.jpg
  Mask_L1/<id>ML.jpg
  ...
  Mask_L5/<id>AP.jpg
  Mask_L5/<id>ML.jpg
splits/
  source_train.txt
  target_train.txt
```

Each list contains one anonymous sample ID per line, without an extension. The selected source view requires an input and all five labels. Target samples require only input images. Keep training subjects separate from evaluation subjects.

Despite the `Mask_L*` folder names, labels are extracted vertebral intensity images, not binary masks. Grayscale values are normalized by dividing by 255. Source inputs and labels are cropped or zero-padded using their joint lumbar label region. Target inputs must already be cropped or padded around L1–L5 to the configured size, normally 512×512.

## Training

```sh
python train.py --config configs/train.toml
```

Configuration paths are resolved relative to this component when using the supplied `configs/train.toml`. Use `plane = "AP"` or `"ML"`; train the selected view with a separate output directory.

Defaults: source and target batch size 32, 200 epochs, AdamW learning rates 0.001 for the generator and 0.0001 for the discriminator, weight decay 0.001. The generator learning rate is multiplied by 0.9 every 20 epochs.

The extraction loss is `60 * (1 - NCC) + 15000 * MSE`, averaging pixels, samples and channels. The domain objective averages source and target cross-entropy losses equally. Its external weight is 1; the GRL applies `2/(1+exp(-10*p))-1` once to feature-network gradients. Discriminator gradients are not reversed.

Source augmentation uses horizontal flipping with probability 0.5, edge occlusion with probability 0.5 and width up to 15%, and intensity augmentation with probability 0.8. Intensity augmentation includes brightness shifts up to 0.2, contrast 0.8–1.2, gamma 0.8–1.5 and Gaussian noise standard deviation 0–0.03.

Checkpoints are saved every 20 epochs by default. The trainer writes model weights, epoch and configuration; it does not implement optimizer-state resume, validation during training or TensorBoard logging.

## Inference

```sh
python predict.py --checkpoint outputs/AP/epoch_200.pt --image prepared_AP.png --output results/case001/AP
```

The input must be a prepared 512×512 image. Outputs are `L1.png` through `L5.png`. Apply the appropriate trained model to each view.

For reconstruction, place each level's paired outputs under the names `<id>AP.png` and `<id>ML.png`. Preserve the coordinate relationship between the prepared images and camera calibration; image transforms are not transferred automatically.

## Test-set evaluation

Set the held-out ID list, checkpoint and output directory in `configs/evaluate.toml`, then run:

```sh
python evaluate.py --config configs/evaluate.toml
```

This command loads the checkpoint, predicts all listed radiographs without augmentation, and writes per-case Dice/PSNR/SSIM to `metrics.csv`. It also writes `summary.json` and five extracted PNG channels per case under `extracted/<id>/`. The final partial batch is included. Scores use float predictions before PNG quantization. Nonfinite PSNR is represented as a string in the JSON summary.

The same crop/pad operation is applied jointly to inputs and intensity labels. Set `plane` to `AP` or `ML` for the desired view. Training and evaluation require independently prepared datasets and ID lists; no patient data are included.

## Evaluate already saved images

```sh
python evaluate_images.py --prediction results/case001/AP --reference reference/case001/AP
```

Both folders must contain `L1.png` through `L5.png` in the same image frame. The script averages per-channel Dice, PSNR and SSIM. Dice uses an intensity threshold of 0.005 by default, adjustable with `--threshold`. Identical images have infinite PSNR because their mean squared error is zero.

## License

See the repository [Research-Only Noncommercial License](../LICENSE), [third-party notices](../THIRD_PARTY_NOTICES.md) and retained [HRNet MIT text](../LICENSES/HRNet-MIT.txt).

## Source organization

`train.py` performs domain-adversarial training. `predict.py` loads a checkpoint and extracts five channels from one image. `evaluate.py` evaluates a labeled test set; `evaluate_images.py` scores already exported channels. `datasets.py` contains source/target datasets, and `models/` contains the HRNet extraction model, feature-domain classifier and NCC loss.

The full training/prediction/evaluation path is included. Data, extraction targets, pre-trained weights and the CT-to-DRR generation pipeline are supplied separately. Optimizer-state resume and training-time validation are not implemented.

Project-owned code and modifications are limited to noncommercial scientific research. Publicly disseminated research results must include the manuscript reference in [CITATION.md](../CITATION.md). Commercial use is not permitted under the project license.
