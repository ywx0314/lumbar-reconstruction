# Third-party notices

## Microsoft HRNet

The HRNet implementation in `vertebra_extraction/models/hrnet/extraction_hrnet.py` and its configuration in `vertebra_extraction/models/hrnet/hrnet_config.py` derive from Microsoft HRNet code. Their original Microsoft copyright and author notices are retained.

Upstream repository: [HRNet Semantic Segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation).

Upstream license: [MIT License](https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/LICENSE). The complete applicable MIT text is included in [LICENSES/HRNet-MIT.txt](LICENSES/HRNet-MIT.txt).

The upstream repository separately identifies a BSD license for its `syncbn` component. That component is not included in this repository.

## DAGNet: reconstruction image encoder

The reconstruction image-encoder design references [ShilongHong/DAGNet](https://github.com/ShilongHong/DAGNet), including its frequency-domain enhancement, dual-view cross-attention and convolutional attention fusion concepts (FCEM, DVCA and CAFM). The project author confirms that this implementation was independently written from those architectural ideas, without copying DAGNet source code. The corresponding project implementation is in `template_nlf/models/image_encoder.py`, its `biplanar_encoder.py` wrapper and `template_nlf/models/cross_view/`, for AP/lateral vertebral feature extraction and template-based reconstruction.

Repository reference: ShilongHong. *DAGNet: A Dual-View Attention-Guided Network for Efficient X-ray Security Inspection* [software repository]. https://github.com/ShilongHong/DAGNet (accessed 2026-10-20).

This is an architectural acknowledgment; DAGNet source code is not included in this distribution. DAGNet retains its own authorship and licensing terms, described in its [upstream README](https://github.com/ShilongHong/DAGNet/blob/main/readme.md). The reference above identifies the repository because its README does not provide a complete paper citation. Wanxin Yu's copyright and research-only terms apply to this project's independently written implementation and do not claim ownership of DAGNet.

## Project code and dependencies

Code and modifications owned by Wanxin Yu are distributed under the custom [Research-Only Noncommercial License](LICENSE), Copyright (c) 2026 Wanxin Yu (yuwanxin@sjtu.edu.cn). Noncommercial scientific research and the citation requirements in [CITATION.md](CITATION.md) apply to those project-owned materials. They do not replace, revoke or restrict permissions independently granted for original Microsoft HRNet code under MIT. The two files containing HRNet adaptations identify both MIT for the upstream material and the custom license for project-owned modifications; their original Microsoft notices remain intact.

Runtime dependencies are installed separately from the component requirements files. Their packages retain their respective licenses; their source code is not bundled here. Datasets, template assets and trained weights are not included in this source distribution.
