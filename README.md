# Flow Mismatching

Official implementation of the **NeurIPS 2026 paper**

> **Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models**

[[arXiv](https://arxiv.org/abs/2605.23070)]

## Overview

This repository provides the official implementation of **Flow Mismatching**, an unsupervised anomaly detection method that does not rely on reconstruction.

A flow matching model is trained only on normal images. To score a test image, we draw affine paths from Gaussian noise to the image and compare, along each path, the velocity predicted by the model with the geometric velocity toward the image. Anomalies are where the two velocities disagree. Aggregating the mismatch over time steps and paths gives pixel-level heatmaps and image-level scores with forward passes only: no test-time optimization, memory bank, or pretrained feature extractor.

## Quick Start

The scoring rule is implemented in:

```text
flow_mismatching/scoring.py
```

Scoring one image with a trained model:

```python
import torch
from PIL import Image

from flow_mismatching import anomaly_heatmap, gaussian_smooth, image_score
from flow_mismatching.data import NORM_STATS, image_transform
from flow_mismatching.utils import load_model

device = torch.device("cuda")
model, meta = load_model("runs/mvtec/ckpt_epoch2000.pt", device)

transform = image_transform(128, *NORM_STATS["mvtec"])
image = transform(Image.open("bottle.png").convert("RGB"))[None].to(device)  # [1, 3, 128, 128]
class_idx = meta["class_names"].index("bottle")

heatmap = anomaly_heatmap(model, image, num_paths=10, num_steps=10, class_idx=class_idx)
heatmap = gaussian_smooth(heatmap, ksize=9, sigma=3.0)  # [128, 128] anomaly map
score = image_score(heatmap)                             # image-level anomaly score
```

## Code Structure

```text
train.py                  # train the class-conditional flow matching model
evaluate.py               # heatmaps, image scores and metrics on MVTec-AD / VisA
flow_mismatching/
├── scoring.py            # velocity mismatch heatmap, Gaussian smoothing, image score
├── metrics.py            # AUROC, AP, F1-max, AUPRO
├── data.py               # datasets in the MVTec-AD folder layout
├── unet.py               # U-Net with attention (adapted from facebookresearch/flow_matching)
├── optim.py              # StableAdamW optimizer
└── utils.py              # checkpoints, mixed precision, visualization
```

## Installation

```bash
git clone https://github.com/ShengzheC/Flow-Mismatching.git
cd Flow-Mismatching
pip install -r requirements.txt  # Python 3.9+, PyTorch 2.3+
```

## Data

**MVTec-AD.** Download it from the [official website](https://www.mvtec.com/company/research/datasets/mvtec-ad) and extract it to `data/mvtec`.

**VisA.** Download it from the [official repository](https://github.com/amazon-science/spot-diff) and convert it to the MVTec-AD layout with their script (1-class setup), which also binarizes the masks:

```bash
python ./utils/prepare_data.py --split-type 1cls --data-folder ./VisA --save-folder ./data/visa --split-file ./split_csv/1cls.csv
```

`--data_root` is the folder that contains the class folders (`data/mvtec` or `data/visa/1cls`):

```text
<data_root>/<class>/train/good/*.png
<data_root>/<class>/test/good/*.png
<data_root>/<class>/test/<defect type>/*.png
<data_root>/<class>/ground_truth/<defect type>/*.png
```

## Usage

### 1. Training

A single class-conditional model is trained for each dataset on the normal images of all its classes:

```bash
python train.py --dataset mvtec --data_root data/mvtec
python train.py --dataset visa --data_root data/visa/1cls
```

For several GPUs, launch the same command with `torchrun --nproc_per_node <num_gpus> train.py ...`. The defaults follow the paper: 128x128 images, 2000 epochs, batch size 32 per GPU, StableAdamW with AMSGrad (learning rate 1e-4, weight decay 1e-4), bf16 mixed precision. A checkpoint and a grid of generated samples are saved to `runs/<dataset>/` every 500 epochs, and `--resume` continues from the latest checkpoint.

### 2. Evaluation

```bash
python evaluate.py --dataset mvtec --data_root data/mvtec --ckpt runs/mvtec/ckpt_epoch2000.pt
python evaluate.py --dataset visa --data_root data/visa/1cls --ckpt runs/visa/ckpt_epoch2000.pt
```

Main options (defaults in parentheses):

| Option | Description |
| --- | --- |
| `--num_paths` (10) | number of noise paths K per image |
| `--num_steps` (10) | number of time steps T, evenly spaced inside (0, 1) |
| `--aggregation` (min) | aggregation over the paths, `min` or `mean` |
| `--power` (2) | time weight w(t) = t^r |
| `--gauss_ksize` (9), `--gauss_sigma` (3) | Gaussian smoothing of the heatmap |
| `--topk_ratio` (0.01), `--lam` (0.5) | image score = lam * max + (1 - lam) * mean of the top 1% pixels |
| `--classes` | evaluate only the given classes, e.g. `--classes bottle cable` |
| `--save_vis` | save image / heatmap / ground-truth figures |

The noise paths are drawn from a generator seeded with `--seed` (reset for each class), so results are reproducible.

### 3. Outputs

`evaluate.py` prints the per-class results and writes to `results/<dataset>/`:

```text
per_image.csv           # image score and pixel metrics of every test image
metrics_pooled.csv      # per-class metrics, pixels pooled over each class, and the class mean
metrics_per_image.csv   # per-class metrics, pixel metrics averaged over anomalous images
config.json             # arguments of the run
vis/<class>/*.png       # figures, with --save_vis
```

The metrics are image-level AUROC, AP and F1-max, and pixel-level AUROC, AP, F1-max and AUPRO (normalized area under the PRO curve up to 30% FPR).

## Citation

```bibtex
@inproceedings{chen2026flowmismatching,
  title     = {Flow Mismatching: Unsupervised Anomaly Detection via Velocity Discrepancies in Flow Matching Models},
  author    = {Chen, Shengzhe and Moradi, Mehrdad and Paynabar, Kamran and Yan, Hao},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## Acknowledgements

The U-Net is adapted from [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching), which builds on [openai/guided-diffusion](https://github.com/openai/guided-diffusion). Training samples are generated with the ODE solver of the `flow_matching` library.

## License

This repository is released under the [MIT License](LICENSE), except `flow_mismatching/unet.py`, which is adapted from facebookresearch/flow_matching and remains under the [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) license.
