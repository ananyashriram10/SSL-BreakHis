# SSL-BreakHis

Hybrid self-supervised learning for breast cancer subtype classification on the
[BreakHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
histopathology dataset.

## What Was Fixed

The original notebook flow had several issues that can hurt results or make the
reported results unreliable:

- Image-level train/validation/test splits leaked images from the same patient
  across splits. The new pipeline splits by patient ID.
- SimMIM created a mask but encoded the unmasked image. The fixed model feeds the
  masked image to the encoder and computes loss only on masked pixels.
- SimMIM pixel accuracy used the visible region instead of the masked region.
  The fixed training objective reports masked reconstruction loss instead.
- Validation code divided by `len(val_loader)` even when evaluating another
  loader.
- SimMIM encoder weights were loaded into the BYOL backbone with `strict=False`,
  which could silently skip mismatched keys. The new loaders fail loudly if the
  encoder weights do not load.
- Downstream training now saves the best validation macro-F1 checkpoint and tests
  that checkpoint, instead of testing whichever weights happen to be left after
  the last epoch.
- Downstream training uses normalization, class-weighted cross entropy, macro-F1,
  a classification report, and a confusion matrix.

## Files

| Path | Purpose |
| --- | --- |
| `ssl_breakhis/data.py` | Dataset scanning, patient-level splitting, labels, class weights |
| `ssl_breakhis/models.py` | SimMIM, BYOL, ResNet classifier, checkpoint loaders |
| `ssl_breakhis/transforms.py` | SSL and downstream image transforms |
| `scripts/pretrain_simmim.py` | Stage 1 masked-image pretraining |
| `scripts/pretrain_byol.py` | Stage 2 BYOL continuation pretraining |
| `scripts/train_downstream.py` | Fine-tuning and final evaluation |
| `*.ipynb` | Original notebooks, kept for reference |

## Setup

```bash
pip install -r requirements.txt
```

Set your BreakHis root to the folder that contains:

```text
benign/SOB/...
malignant/SOB/...
```

For Kaggle this is usually:

```bash
DATA_ROOT=/kaggle/input/breakhis/BreaKHis_v1/BreaKHis_v1/histology_slides/breast
```

## Recommended Run

Run from the repository root.

```bash
python scripts/pretrain_simmim.py \
  --data-root "$DATA_ROOT" \
  --out checkpoints/best_simmim_resnet18.pth \
  --epochs 100
```

```bash
python scripts/pretrain_byol.py \
  --data-root "$DATA_ROOT" \
  --simmim-checkpoint checkpoints/best_simmim_resnet18.pth \
  --out checkpoints/best_byol_resnet18.pth \
  --epochs 100
```

```bash
python scripts/train_downstream.py \
  --data-root "$DATA_ROOT" \
  --checkpoint checkpoints/best_byol_resnet18.pth \
  --task subtype \
  --out checkpoints/best_downstream_subtype.pth \
  --epochs 80
```

For benign vs malignant classification:

```bash
python scripts/train_downstream.py \
  --data-root "$DATA_ROOT" \
  --checkpoint checkpoints/best_byol_resnet18.pth \
  --task binary \
  --out checkpoints/best_downstream_binary.pth
```

For one magnification:

```bash
python scripts/train_downstream.py \
  --data-root "$DATA_ROOT" \
  --checkpoint checkpoints/best_byol_resnet18.pth \
  --task subtype \
  --magnification 400X
```

## Notes For Better Results

Patient-safe splits are stricter than random image splits, so accuracy can drop
at first while becoming more trustworthy. Tune against validation macro-F1, not
test accuracy. Good next experiments are longer SSL pretraining, per-magnification
fine-tuning, ImageNet initialization for the downstream baseline, and reporting
mean/std across several seeds.
