# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered palm oil Fresh Fruit Bunch (FFB) ripeness classifier. Classifies FFB images into three classes: **unripe**, **ripe**, **overripe**. Supports five model architectures and is served via a FastAPI web app.

## Environment Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the App

```bash
python run.py                          # default: 0.0.0.0:8000
python run.py --port 8080
python run.py --reload                 # dev mode with auto-reload
```

The app serves the web UI at `/` and exposes `/predict` (file upload), `/predict-base64` (webcam JSON), `/health`, and `/model-info`.

## Full Pipeline: Data → Train → Serve

```bash
# 1. Download data
python scripts/download_kaggle.py
python scripts/download_roboflow.py --api-key YOUR_KEY --all   # downloads all 3 Roboflow datasets

# 2. Merge and split into train/val
python scripts/prepare_dataset.py                              # Kaggle + Roboflow original only
python scripts/prepare_dataset.py --all-roboflow               # include oxfwt + CMU datasets

# 3a. Train a single model
python scripts/train.py                                        # default: efficientnet_b0, 25 epochs
python scripts/train.py --model dinov2 --epochs 20
python scripts/train.py --model efficientnet_b2 --models-dir models/b2

# 3b. Compare all models (trains sequentially, takes hours)
python scripts/compare_models.py
python scripts/compare_models.py --epochs 15 --models efficientnet_b0 efficientnet_b2 dinov2 dinov3

# 4. Serve the best model
cp models/<best_model_name>/best_model.pth models/best_model.pth
python run.py
```

## Supported Models

| Model | Input | Params | Notes |
|-------|-------|--------|-------|
| `efficientnet_b0` | 224×224 | ~5 M | default; fast |
| `efficientnet_b1` | 240×240 | ~7.8 M | slightly larger |
| `efficientnet_b2` | 260×260 | ~9.1 M | best of B-series |
| `dinov2` | 224×224 | ~86 M | HuggingFace `facebook/dinov2-base`; LR 5e-5 |
| `dinov3` | 224×224 | ~22 M | HuggingFace `facebook/dinov3-vits16-pretrain-lvd1689m`; LR 5e-5 |

Default LR for EfficientNet is 3e-4; for DINO models 5e-5 (set automatically unless `--lr` is given).

## Data Sources

| Key | URL | Output dir |
|-----|-----|-----------|
| Kaggle | `ramadanizikri112/ripeness-of-oil-palm-fruit` | `data/kaggle` |
| `original` | `palm-fruit-classification/palm-fruit-ripeness-classificationcnn` | `data/roboflow` |
| `oxfwt` | `palmoilobjectdetectionclassification/palm-oil-ripeness-oxfwt` | `data/roboflow_oxfwt` |
| `cmu` | `chiang-mai-university-skeej/ripeness-ffb-palm-oil` | `data/roboflow_cmu` |

Kaggle requires `~/.kaggle/kaggle.json` or `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars.
Roboflow requires `ROBOFLOW_API_KEY` or `--api-key`. If download of oxfwt fails, try `--version 2`.

After downloading new Roboflow datasets, run `prepare_dataset.py` and check for `[ UNKNOWN]` warnings — add any unrecognised folder names to `CLASS_MAP` in `scripts/prepare_dataset.py`.

## Architecture

**`app.py`** — FastAPI application. `FFBPredictor` is a singleton that loads the checkpoint at startup via `_build_inference_model()`, which dispatches on `checkpoint["model_name"]` to reconstruct the correct architecture. The transform (`_transform`) is rebuilt to match the checkpoint's `img_size`.

**`run.py`** — Thin launcher that resolves the local IP, checks model existence, and delegates to uvicorn via subprocess.

**`scripts/train.py`** — Trains a single model. Three class-imbalance strategies applied for all architectures:
1. `WeightedRandomSampler` — balanced batches
2. `CrossEntropyLoss(weight=...)` — heavier loss on minority class
3. `PerClassAugDataset` — stronger augmentation on the minority class (auto-detected)

For DINOv2/DINOv3: the CLS token (`last_hidden_state[:, 0]`) feeds a `Dropout → Linear` head.

**`scripts/compare_models.py`** — Trains all selected models sequentially, collects `val_acc / training_time / params`, and writes `models/comparison.json`, `comparison.txt`, `comparison.png`.

**`scripts/prepare_dataset.py`** — Merges all data sources via `CLASS_MAP` (handles Malay names, Roboflow variants). `empty_bunch` images are excluded. Use `--all-roboflow` to include the two additional datasets.

**`static/index.html`** — Single-file frontend (no build step). Served directly by FastAPI.

## Model Checkpoint Format

`best_model.pth` keys: `model_name`, `model_state`, `class_names`, `img_size`, `val_acc`, `epoch`, `mean`, `std`. DINO checkpoints also include `dinov2_backbone` or `dinov3_backbone` (the HuggingFace model ID string).

Class order is alphabetical from `ImageFolder` → `["overripe", "ripe", "unripe"]`, which differs from the display order in `CLASS_INFO`.

Old checkpoints without `model_name` default to `efficientnet_b0` in `_build_inference_model`.
