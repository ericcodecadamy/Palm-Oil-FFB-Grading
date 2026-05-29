# Palm Oil FFB Ripeness Predictor

AI-powered classifier that detects the ripeness of palm oil **Fresh Fruit Bunches (FFB)** from a single photo. Supports five model architectures and is served as a web application via FastAPI.

**Classes:** Unripe · Ripe · Overripe

---

## Model Benchmarks

All models trained for up to 40 epochs with early stopping, two-phase fine-tuning (frozen backbone → full fine-tune), weighted loss, and RandomErasing augmentation.

| Model | Val Accuracy | Params | Train Time |
|---|---|---|---|
| **DINOv2** ⭐ | **86.15 %** | 86.6 M | 28.5 min |
| EfficientNet-B2 | 81.41 % | 7.7 M | 23.1 min |
| EfficientNet-B0 | 79.15 % | 4.0 M | 16.3 min |
| EfficientNet-B1 | 77.53 % | 6.5 M | 16.7 min |
| DINOv3 | 76.47 % | 21.6 M | 20.5 min |

> Best model: `facebook/dinov2-base` fine-tuned with a Dropout → Linear head on the CLS token.

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download training data

```bash
# Kaggle dataset (requires ~/.kaggle/kaggle.json)
python scripts/download_kaggle.py

# Roboflow datasets (requires ROBOFLOW_API_KEY)
export ROBOFLOW_API_KEY=your_key_here
python scripts/download_roboflow.py --all
```

### 3. Prepare the merged dataset

```bash
python scripts/prepare_dataset.py                  # Kaggle + Roboflow original
python scripts/prepare_dataset.py --all-roboflow   # include OXFWT + CMU datasets
```

### 4. Train

```bash
python scripts/train.py                            # EfficientNet-B0, 40 epochs
python scripts/train.py --model dinov2             # best accuracy
python scripts/train.py --model efficientnet_b2    # best size/accuracy tradeoff
```

To compare all models:

```bash
python scripts/compare_models.py
```

### 5. Serve

```bash
cp models/<best_model>/best_model.pth models/best_model.pth
python run.py
```

Open **http://localhost:8000** in your browser.

---

## Data Sources

| Source | Description |
|---|---|
| [Kaggle – ramadanizikri112](https://www.kaggle.com/datasets/ramadanizikri112/ripeness-of-oil-palm-fruit) | Malay-labelled FFB images |
| [Roboflow – palm-fruit-classification](https://universe.roboflow.com/palm-fruit-classification/palm-fruit-ripeness-classificationcnn) | Original Roboflow FFB dataset |
| [Roboflow – OXFWT](https://universe.roboflow.com/palmoilobjectdetectionclassification/palm-oil-ripeness-oxfwt) | Object-detection style dataset |
| [Roboflow – CMU](https://universe.roboflow.com/chiang-mai-university-skeej/ripeness-ffb-palm-oil) | Chiang Mai University FFB dataset |

> **Note:** Training data is **not included** in this repository due to licensing restrictions. Download via the scripts above.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/predict` | Upload image → JSON prediction |
| `POST` | `/predict-base64` | Webcam JSON `{ "image": "<base64>" }` |
| `GET` | `/health` | Model load status |
| `GET` | `/model-info` | Architecture, accuracy, class names |

### Example

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@bunch.jpg"
```

```json
{
  "prediction": "ripe",
  "confidence": 0.923,
  "label": "Ripe",
  "description": "The FFB is ready for harvest! Optimal oil content.",
  "all_scores": { "overripe": 0.041, "ripe": 0.923, "unripe": 0.036 }
}
```

---

## Supported Models

| Key | Architecture | Input | Notes |
|---|---|---|---|
| `efficientnet_b0` | EfficientNet-B0 | 224×224 | Fastest; default |
| `efficientnet_b1` | EfficientNet-B1 | 240×240 | Slightly larger |
| `efficientnet_b2` | EfficientNet-B2 | 260×260 | Best of B-series |
| `dinov2` | DINOv2-Base (ViT) | 224×224 | Highest accuracy; HuggingFace `facebook/dinov2-base` |
| `dinov3` | DINOv3-ViT-S/16 | 224×224 | Gated model — requires Meta approval + `huggingface-cli login` |

---

## Project Structure

```
palm_oil_ffb_predictor/
├── app.py                    # FastAPI application & inference engine
├── run.py                    # Launcher (resolves local IP, starts uvicorn)
├── requirements.txt
├── static/
│   └── index.html            # Single-file web UI (no build step)
├── scripts/
│   ├── train.py              # Train a single model
│   ├── compare_models.py     # Benchmark all architectures sequentially
│   ├── prepare_dataset.py    # Merge & split datasets into train/val
│   ├── download_kaggle.py    # Download Kaggle dataset
│   └── download_roboflow.py  # Download Roboflow datasets
└── models/                   # Saved checkpoints & plots (*.pth excluded from git)
```

---

## Security Notes

- **API keys** are read from environment variables (`ROBOFLOW_API_KEY`, `KAGGLE_USERNAME` / `KAGGLE_KEY`, `HF_TOKEN`). Never hard-coded.
- **Model weights** (`*.pth`) are excluded from git. Distribute via cloud storage or Git LFS.
- **Training data** is excluded from git (licensed; download via scripts).
- The FastAPI app validates file content type and handles malformed images safely.

---

## Class Imbalance Handling

Three strategies are applied simultaneously during training:

1. **`WeightedRandomSampler`** — balanced mini-batches from the first epoch
2. **`CrossEntropyLoss(weight=...)`** — heavier gradient signal for minority class
3. **`PerClassAugDataset`** — stronger augmentation (colour jitter, affine, RandomErasing) applied only to the minority class

---

## Environment Variables

| Variable | Purpose |
|---|---|
| `ROBOFLOW_API_KEY` | Roboflow dataset download |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle dataset download (alternative to `kaggle.json`) |
| `HF_TOKEN` | HuggingFace token for gated models (DINOv3) |

---

## License

Source code: MIT.  
Training data and model weights are subject to their respective upstream licenses (Kaggle, Roboflow, Meta).
