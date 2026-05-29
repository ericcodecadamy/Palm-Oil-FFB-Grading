"""
Palm Oil FFB Ripeness Predictor — FastAPI Backend
──────────────────────────────────────────────────
Endpoints:
  GET  /             → Serve web UI
  POST /predict      → Predict ripeness from uploaded image (JSON response)
  GET  /health       → Health check + model status
  GET  /model-info   → Model metadata (classes, accuracy, etc.)

Run with:
  python run.py
  or directly: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import io
import os
import json
import base64
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "models" / "best_model.pth"
STATIC_DIR   = BASE_DIR / "static"
INDEX_HTML   = STATIC_DIR / "index.html"

# ─── Constants ────────────────────────────────────────────────────────────────
DEFAULT_CLASSES = ["unripe", "ripe", "overripe"]
IMG_SIZE        = 224
MEAN            = [0.485, 0.456, 0.406]
STD             = [0.229, 0.224, 0.225]

# ─── Class descriptions & colors ──────────────────────────────────────────────
CLASS_INFO = {
    "unripe": {
        "label":       "Unripe",
        "emoji":       "🟢",
        "description": "The FFB is not ready for harvest. Continue to monitor.",
        "color":       "#22c55e",
    },
    "ripe": {
        "label":       "Ripe",
        "emoji":       "🟡",
        "description": "The FFB is ready for harvest! Optimal oil content.",
        "color":       "#f59e0b",
    },
    "overripe": {
        "label":       "Overripe",
        "emoji":       "🔴",
        "description": "The FFB is past peak ripeness. Harvest immediately to avoid oil loss.",
        "color":       "#ef4444",
    },
}

# ─── Model Builder ───────────────────────────────────────────────────────────
def _build_inference_model(model_name: str, num_classes: int,
                           checkpoint: dict) -> "torch.nn.Module":
    """Rebuild the model architecture and load weights from checkpoint."""
    if model_name in ("efficientnet_b0", "efficientnet_b1", "efficientnet_b2"):
        builders = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
        }
        m = builders[model_name](weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
        m.load_state_dict(checkpoint["model_state"])
        return m

    elif model_name == "dinov2":
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError(
                "transformers not installed — needed for DINOv2. "
                "Run: pip install transformers accelerate"
            )
        backbone_name = checkpoint.get("dinov2_backbone", "facebook/dinov2-base")

        class _DINOv2(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone   = AutoModel.from_pretrained(backbone_name)
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(self.backbone.config.hidden_size, num_classes),
                )
            def forward(self, x):
                out = self.backbone(pixel_values=x)
                return self.classifier(out.last_hidden_state[:, 0])

        m = _DINOv2()
        m.load_state_dict(checkpoint["model_state"])
        return m

    elif model_name == "dinov3":
        try:
            from transformers import AutoModel
        except ImportError:
            raise RuntimeError(
                "transformers not installed — needed for DINOv3. "
                "Run: pip install transformers accelerate"
            )
        backbone_name = checkpoint.get("dinov3_backbone",
                                       "facebook/dinov3-vits16-pretrain-lvd1689m")

        class _DINOv3(nn.Module):
            def __init__(self):
                super().__init__()
                try:
                    self.backbone = AutoModel.from_pretrained(backbone_name)
                except Exception as e:
                    if "gated" in str(e).lower() or "401" in str(e):
                        raise RuntimeError(
                            f"DINOv3 requires Meta's permission (gated model).\n"
                            f"  1. Accept terms: https://huggingface.co/{backbone_name}\n"
                            f"  2. Run: huggingface-cli login"
                        ) from None
                    raise
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.3),
                    nn.Linear(self.backbone.config.hidden_size, num_classes),
                )
            def forward(self, x):
                out = self.backbone(pixel_values=x)
                return self.classifier(out.last_hidden_state[:, 0])

        m = _DINOv3()
        m.load_state_dict(checkpoint["model_state"])
        return m

    else:
        # Fallback: try EfficientNet-B0 for old checkpoints without model_name
        log.warning(f"Unknown model_name '{model_name}' in checkpoint — falling back to efficientnet_b0")
        m = models.efficientnet_b0(weights=None)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
        m.load_state_dict(checkpoint["model_state"])
        return m


# ─── Model Loading ────────────────────────────────────────────────────────────
class FFBPredictor:
    def __init__(self):
        self.model       = None
        self.device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = DEFAULT_CLASSES
        self.img_size    = IMG_SIZE
        self.val_acc     = None
        self.loaded      = False
        self.error_msg   = None

        self._transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

    def load(self) -> bool:
        if not MODEL_PATH.exists():
            self.error_msg = (
                f"Model not found at '{MODEL_PATH}'. "
                "Please train the model first: python scripts/train.py"
            )
            log.warning(self.error_msg)
            return False

        try:
            log.info(f"Loading model from {MODEL_PATH} on {self.device} ...")
            checkpoint = torch.load(MODEL_PATH, map_location=self.device, weights_only=False)

            self.class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
            self.img_size    = checkpoint.get("img_size",    IMG_SIZE)
            self.val_acc     = checkpoint.get("val_acc",     None)
            model_name       = checkpoint.get("model_name",  "efficientnet_b0")

            num_classes = len(self.class_names)
            model = _build_inference_model(model_name, num_classes, checkpoint)
            model.eval()
            model.to(self.device)

            self._transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ])

            self.model  = model
            self.loaded = True

            acc_str = f"{self.val_acc:.4f}" if self.val_acc else "N/A"
            log.info(f"✅ Model loaded — arch: {model_name}  classes: {self.class_names}  val_acc: {acc_str}")
            return True

        except Exception as e:
            self.error_msg = f"Failed to load model: {e}"
            log.error(self.error_msg)
            return False

    @torch.no_grad()
    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes. Returns prediction dict."""
        if not self.loaded or self.model is None:
            raise RuntimeError(self.error_msg or "Model not loaded.")

        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except UnidentifiedImageError:
            raise ValueError("Cannot decode image. Please upload a valid JPG or PNG.")

        tensor = self._transform(img).unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            logits = self.model(tensor)

        probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        pred_idx  = int(probs.argmax())
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        all_scores = {
            cls: float(p) for cls, p in zip(self.class_names, probs)
        }

        info = CLASS_INFO.get(pred_class, {})

        return {
            "prediction":   pred_class,
            "confidence":   confidence,
            "label":        info.get("label",       pred_class.title()),
            "emoji":        info.get("emoji",        ""),
            "description":  info.get("description", ""),
            "color":        info.get("color",       "#6b7280"),
            "all_scores":   all_scores,
            "class_info":   {
                cls: CLASS_INFO.get(cls, {"label": cls, "color": "#6b7280"})
                for cls in self.class_names
            },
        }


# ─── App & Predictor Singleton ────────────────────────────────────────────────
predictor = FFBPredictor()

app = FastAPI(
    title       = "Palm Oil FFB Ripeness Predictor",
    description = "AI-powered palm oil Fresh Fruit Bunch ripeness detection",
    version     = "1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Startup ──────────────────────────────────────────────────────────────────
def _maybe_download_model() -> None:
    """Download model from Hugging Face Hub if not present locally.

    Set HF_MODEL_REPO=username/repo-name in the environment (e.g. on Render).
    Optional: set HF_TOKEN for private repos.
    """
    if MODEL_PATH.exists():
        return

    hf_repo = os.getenv("HF_MODEL_REPO", "")
    if not hf_repo:
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.warning("huggingface_hub not installed — cannot auto-download model.")
        return

    log.info(f"Model not found locally. Downloading from HF Hub: {hf_repo} ...")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        hf_hub_download(
            repo_id=hf_repo,
            filename="best_model.pth",
            local_dir=str(MODEL_PATH.parent),
            token=os.getenv("HF_TOKEN") or None,
        )
        log.info(f"Model downloaded to {MODEL_PATH}")
    except Exception as e:
        log.error(f"Failed to download model from HF Hub: {e}")


@app.on_event("startup")
async def startup_event():
    _maybe_download_model()
    predictor.load()
    log.info("🚀 Server ready.")


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404,
                            detail="Frontend not found. Check static/index.html")
    return HTMLResponse(content=INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  predictor.loaded,
        "model_error":   predictor.error_msg,
        "device":        str(predictor.device),
        "classes":       predictor.class_names,
    }


@app.get("/model-info")
async def model_info():
    meta_path = BASE_DIR / "models" / "training_meta.json"
    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
    model_name = meta.get("model_name") if meta else None
    return {
        "loaded":      predictor.loaded,
        "model_name":  model_name,
        "val_acc":     predictor.val_acc,
        "classes":     predictor.class_names,
        "device":      str(predictor.device),
        "model_path":  str(MODEL_PATH),
        "training":    meta,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an image (JPG/PNG) and return ripeness prediction.
    Supports both file upload and base64-encoded images (for webcam frames).
    """
    if not predictor.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "error":   "Model not loaded",
                "message": predictor.error_msg or "Train the model first.",
                "fix":     "Run: python scripts/train.py",
            },
        )

    content_type = file.content_type or ""
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty file received.")

        result = predictor.predict(image_bytes)
        return JSONResponse(content=result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        log.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.post("/predict-base64")
async def predict_base64(request: Request):
    """Accept a JSON body with { 'image': '<base64 string>' } for webcam frames."""
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    body = await request.json()
    b64  = body.get("image", "")
    if not b64:
        raise HTTPException(status_code=400, detail="No image data provided.")

    # Strip data URL prefix if present
    if "," in b64:
        b64 = b64.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(b64)
        result      = predictor.predict(image_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        log.exception("Error in base64 prediction")
        raise HTTPException(status_code=400, detail=f"Could not process image: {e}")
