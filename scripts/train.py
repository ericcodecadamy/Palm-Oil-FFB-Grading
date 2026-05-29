"""
Train Palm Oil FFB Ripeness Classifier

Supported models (--model):
  efficientnet_b0  — EfficientNet-B0  224×224  ~5 M params  [default]
  efficientnet_b1  — EfficientNet-B1  240×240  ~7.8 M params
  efficientnet_b2  — EfficientNet-B2  260×260  ~9.1 M params
  dinov2           — DINOv2-Base ViT  224×224  ~86 M params (HuggingFace)
  dinov3           — DINOv3-ViT-S/16  224×224  ~22 M params (HuggingFace)

Three class-imbalance strategies (applied to all models):
  1. WeightedRandomSampler  — balanced batches
  2. Weighted CrossEntropyLoss — minority errors cost more
  3. PerClassAugDataset — heavier augmentation on the smallest class

Output: {models-dir}/best_model.pth  +  training_meta.json  +  plots

Usage:
  python scripts/train.py
  python scripts/train.py --model dinov2 --epochs 20
  python scripts/train.py --model efficientnet_b2 --models-dir models/b2
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision import datasets, transforms, models
from torchvision.models import (
    EfficientNet_B0_Weights,
    EfficientNet_B1_Weights,
    EfficientNet_B2_Weights,
)
from PIL import Image

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm


# ─── CONFIG ───────────────────────────────────────────────────────────────────
CLASSES = ["unripe", "ripe", "overripe"]
MEAN    = [0.485, 0.456, 0.406]
STD     = [0.229, 0.224, 0.225]

MODEL_CONFIGS: dict[str, dict] = {
    "efficientnet_b0": {"img_size": 224, "default_lr": 3e-4},
    "efficientnet_b1": {"img_size": 240, "default_lr": 3e-4},
    "efficientnet_b2": {"img_size": 260, "default_lr": 3e-4},
    "dinov2":          {"img_size": 224, "default_lr": 5e-5},
    "dinov3":          {"img_size": 224, "default_lr": 5e-5},
}
# ──────────────────────────────────────────────────────────────────────────────


# ==============================================================================
# Transforms  (model-size-aware)
# ==============================================================================

def get_majority_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.02),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
    ])


def get_minority_transform(img_size: int) -> transforms.Compose:
    """Heavier augmentation for the underrepresented class."""
    return transforms.Compose([
        transforms.Resize((img_size + 48, img_size + 48)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.10),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.4),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.35, scale=(0.02, 0.25)),
    ])


def get_val_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ==============================================================================
# Dataset
# ==============================================================================

class PerClassAugDataset(Dataset):
    """
    Wraps ImageFolder and applies stronger augmentation to the minority class
    (whichever class has the fewest training samples).
    """

    def __init__(self, root, minority_class_idx: int, img_size: int):
        self.base               = datasets.ImageFolder(str(root))
        self.minority_idx       = minority_class_idx
        self.majority_transform = get_majority_transform(img_size)
        self.minority_transform = get_minority_transform(img_size)
        self.classes            = self.base.classes
        self.class_to_idx       = self.base.class_to_idx
        self.targets            = self.base.targets
        self.samples            = self.base.samples

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx):
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        t   = self.minority_transform if label == self.minority_idx else self.majority_transform
        return t(img), label


# ==============================================================================
# Imbalance helpers
# ==============================================================================

def make_weighted_sampler(targets: list[int]) -> WeightedRandomSampler:
    counts        = Counter(targets)
    total         = len(targets)
    n_cls         = len(counts)
    class_weights = {c: total / (n_cls * cnt) for c, cnt in counts.items()}
    sample_w      = [class_weights[t] for t in targets]
    return WeightedRandomSampler(weights=sample_w, num_samples=len(sample_w), replacement=True)


def compute_loss_weights(targets: list[int], num_classes: int,
                         device: torch.device) -> torch.Tensor:
    counts = Counter(targets)
    total  = len(targets)
    return torch.tensor(
        [total / (num_classes * counts[c]) for c in range(num_classes)],
        dtype=torch.float32, device=device,
    )


# ==============================================================================
# Model definitions
# ==============================================================================

class DINOv2Classifier(nn.Module):
    """DINOv2-Base backbone with a linear classification head."""

    BACKBONE = "facebook/dinov2-base"

    def __init__(self, num_classes: int):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers accelerate"
            )
        self.backbone   = AutoModel.from_pretrained(self.BACKBONE)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.backbone.config.hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        return self.classifier(out.last_hidden_state[:, 0])  # CLS token


class DINOv3Classifier(nn.Module):
    """DINOv3-ViT-S/16 backbone with a linear classification head.

    The model is gated on HuggingFace (Meta terms agreement required).
    One-time setup:
      1. Visit https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
         and click "Agree and access repository"
      2. Run: huggingface-cli login   (paste your HF token from hf.co/settings/tokens)
    """

    BACKBONE = "facebook/dinov3-vits16-pretrain-lvd1689m"

    def __init__(self, num_classes: int):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError:
            raise ImportError(
                "transformers not installed. Run: pip install transformers accelerate"
            )
        try:
            self.backbone = AutoModel.from_pretrained(self.BACKBONE)
        except Exception as e:
            if "gated" in str(e).lower() or "401" in str(e):
                raise RuntimeError(
                    f"\n\nDINOv3 requires Meta's permission (gated model).\n"
                    f"One-time setup — do both steps:\n"
                    f"  1. Accept terms at:  https://huggingface.co/{self.BACKBONE}\n"
                    f"  2. Authenticate:     huggingface-cli login\n"
                    f"     (get your token from https://huggingface.co/settings/tokens)\n"
                    f"\nAlternatively: export HF_TOKEN=hf_...\n"
                ) from None
            raise
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(self.backbone.config.hidden_size, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.backbone(pixel_values=x)
        return self.classifier(out.last_hidden_state[:, 0])  # CLS token


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    if model_name == "efficientnet_b0":
        m = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
    elif model_name == "efficientnet_b1":
        m = models.efficientnet_b1(weights=EfficientNet_B1_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
    elif model_name == "efficientnet_b2":
        m = models.efficientnet_b2(weights=EfficientNet_B2_Weights.IMAGENET1K_V1)
        m.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(m.classifier[1].in_features, num_classes),
        )
    elif model_name == "dinov2":
        m = DINOv2Classifier(num_classes)
    elif model_name == "dinov3":
        m = DINOv3Classifier(num_classes)
    else:
        raise ValueError(f"Unknown model '{model_name}'. Choices: {list(MODEL_CONFIGS)}")
    return m.to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def freeze_backbone(model: nn.Module, model_name: str) -> None:
    """Freeze pre-trained feature extractor; leave classifier head trainable."""
    if model_name.startswith("efficientnet"):
        for p in model.features.parameters():
            p.requires_grad = False
    elif model_name in ("dinov2", "dinov3"):
        for p in model.backbone.parameters():
            p.requires_grad = False
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Backbone frozen — trainable params: {trainable:,}")


def unfreeze_backbone(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True


def get_param_groups(model: nn.Module, model_name: str,
                     head_lr: float, backbone_lr: float) -> list[dict]:
    """Separate param groups so backbone trains at a lower LR than the head."""
    if model_name.startswith("efficientnet"):
        return [
            {"params": list(model.features.parameters()),   "lr": backbone_lr},
            {"params": list(model.classifier.parameters()), "lr": head_lr},
        ]
    else:
        return [
            {"params": list(model.backbone.parameters()),    "lr": backbone_lr},
            {"params": list(model.classifier.parameters()),  "lr": head_lr},
        ]


# ==============================================================================
# Training / eval loops
# ==============================================================================

def train_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = correct = total = 0
    pbar = tqdm(loader, desc="  Train", leave=False, ncols=90)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return running_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = correct = total = 0
    all_preds, all_labels = [], []
    for imgs, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=90):
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out  = model(imgs)
            loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)
        correct      += (out.argmax(1) == labels).sum().item()
        total        += imgs.size(0)
        all_preds.extend(out.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return running_loss / total, correct / total, all_preds, all_labels


# ==============================================================================
# Plots
# ==============================================================================

def save_plots(history: dict, class_names: list[str], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, key, title in [
        (axes[0], "loss",  "Loss"),
        (axes[1], "acc",   "Accuracy"),
    ]:
        ax.plot(epochs, history[f"train_{key}"], label="Train")
        ax.plot(epochs, history[f"val_{key}"],   label="Val")
        ax.set_title(title); ax.legend(); ax.set_xlabel("Epoch")
    axes[1].set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(str(out_dir / "training_curves.png"), dpi=120)
    plt.close(fig)
    print("   Saved: " + str(out_dir / "training_curves.png"))

    cm = confusion_matrix(history["val_labels"], history["val_preds"])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (Best Val Epoch)")
    plt.tight_layout()
    fig.savefig(str(out_dir / "confusion_matrix.png"), dpi=120)
    plt.close(fig)
    print("   Saved: " + str(out_dir / "confusion_matrix.png"))


# ==============================================================================
# Main training routine
# ==============================================================================

def train(args) -> dict:
    """Train one model. Returns a results dict for use by compare_models.py."""
    model_name = args.model
    img_size   = MODEL_CONFIGS[model_name]["img_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Model : {model_name}  |  Device: {device}  |  ImgSize: {img_size}")
    if device.type == "cuda":
        print("   GPU: " + torch.cuda.get_device_name(0))

    data_root = Path(args.dataset)
    if not (data_root / "train").exists():
        print(f"Dataset not found at '{data_root}'. Run: python scripts/prepare_dataset.py")
        sys.exit(1)

    # Detect minority class
    tmp_ds       = datasets.ImageFolder(str(data_root / "train"))
    class_counts = Counter(tmp_ds.targets)
    minority_idx = min(class_counts, key=class_counts.get)
    minority_cls = tmp_ds.classes[minority_idx]

    print("\nClass distribution (train):")
    for i, cls in enumerate(tmp_ds.classes):
        flag = " <- minority (heavy augment)" if i == minority_idx else ""
        print(f"   [{i}] {cls:>10}: {class_counts[i]:,} images{flag}")

    train_ds = PerClassAugDataset(data_root / "train", minority_idx, img_size)
    val_ds   = datasets.ImageFolder(str(data_root / "val"), transform=get_val_transform(img_size))

    print(f"\n   Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Classes: {train_ds.classes}")

    sampler      = make_weighted_sampler(train_ds.targets)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=(device.type == "cuda"))

    num_classes  = len(train_ds.classes)
    model        = build_model(model_name, num_classes, device)
    n_params     = count_params(model)
    print(f"\n  Params: {n_params:,}")

    loss_weights = compute_loss_weights(train_ds.targets, num_classes, device)
    criterion    = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.1)
    scaler       = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    models_dir   = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Two-phase training setup ───────────────────────────────────────────────
    freeze_eps = args.freeze_epochs
    if freeze_eps > 0 and freeze_eps < args.epochs:
        freeze_backbone(model, model_name)
        print(f"\n  Phase 1 ({freeze_eps} ep): head only — backbone frozen")
        print(f"  Phase 2 ({args.epochs - freeze_eps} ep): full fine-tune — backbone LR x0.1")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    t_max_1   = freeze_eps if freeze_eps > 0 else args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max_1, eta_min=1e-6)

    best_val_acc = 0.0
    no_improve   = 0
    history      = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [],
        "val_preds":  [], "val_labels": [],
    }

    print(f"\nTraining {args.epochs} epochs  "
          f"(lr={args.lr}, wd={args.weight_decay}, batch={args.batch_size}, "
          f"patience={args.patience})\n")
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Phase 2 transition ─────────────────────────────────────────────────
        if freeze_eps > 0 and epoch == freeze_eps + 1:
            unfreeze_backbone(model)
            backbone_lr = args.lr * 0.1
            print(f"\n  Phase 2: fine-tuning all layers  "
                  f"(head LR={args.lr:.1e}, backbone LR={backbone_lr:.1e})")
            optimizer = optim.AdamW(
                get_param_groups(model, model_name, args.lr, backbone_lr),
                weight_decay=args.weight_decay,
            )
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.epochs - freeze_eps, eta_min=1e-6,
            )

        t0 = time.time()
        tl, ta = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl, va, vp, vlab = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl);   history["val_acc"].append(va)

        phase = "P1" if epoch <= freeze_eps else "P2"
        star  = "(*)" if va > best_val_acc else "   "
        print(
            f"Epoch {epoch:03d}/{args.epochs} [{phase}]  "
            f"| Train {tl:.4f}/{ta:.4f}  "
            f"| Val {vl:.4f}/{va:.4f}  "
            f"| LR {scheduler.get_last_lr()[0]:.2e}  | {time.time()-t0:.1f}s  {star}"
        )

        if va > best_val_acc:
            best_val_acc          = va
            no_improve            = 0
            history["val_preds"]  = vp
            history["val_labels"] = vlab

            ckpt = {
                "model_name":  model_name,
                "epoch":       epoch,
                "val_acc":     va,
                "model_state": model.state_dict(),
                "class_names": train_ds.classes,
                "img_size":    img_size,
                "mean":        MEAN,
                "std":         STD,
            }
            if model_name == "dinov2":
                ckpt["dinov2_backbone"] = DINOv2Classifier.BACKBONE
            elif model_name == "dinov3":
                ckpt["dinov3_backbone"] = DINOv3Classifier.BACKBONE
            torch.save(ckpt, models_dir / "best_model.pth")
            print(f"   Saved best model → val_acc={va:.4f}")
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"\n  Early stopping: no val improvement for {args.patience} epochs.")
                break

    total_time = time.time() - t_start

    print(f"\nTraining complete!  Best Val Accuracy: {best_val_acc:.4f}")
    print(classification_report(history["val_labels"], history["val_preds"],
                                target_names=train_ds.classes, digits=4))

    print("Saving plots...")
    save_plots(history, train_ds.classes, models_dir)

    meta = {
        "model_name":      model_name,
        "best_val_acc":    best_val_acc,
        "training_time_s": total_time,
        "params":          n_params,
        "epochs":          args.epochs,
        "freeze_epochs":   args.freeze_epochs,
        "patience":        args.patience,
        "batch_size":      args.batch_size,
        "lr":              args.lr,
        "weight_decay":    args.weight_decay,
        "class_names":     train_ds.classes,
        "img_size":        img_size,
        "minority_class":  minority_cls,
        "imbalance_fixes": [
            "WeightedRandomSampler",
            "Weighted CrossEntropyLoss",
            "PerClassAugDataset (minority heavy augment)",
            "RandomErasing",
        ],
    }
    with open(models_dir / "training_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nModel saved at: {models_dir / 'best_model.pth'}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train EfficientNet/DINOv2/DINOv3 for FFB ripeness with imbalance handling"
    )
    parser.add_argument("--model",      default="efficientnet_b0",
                        choices=list(MODEL_CONFIGS),
                        help="Model architecture (default: efficientnet_b0)")
    parser.add_argument("--dataset",    default="dataset",
                        help="Dataset root with train/val subfolders")
    parser.add_argument("--models-dir", default="models",
                        help="Directory to save checkpoint + plots")
    parser.add_argument("--epochs",        type=int,   default=40,
                        help="Max training epochs (default: 40)")
    parser.add_argument("--freeze-epochs", type=int,   default=5,
                        help="Epochs to train head only with backbone frozen (default: 5). "
                             "Set 0 to disable two-phase training.")
    parser.add_argument("--patience",      type=int,   default=8,
                        help="Early-stop after this many epochs with no val improvement (default: 8). "
                             "Set 0 to disable.")
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=None,
                        help="Head learning rate (default: 3e-4 EfficientNet, 5e-5 DINO). "
                             "Backbone LR is lr × 0.1 during phase 2.")
    parser.add_argument("--weight-decay",  type=float, default=5e-3,
                        help="AdamW weight decay (default: 5e-3)")
    parser.add_argument("--workers",       type=int,   default=4)
    args = parser.parse_args()

    if args.lr is None:
        args.lr = MODEL_CONFIGS[args.model]["default_lr"]

    train(args)
