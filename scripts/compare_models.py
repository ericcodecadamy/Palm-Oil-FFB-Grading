"""
Compare EfficientNet-B0/B1/B2, DINOv2, and DINOv3 for FFB ripeness classification.

For each model this script:
  1. Trains from scratch on the specified dataset
  2. Records best val accuracy, total training time, and parameter count
  3. Saves checkpoint + plots to {output-dir}/{model_name}/

Final outputs (in {output-dir}/):
  comparison.json   — raw results for all models
  comparison.txt    — formatted table
  comparison.png    — bar chart

Usage:
  python scripts/compare_models.py                              # all 5 models, 25 epochs
  python scripts/compare_models.py --epochs 15                 # faster comparison
  python scripts/compare_models.py --models efficientnet_b0 efficientnet_b2 dinov2
  python scripts/compare_models.py --models efficientnet_b0 dinov3 --epochs 20
"""

import sys
import json
import argparse
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Allow importing from the same scripts/ folder
sys.path.insert(0, str(Path(__file__).parent))
from train import train, MODEL_CONFIGS

ALL_MODELS = list(MODEL_CONFIGS.keys())
BAR_COLORS = {
    "efficientnet_b0": "#3b82f6",
    "efficientnet_b1": "#8b5cf6",
    "efficientnet_b2": "#ec4899",
    "dinov2":          "#10b981",
    "dinov3":          "#f59e0b",
}


def compare(args) -> dict:
    results: dict[str, dict] = {}

    for model_name in args.models:
        sep = "=" * 64
        print(f"\n{sep}")
        print(f"  Training: {model_name}  ({args.models.index(model_name)+1}/{len(args.models)})")
        print(sep)

        model_args = SimpleNamespace(
            model         = model_name,
            dataset       = args.dataset,
            models_dir    = str(Path(args.output_dir) / model_name),
            epochs        = args.epochs,
            freeze_epochs = args.freeze_epochs,
            patience      = args.patience,
            batch_size    = args.batch_size,
            lr            = MODEL_CONFIGS[model_name]["default_lr"],
            weight_decay  = args.weight_decay,
            workers       = args.workers,
        )

        try:
            meta = train(model_args)
            results[model_name] = meta
        except Exception as e:
            print(f"\n⚠️  {model_name} training failed: {e}")
            results[model_name] = {"error": str(e), "best_val_acc": 0.0}

    # ─── Save raw JSON ─────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # ─── Determine best model ──────────────────────────────────────────────────
    best_model = max(
        (m for m in results if "error" not in results[m]),
        key=lambda m: results[m].get("best_val_acc", 0),
        default=None,
    )

    # ─── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print("  MODEL COMPARISON RESULTS")
    print(f"{'='*72}")
    print(f"  {'Model':<22} {'Val Acc':>9} {'Params':>12} {'Train Time':>12}  Status")
    print(f"  {'-'*66}")
    for model_name, meta in results.items():
        if "error" in meta:
            print(f"  {model_name:<22}  {'ERROR':>9}  {'—':>12}  {'—':>12}  {meta['error'][:30]}")
            continue
        acc    = meta.get("best_val_acc", 0)
        params = meta.get("params", 0)
        t_min  = meta.get("training_time_s", 0) / 60
        marker = "  ← BEST" if model_name == best_model else ""
        print(f"  {model_name:<22}  {acc:>9.4f}  {params:>12,}  {t_min:>10.1f}m{marker}")

    if best_model:
        best_acc = results[best_model]["best_val_acc"]
        ckpt_path = out_dir / best_model / "best_model.pth"
        print(f"\n  Best model : {best_model}  (val_acc={best_acc:.4f})")
        print(f"  Checkpoint : {ckpt_path}")
        print(f"\n  To serve the best model:")
        print(f"    cp {ckpt_path} models/best_model.pth && python run.py")
    print(f"{'='*72}\n")

    # ─── Save text table ───────────────────────────────────────────────────────
    lines = [
        "MODEL COMPARISON RESULTS",
        "=" * 66,
        f"{'Model':<22} {'Val Acc':>9} {'Params':>12} {'Train Time':>12}",
        "-" * 66,
    ]
    for model_name, meta in results.items():
        if "error" in meta:
            lines.append(f"{model_name:<22}  ERROR  {meta['error'][:30]}")
            continue
        acc   = meta.get("best_val_acc", 0)
        params = meta.get("params", 0)
        t_min  = meta.get("training_time_s", 0) / 60
        marker = "  <- BEST" if model_name == best_model else ""
        lines.append(f"{model_name:<22}  {acc:>9.4f}  {params:>12,}  {t_min:>10.1f}m{marker}")
    if best_model:
        lines += [
            "",
            f"Best: {best_model}  val_acc={results[best_model]['best_val_acc']:.4f}",
            f"Checkpoint: {out_dir}/{best_model}/best_model.pth",
        ]
    with open(out_dir / "comparison.txt", "w") as f:
        f.write("\n".join(lines) + "\n")

    # ─── Bar chart ─────────────────────────────────────────────────────────────
    valid = {m: r for m, r in results.items() if "error" not in r}
    if valid:
        names    = list(valid.keys())
        val_accs = [valid[m].get("best_val_acc", 0) for m in names]
        colors   = [BAR_COLORS.get(m, "#6b7280") for m in names]

        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.4), 5))
        bars = ax.bar(names, val_accs, color=colors, width=0.55, edgecolor="white", linewidth=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=9)
        ax.set_ylim(0, min(1.0, max(val_accs) + 0.12))
        ax.set_ylabel("Validation Accuracy")
        ax.set_title("Model Comparison — Palm Oil FFB Ripeness Classifier")
        ax.axhline(y=max(val_accs), color="red", linestyle="--", alpha=0.4, linewidth=1)
        ax.tick_params(axis="x", labelrotation=15)
        plt.tight_layout()
        fig.savefig(str(out_dir / "comparison.png"), dpi=130)
        plt.close(fig)
        print(f"  Comparison chart saved: {out_dir / 'comparison.png'}")

    print(f"  All results saved to: {out_dir}/\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare FFB ripeness classifier architectures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Models available: " + ", ".join(ALL_MODELS) + "\n\n"
            "Tip: --epochs 10-15 gives a quick comparison; use 25+ for final results.\n"
            "     DINOv2/DINOv3 are slower per epoch — consider --batch-size 16 on small GPUs."
        ),
    )
    parser.add_argument("--models",     nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS,
                        help="Models to compare (default: all 5)")
    parser.add_argument("--dataset",    default="dataset",
                        help="Dataset root with train/val subfolders")
    parser.add_argument("--output-dir", default="models",
                        help="Root directory for per-model subdirs and comparison outputs")
    parser.add_argument("--epochs",        type=int,   default=40)
    parser.add_argument("--freeze-epochs", type=int,   default=5,
                        help="Frozen-backbone epochs per model (default: 5)")
    parser.add_argument("--patience",      type=int,   default=8,
                        help="Early-stop patience per model (default: 8)")
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--weight-decay",  type=float, default=5e-3)
    parser.add_argument("--workers",       type=int,   default=4)
    args = parser.parse_args()

    compare(args)
