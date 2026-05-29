"""
Dataset Preparation Script
──────────────────────────
Merges Kaggle + Roboflow palm oil FFB datasets, normalizes class names,
and splits into train / val sets.

Target class structure:
  dataset/
    train/
      unripe/      ← all unripe FFB images
      ripe/        ← all ripe FFB images
      overripe/    ← all overripe/too-ripe FFB images
    val/
      unripe/
      ripe/
      overripe/

Usage:
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --kaggle data/kaggle --roboflow data/roboflow --val-split 0.2
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ─── CLASS NAME NORMALIZATION ─────────────────────────────────────────────────
# Maps any folder name variation → canonical class name
#
# Confirmed source folders:
#   Kaggle  : "Belum Masak" (unripe), "Masak" (ripe), "Terlalu Masak" (overripe)
#   Roboflow: "underripe", "unripe" (unripe) | "ripe" (ripe) |
#             "overripe", "rotten" (overripe) | "empty_bunch" (EXCLUDED)
#
# normalize_class() lowercases + strips + replaces spaces with underscores
# before looking up in this map, so keys here should already be in that form.

CLASS_MAP = {
    # ── unripe ─────────────────────────────────────────────────────
    "unripe":           "unripe",
    "underripe":        "unripe",
    "under_ripe":       "unripe",
    "un_ripe":          "unripe",

    # Kaggle: "Belum Masak" → belum_masak
    "belum_masak":      "unripe",   # Kaggle confirmed
    # fallback spellings
    "belum_matang":     "unripe",
    "mentah":           "unripe",
    "0":                "unripe",

    # ── ripe ───────────────────────────────────────────────────────
    "ripe":             "ripe",

    # Kaggle: "Masak" → masak
    "masak":            "ripe",     # Kaggle confirmed
    # fallback spellings
    "matang":           "ripe",
    "mature":           "ripe",
    "ready":            "ripe",
    "1":                "ripe",

    # ── overripe ───────────────────────────────────────────────────
    "overripe":         "overripe",
    "over_ripe":        "overripe",

    # Kaggle: "Terlalu Masak" → terlalu_masak
    "terlalu_masak":    "overripe", # Kaggle confirmed
    # fallback spellings
    "terlalu_matang":   "overripe",
    "lewat_matang":     "overripe",
    "too_ripe":         "overripe",
    "tooripe":          "overripe",

    # Roboflow: "rotten" → map to overripe (decomposed / past-peak)
    "rotten":           "overripe", # Roboflow confirmed

    # fallback numeric
    "busuk":            "overripe",
    "2":                "overripe",
}

# ─── EXCLUDED FOLDERS ────────────────────────────────────────────────────────
# These folder names are explicitly recognised but intentionally skipped.
EXCLUDE_MAP = {
    # Roboflow: bunches with no fruit attached — not a ripeness class
    "empty_bunch":      "empty_bunch",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
CLASSES    = ["unripe", "ripe", "overripe"]


def normalize_key(folder_name: str) -> str:
    """Lowercase, strip, and replace spaces/hyphens with underscores."""
    return folder_name.lower().strip().replace(" ", "_").replace("-", "_")


def normalize_class(folder_name: str) -> str | None:
    """
    Map a raw folder name to a canonical class.
    Returns:
      canonical class string  -- if the folder should be included
      "EXCLUDE"               -- if the folder is explicitly excluded (e.g. empty_bunch)
      None                    -- if the folder name is unrecognised
    """
    key = normalize_key(folder_name)
    if key in EXCLUDE_MAP:
        return "EXCLUDE"
    return CLASS_MAP.get(key, None)


def collect_images(source_dir: Path) -> dict[str, list[Path]]:
    """
    Walk source_dir, find all class sub-folders, and return a dict:
    { canonical_class: [image_path, ...] }
    Explicitly excluded folders are reported but skipped.
    Unrecognised folders are warned about.
    """
    collected  = defaultdict(list)
    excluded   = {}   # folder_name → image count
    unrecognised = []

    for folder in sorted(source_dir.rglob("*")):
        if not folder.is_dir():
            continue
        cls = normalize_class(folder.name)
        images = [f for f in folder.iterdir()
                  if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if not images:
            continue

        if cls == "EXCLUDE":
            excluded[folder.name] = len(images)
        elif cls is None:
            unrecognised.append((folder.relative_to(source_dir), len(images)))
        else:
            collected[cls].extend(images)
            print(f"   [{cls:>8}]  {folder.relative_to(source_dir)}  ->  {len(images)} images")

    for fname, n in excluded.items():
        print(f"   [EXCLUDED]  {fname}  ->  {n} images  (intentionally skipped)")

    for rel, n in unrecognised:
        print(f"   [ UNKNOWN]  {rel}  ->  {n} images  *** add to CLASS_MAP if needed ***")

    return dict(collected)


def build_dataset(kaggle_dir: str,
                  roboflow_dir: str,
                  output_dir: str,
                  val_split: float,
                  seed: int,
                  extra_dirs: list[tuple[str, str]] | None = None) -> None:
    """
    extra_dirs: list of (label, path) tuples for additional source directories,
                e.g. [("Roboflow-OXFWT", "data/roboflow_oxfwt"),
                      ("Roboflow-CMU",   "data/roboflow_cmu")]
    """
    random.seed(seed)

    kaggle_path   = Path(kaggle_dir)
    roboflow_path = Path(roboflow_dir)
    out_path      = Path(output_dir)

    # Validate sources
    sources = []
    if kaggle_path.exists():
        sources.append(("Kaggle", kaggle_path))
    else:
        print(f"⚠️  Kaggle data not found at '{kaggle_dir}' — skipping.")

    if roboflow_path.exists():
        sources.append(("Roboflow (original)", roboflow_path))
    else:
        print(f"⚠️  Roboflow data not found at '{roboflow_dir}' — skipping.")

    for label, path_str in (extra_dirs or []):
        p = Path(path_str)
        if p.exists():
            sources.append((label, p))
        else:
            print(f"⚠️  {label} data not found at '{path_str}' — skipping.")

    if not sources:
        print("❌ No data sources found. Run download scripts first:")
        print("   python scripts/download_kaggle.py")
        print("   python scripts/download_roboflow.py --api-key YOUR_KEY")
        sys.exit(1)

    # Collect all images grouped by class
    all_images: dict[str, list[Path]] = defaultdict(list)

    for name, path in sources:
        print(f"\n📂 Scanning {name} dataset: {path}")
        imgs = collect_images(path)
        for cls, paths in imgs.items():
            all_images[cls].extend(paths)

    # Summary
    print("\n─────────────────────────────────────────────")
    print("Combined image counts before split:")
    for cls in CLASSES:
        n = len(all_images.get(cls, []))
        print(f"   {cls:>10}: {n:,} images")
    print("─────────────────────────────────────────────")

    if not any(all_images.values()):
        print("❌ No images collected. Check that your data folders contain images.")
        sys.exit(1)

    # Create output directories
    for split in ("train", "val"):
        for cls in CLASSES:
            (out_path / split / cls).mkdir(parents=True, exist_ok=True)

    # Split and copy
    train_counts: dict[str, int] = {}
    val_counts:   dict[str, int] = {}

    for cls in CLASSES:
        imgs = list(set(all_images.get(cls, [])))  # deduplicate
        if not imgs:
            print(f"⚠️  No images found for class '{cls}' — it will be empty!")
            train_counts[cls] = 0
            val_counts[cls]   = 0
            continue

        random.shuffle(imgs)
        n_val   = max(1, int(len(imgs) * val_split))
        n_train = len(imgs) - n_val

        val_imgs   = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for i, src in enumerate(train_imgs):
            dst = out_path / "train" / cls / f"{cls}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

        for i, src in enumerate(val_imgs):
            dst = out_path / "val" / cls / f"{cls}_{i:05d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

        train_counts[cls] = len(train_imgs)
        val_counts[cls]   = len(val_imgs)

    # Final summary
    print("\n✅ Dataset prepared successfully!")
    print(f"\n{'Class':>12}  {'Train':>8}  {'Val':>8}  {'Total':>8}")
    print("─" * 44)
    total_train = total_val = 0
    for cls in CLASSES:
        t = train_counts.get(cls, 0)
        v = val_counts.get(cls, 0)
        total_train += t
        total_val   += v
        print(f"{cls:>12}  {t:>8,}  {v:>8,}  {t+v:>8,}")
    print("─" * 44)
    print(f"{'TOTAL':>12}  {total_train:>8,}  {total_val:>8,}  {total_train+total_val:>8,}")
    print(f"\n📁 Output: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare merged FFB dataset from Kaggle + Roboflow sources",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/prepare_dataset.py\n"
            "  python scripts/prepare_dataset.py --all-roboflow\n"
            "  python scripts/prepare_dataset.py --roboflow-oxfwt data/roboflow_oxfwt "
            "--roboflow-cmu data/roboflow_cmu"
        ),
    )
    parser.add_argument("--kaggle",         default="data/kaggle",
                        help="Kaggle data directory")
    parser.add_argument("--roboflow",       default="data/roboflow",
                        help="Roboflow original dataset directory")
    parser.add_argument("--roboflow-oxfwt", default=None,
                        help="Roboflow palmoilobjectdetectionclassification/palm-oil-ripeness-oxfwt")
    parser.add_argument("--roboflow-cmu",   default=None,
                        help="Roboflow chiang-mai-university-skeej/ripeness-ffb-palm-oil")
    parser.add_argument("--all-roboflow",   action="store_true",
                        help="Include all Roboflow datasets with default paths "
                             "(data/roboflow_oxfwt and data/roboflow_cmu)")
    parser.add_argument("--output",         default="dataset",
                        help="Output dataset directory")
    parser.add_argument("--val-split",      type=float, default=0.2,
                        help="Validation fraction (default: 0.2)")
    parser.add_argument("--seed",           type=int,   default=42,
                        help="Random seed")
    args = parser.parse_args()

    extra = []
    oxfwt_path = args.roboflow_oxfwt or ("data/roboflow_oxfwt" if args.all_roboflow else None)
    cmu_path   = args.roboflow_cmu   or ("data/roboflow_cmu"   if args.all_roboflow else None)
    if oxfwt_path:
        extra.append(("Roboflow-OXFWT", oxfwt_path))
    if cmu_path:
        extra.append(("Roboflow-CMU", cmu_path))

    build_dataset(
        kaggle_dir=args.kaggle,
        roboflow_dir=args.roboflow,
        output_dir=args.output,
        val_split=args.val_split,
        seed=args.seed,
        extra_dirs=extra,
    )
