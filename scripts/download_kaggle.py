"""
Download Palm Oil FFB Dataset from Kaggle
Dataset: ramadanizikri112/ripeness-of-oil-palm-fruit

Requirements:
  - kagglehub: pip install kagglehub
  - Kaggle credentials configured (kaggle.json in ~/.kaggle/)
    or set env vars: KAGGLE_USERNAME and KAGGLE_KEY
    Get your API key at: https://www.kaggle.com/settings/account → Create New Token
"""

import os
import shutil
import sys
from pathlib import Path

def download_kaggle_dataset(output_dir: str = "data/kaggle") -> str:
    """
    Downloads the Kaggle palm oil FFB dataset and returns the local path.
    """
    try:
        import kagglehub
    except ImportError:
        print("❌ kagglehub not installed. Run: pip install kagglehub")
        sys.exit(1)

    print("📦 Downloading Kaggle dataset: ripeness-of-oil-palm-fruit ...")
    print("   (This may take a few minutes on first download)")

    try:
        path = kagglehub.dataset_download("ramadanizikri112/ripeness-of-oil-palm-fruit")
        print(f"✅ Kaggle dataset downloaded to: {path}")
    except Exception as e:
        print(f"❌ Failed to download Kaggle dataset: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure you have a Kaggle account and API key.")
        print("  2. Download kaggle.json from: https://www.kaggle.com/settings/account")
        print("  3. Place it at: ~/.kaggle/kaggle.json")
        print("  4. Or set environment variables: KAGGLE_USERNAME and KAGGLE_KEY")
        sys.exit(1)

    # Copy to project's data folder for organized access
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    src_path = Path(path)
    if src_path.exists():
        # Mirror the dataset into our output dir
        for item in src_path.rglob("*"):
            if item.is_file():
                rel = item.relative_to(src_path)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target)
        print(f"✅ Copied to project folder: {dest.resolve()}")

    # Show class folders found
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    print("\n📂 Dataset structure detected:")
    total_images = 0
    for folder in sorted(dest.rglob("*")):
        if folder.is_dir():
            images = [f for f in folder.iterdir() if f.suffix.lower() in image_exts]
            if images:
                print(f"   {folder.relative_to(dest)}/  →  {len(images)} images")
                total_images += len(images)
    print(f"\n   Total images: {total_images}")

    return str(dest.resolve())


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download Kaggle palm oil FFB dataset")
    parser.add_argument("--output", default="data/kaggle", help="Output directory")
    args = parser.parse_args()
    download_kaggle_dataset(args.output)
