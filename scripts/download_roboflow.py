"""
Download Palm Oil FFB Datasets from Roboflow

Supported datasets (--dataset):
  original  — palm-fruit-classification / palm-fruit-ripeness-classificationcnn
  oxfwt     — palmoilobjectdetectionclassification / palm-oil-ripeness-oxfwt
  cmu       — chiang-mai-university-skeej / ripeness-ffb-palm-oil

Usage:
  python scripts/download_roboflow.py --api-key YOUR_KEY            # original only
  python scripts/download_roboflow.py --api-key YOUR_KEY --all      # all 3 datasets
  python scripts/download_roboflow.py --api-key YOUR_KEY --dataset cmu
  export ROBOFLOW_API_KEY=YOUR_KEY && python scripts/download_roboflow.py --all
"""

import os
import sys
from pathlib import Path


ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY", "")

# All known Roboflow palm oil FFB datasets.
# If a dataset version fails, try --version 2 or 3.
DATASETS = {
    "original": {
        "workspace": "palm-fruit-classification",
        "project":   "palm-fruit-ripeness-classificationcnn",
        "version":   1,
        "output":    "data/roboflow",
        "note":      None,
    },
    "oxfwt": {
        "workspace": "palmoilobjectdetectionclassification",
        "project":   "palm-oil-ripeness-oxfwt",
        "version":   1,
        "output":    "data/roboflow_oxfwt",
        "note":      "Object-detection workspace — if 'folder' format fails, try --version 2",
    },
    "cmu": {
        "workspace": "chiang-mai-university-skeej",
        "project":   "ripeness-ffb-palm-oil",
        "version":   1,
        "output":    "data/roboflow_cmu",
        "note":      "Chiang Mai University FFB dataset",
    },
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def download_dataset(key: str, api_key: str,
                     output_override: str | None = None,
                     version_override: int | None = None) -> str:
    if key not in DATASETS:
        print(f"❌ Unknown dataset '{key}'. Choices: {list(DATASETS)}")
        sys.exit(1)

    cfg       = DATASETS[key]
    workspace = cfg["workspace"]
    project   = cfg["project"]
    version   = version_override or cfg["version"]
    out_dir   = output_override  or cfg["output"]

    if not api_key:
        print("❌ Roboflow API key not set!")
        print("   Pass with --api-key YOUR_KEY  or  export ROBOFLOW_API_KEY=YOUR_KEY")
        print("   Get a free key at: https://app.roboflow.com → avatar → Settings → API")
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ roboflow not installed. Run: pip install roboflow")
        sys.exit(1)

    print(f"\n{'─'*54}")
    print(f"  Dataset  : {key}")
    print(f"  Workspace: {workspace}")
    print(f"  Project  : {project}  (v{version})")
    print(f"  Output   : {out_dir}")
    if cfg["note"]:
        print(f"  Note     : {cfg['note']}")
    print(f"{'─'*54}")

    try:
        rf      = Roboflow(api_key=api_key)
        dataset = rf.workspace(workspace).project(project).version(version)
        dataset.download("folder", location=out_dir)
        print(f"✅ Downloaded to: {out_dir}")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print(f"   Tips:")
        print(f"   - Check your API key is correct")
        print(f"   - Try --version 2 or 3 (currently v{version})")
        print(f"   - Verify the project is public at https://universe.roboflow.com")
        return ""

    dest  = Path(out_dir)
    total = 0
    print("\n  Folder structure detected:")
    for folder in sorted(dest.rglob("*")):
        if not folder.is_dir():
            continue
        imgs = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTS]
        if imgs:
            print(f"   {folder.relative_to(dest)}/  →  {len(imgs)} images")
            total += len(imgs)
    print(f"\n  Total images: {total}")
    print(f"\n  Next step: python scripts/prepare_dataset.py")
    print(f"  (If you see [ UNKNOWN] class warnings, add the folder names to CLASS_MAP in prepare_dataset.py)")

    return str(dest.resolve())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Download Roboflow palm oil FFB datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            [f"  {k}: {v['workspace']}/{v['project']}" for k, v in DATASETS.items()]
        ),
    )
    parser.add_argument("--api-key",  default=ROBOFLOW_API_KEY,
                        help="Roboflow Private API key (or set ROBOFLOW_API_KEY env var)")
    parser.add_argument("--dataset",  default="original", choices=list(DATASETS),
                        help="Which dataset to download (default: original)")
    parser.add_argument("--all",      action="store_true",
                        help="Download all 3 datasets")
    parser.add_argument("--output",   default=None,
                        help="Override output directory (single-dataset mode only)")
    parser.add_argument("--version",  type=int, default=None,
                        help="Override Roboflow dataset version number")
    args = parser.parse_args()

    keys = list(DATASETS.keys()) if args.all else [args.dataset]
    output_override = args.output if len(keys) == 1 else None

    for key in keys:
        download_dataset(key, args.api_key, output_override, args.version)
