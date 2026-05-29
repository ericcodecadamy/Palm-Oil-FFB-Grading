"""
Palm Oil FFB Predictor — One-Click Launcher
────────────────────────────────────────────
Starts the FastAPI web server and prints the local URL for
accessing the app from any device on the same WiFi network.

Usage:
  python run.py
  python run.py --port 8080
  python run.py --host 0.0.0.0 --port 8000
"""

import sys
import socket
import argparse
import subprocess
from pathlib import Path


def get_local_ip() -> str:
    """Return the machine's LAN IP address."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def check_model() -> bool:
    model_path = Path("models/best_model.pth")
    if not model_path.exists():
        print("⚠️  WARNING: No trained model found at models/best_model.pth")
        print("   The web app will start but predictions won't work until you train.")
        print()
        print("   To train the model, first download data then run:")
        print("     python scripts/download_kaggle.py")
        print("     python scripts/download_roboflow.py --api-key YOUR_KEY")
        print("     python scripts/prepare_dataset.py")
        print("     python scripts/train.py")
        print()
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Start the FFB Ripeness Predictor web app")
    parser.add_argument("--host",    default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--port",    type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload",  action="store_true", help="Enable auto-reload (dev mode)")
    args = parser.parse_args()

    local_ip = get_local_ip()
    model_ok = check_model()

    print("━" * 54)
    print("🌴  Palm Oil FFB Ripeness Predictor")
    print("━" * 54)
    print(f"   Model status : {'✅ Ready' if model_ok else '⚠️  Not trained yet'}")
    print()
    print("   Access the app at:")
    print(f"   🖥️  Laptop  →  http://localhost:{args.port}")
    print(f"   📱  Phone   →  http://{local_ip}:{args.port}")
    print()
    print("   (Make sure your phone is on the same WiFi network)")
    print("   Press Ctrl+C to stop the server")
    print("━" * 54)
    print()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "app:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "info",
    ]
    if args.reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Server error: {e}")
        print("   Make sure uvicorn is installed: pip install uvicorn[standard]")
        sys.exit(1)


if __name__ == "__main__":
    main()
