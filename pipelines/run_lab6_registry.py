import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n[INFO] Running src/{script}...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)


def main() -> None:
    print("[INFO] Lab 6: model registry and lifecycle", flush=True)
    for script in ("train_mlflow.py", "register_model.py", "promote_model.py"):
        run(script)
    print("\n[SUCCESS] Lab 6 registry lifecycle completed.", flush=True)


if __name__ == "__main__":
    main()