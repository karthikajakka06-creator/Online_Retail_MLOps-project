import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n[INFO] Running src/{script}...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)


def main() -> None:
    print("[INFO] Lab 4: MLflow tracking and reproducibility", flush=True)
    for script in ("preprocess.py", "train_mlflow.py", "validate_reproducibility.py"):
        run(script)
    print("\n[SUCCESS] Lab 4 tracking pipeline completed.", flush=True)


if __name__ == "__main__":
    main()