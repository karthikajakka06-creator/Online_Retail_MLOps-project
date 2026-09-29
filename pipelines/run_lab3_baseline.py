import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n[INFO] Running src/{script}...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)


def main() -> None:
    print("[INFO] Lab 3: Online Retail baseline pipeline", flush=True)
    for script in ("preprocess.py", "train.py", "evaluate.py"):
        run(script)
    print("\n[SUCCESS] Lab 3 baseline pipeline completed.", flush=True)


if __name__ == "__main__":
    main()