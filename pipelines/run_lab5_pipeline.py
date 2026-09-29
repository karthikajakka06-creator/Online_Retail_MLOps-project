import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n[INFO] Running src/{script}...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / script)], cwd=ROOT, check=True)


def main() -> None:
    print("[INFO] Lab 5: validated production preprocessing pipeline", flush=True)
    for script in ("validate_data.py", "preprocess.py", "preprocess_pipeline.py", "validate_outputs.py"):
        run(script)
    print("\n[SUCCESS] Lab 5 production pipeline completed.", flush=True)


if __name__ == "__main__":
    main()