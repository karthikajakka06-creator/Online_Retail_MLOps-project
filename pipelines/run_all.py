import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script: str) -> None:
    print(f"\n[INFO] Running {script}...", flush=True)
    subprocess.run([sys.executable, str(ROOT / "pipelines" / script)], cwd=ROOT, check=True)


def main() -> None:
    print("[INFO] Running the complete Online Retail MLOps workflow", flush=True)
    for script in (
        "run_lab3_baseline.py",
        "run_lab4_tracking.py",
        "run_lab5_pipeline.py",
        "run_lab6_registry.py",
    ):
        run(script)
    print("\n[SUCCESS] Labs 3-6 completed successfully.", flush=True)


if __name__ == "__main__":
    main()