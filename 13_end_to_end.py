from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for s in ["08_automate_lifecycle.py","09_inference.py","10_failure_injection.py","11_prepare_deployment.py","12_monitoring.py"]:
 print(f"\n=== {s} ===",flush=True)
 if subprocess.run([sys.executable,str(ROOT/s)],cwd=ROOT).returncode: raise SystemExit(f"{s} failed")
print("[LAB 13] End-to-end Online Retail MLOps workflow completed")
