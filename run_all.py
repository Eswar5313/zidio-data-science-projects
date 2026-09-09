"""One command re-runs the whole pipeline from raw data:  python src/run_all.py"""
import subprocess, sys, time
from pathlib import Path
SRC = Path(__file__).resolve().parent
t0 = time.time()
for step in ["pipeline.py", "forecast.py", "risk.py", "eda.py", "export_web.py"]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, str(SRC / step)], check=True)
print(f"\nAll steps complete in {time.time()-t0:.0f}s. Outputs in outputs/, figures in reports/figures/, dashboard data in web/data/.")
