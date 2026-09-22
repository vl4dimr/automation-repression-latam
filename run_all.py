"""Master script: reproduces every table and figure from the raw downloads.

    python run_all.py            # download (idempotent) + build + descriptives + estimation
    python run_all.py --no-download

Seed is fixed in 02_data/scripts/config.py (SEED = 20260907). Package versions in requirements.txt.
"""
import subprocess
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = [
    ("download", "02_data/scripts/01_download.py"),
    ("dag", "02_data/scripts/00_dag.py"),
    ("build panel", "02_data/scripts/02_build_panel.py"),
    ("descriptives", "02_data/scripts/03_descriptives.py"),
    ("H1-H2 fixed effects and IV", "03_results/scripts/01_fe_models.py"),
    ("H3 threshold", "03_results/scripts/02_threshold.py"),
    ("H3 causal forest", "03_results/scripts/03_causal_forest.py"),
    ("H4 coups and event studies", "03_results/scripts/04_h4_coups.py"),
    ("robustness", "03_results/scripts/05_robustness.py"),
    ("minimum detectable effects", "03_results/scripts/06_mde.py"),
    ("interaction marginal effects", "03_results/scripts/07_marginal_effects.py"),
    ("measurement and common-sample robustness", "03_results/scripts/08_measurement_robustness.py"),
    ("weak-IV and sector diagnostics", "03_results/scripts/09_identification_diagnostics.py"),
    ("manuscript tables and docx", "04_manuscript/build_manuscript.py"),
    ("Spanish manuscript", "04_manuscript/build_manuscript_es.py"),
    ("submission documents", "04_manuscript/build_submission.py"),
]

skip_download = "--no-download" in sys.argv
for name, script in STEPS:
    if "--analysis-only" in sys.argv and script.startswith("04_manuscript/"):
        continue
    if "--manuscript-only" in sys.argv and not script.startswith("04_manuscript/"):
        continue
    if skip_download and name == "download":
        continue
    if not (ROOT / script).exists():
        raise FileNotFoundError(f"Required step missing: {script}")
    t0 = time.time()
    print(f"=== {name}: {script}", flush=True)
    runtime = sys.executable
    if script.startswith("04_manuscript/"):
        bundled = Path.home() / '.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
        runtime = os.environ.get('DOCX_PYTHON', str(bundled) if bundled.exists() else sys.executable)
    r = subprocess.run([runtime, str(ROOT / script)], cwd=ROOT)
    print(f"=== {name} finished in {time.time() - t0:.0f}s with code {r.returncode}", flush=True)
    if r.returncode != 0:
        sys.exit(r.returncode)
print("all steps completed")
