"""Replay estimates from the supplied clean panel, without rebuilding documents.

Default: every analysis, requiring the archived supplementary source inputs.
--core-only: descriptives and scripts 01--07, using only the clean panel.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# This file is copied to the root of the anonymous replication ZIP.
ROOT = Path(__file__).resolve().parent
CORE = [
    "02_data/scripts/03_descriptives.py",
    "03_results/scripts/01_fe_models.py",
    "03_results/scripts/02_threshold.py",
    "03_results/scripts/03_causal_forest.py",
    "03_results/scripts/04_h4_coups.py",
    "03_results/scripts/05_robustness.py",
    "03_results/scripts/06_mde.py",
    "03_results/scripts/07_marginal_effects.py",
]
ADDITIONAL = [
    "03_results/scripts/08_measurement_robustness.py",
    "03_results/scripts/09_identification_diagnostics.py",
]
EXTRA_INPUTS = [
    "comtrade_847950_imports.csv",
    "wdi.csv",
    "swiid9_92.zip",
    "swiid9_92_summary.csv",
    "ilo_emp_eco_modelled.csv",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-only", action="store_true",
                        help="Run descriptives and scripts 01--07 only; retain archived 08/09 outputs.")
    parser.add_argument("--check-inputs", action="store_true",
                        help="Validate file availability without estimating models.")
    args = parser.parse_args()
    steps = CORE + ([] if args.core_only else ADDITIONAL)
    required = [ROOT / "02_data/clean/panel.parquet"] + [ROOT / s for s in steps]
    if not args.core_only:
        required += [ROOT / "02_data/raw" / s for s in EXTRA_INPUTS]
    missing = [p.relative_to(ROOT).as_posix() for p in required if not p.is_file()]
    if missing:
        print("Required inputs are missing:\n  " + "\n  ".join(missing), file=sys.stderr)
        print("See RAW_INPUTS.md. Use --core-only for the panel-only replay; "
              "it does not rerun measurement or identification diagnostics.", file=sys.stderr)
        return 2
    if args.check_inputs:
        print(f"Inputs present for {len(steps)} analysis steps.")
        return 0
    # Two diagnostic scripts generate these new reports as outputs.
    (ROOT / "05_red_team").mkdir(exist_ok=True)
    for script in steps:
        print(f"Running {script}", flush=True)
        subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)
    scope = "Core analyses only" if args.core_only else "All analyses from the supplied clean panel"
    print(scope + " completed. No documents rebuilt; no external submission or deposit made.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
