"""Approximate minimum detectable effects (80% power, 5% two-sided) for selected tests, from the
country-clustered standard errors of the baseline specifications: MDE = (1.96 + 0.84) x SE.
Output: tab_mde
Run: python 03_results/scripts/06_mde.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROLS, TABLES, load_panel, twfe, write_table

p = load_panel((1990, 2024))
p["k_x_threat"] = p.k_l1 * p.threat_mob_l1
p["k_x_antisys"] = p.k_l1 * p.antisys_l5
p["k_x_rep"] = p.k_l1 * p.rep_phys_l1
fam = [("H1a: labour share on log K/L", twfe(p, "labsh_any", ["k_l1"] + CONTROLS), "k_l1", "share units (multiply by 100 for pp)"),
       ("H1b: mobilization on log(sK/sL), low repression", twfe(p[p.high_rep_l1 == 0], "mob", ["ln_sk_sl", "ln_sk_sl_l1", "ln_sk_sl_l2"] + CONTROLS), "ln_sk_sl", "mobilization units"),
       ("H2: physical repression on log K/L", twfe(p, "rep_phys", ["k_l1"] + CONTROLS), "k_l1", "index units (SD 0.21)"),
       ("H2: physical repression on K/L x anti-system threat", twfe(p, "rep_phys", ["k_l1", "antisys_l5", "k_x_antisys"] + CONTROLS), "k_x_antisys", "index units"),
       ("H2: redistribution on lagged repression", twfe(p, "abs_red", ["k_l1", "rep_phys_l1", "k_x_rep"] + CONTROLS), "rep_phys_l1", "Gini points"),
       ("H3: log robots per worker on log K/L", twfe(p, "ln_robots_pw", ["k_l1"] + CONTROLS), "k_l1", "log points")]
rows = []
for lab, res, v, unit in fam:
    se = res.bse[v]; b = res.params[v]; mde = 2.8 * se
    sd_y = res._d[res._y].std()
    rows.append([lab, f"{b:.3f}", f"{se:.3f}", f"{mde:.3f}", f"{mde / sd_y:.2f}", unit, str(int(res.nobs))])
write_table(rows, ["estimate", "SE", "MDE (80% power)", "MDE / SD of outcome", "units", "N"], "tab_mde",
            "Approximate minimum detectable effects for selected specifications", "tab:mde",
            "Normal approximation: MDE = 2.8 x country-clustered SE for 80 percent power at a two-sided 5 percent level. "
            "Outcomes retain original units; continuous regressors are standardized. With 18 clusters this is a sensitivity diagnostic, "
            "not a finite-sample power guarantee or an equivalence test. MDE / SD divides by the outcome SD. "
            "For interactions the coefficient is on the standardized product, not a marginal effect. Specifications differ from the multiple-testing family.")
print("done: MDE")
