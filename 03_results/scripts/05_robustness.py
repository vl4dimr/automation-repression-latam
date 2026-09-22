"""Robustness: alternative repression measures, sample exclusions, windows, placebo, multiple testing.

Tables: tab_rob_measures, tab_rob_samples, tab_rob_placebo, tab_rob_multiple
Run: python 03_results/scripts/05_robustness.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROLS, TABLES, coef_rows, load_panel, twfe, write_table, wild_cluster_p

p = load_panel((1990, 2024))
p["k_x_threat"] = p.k_l1 * p.threat_mob_l1
p["k_x_rep"] = p.k_l1 * p.rep_phys_l1
X2 = ["k_l1", "threat_mob_l1", "k_x_threat"] + CONTROLS

# ------------------------------------------------------------------ alternative measures of repression
MEAS = ["rep_phys", "rep_cso", "rep_expr", "pts_mean", "fh_cl", "mmp_repress_share", "rep_index"]
MEAS_LAB = ["physical", "CSO", "expression", "PTS", "FH civil lib.", "MMP repressed share", "composite"]
m = [twfe(p, y, X2) for y in MEAS]
rows = coef_rows(m, [("k_l1", "log K/L (t-1)"), ("threat_mob_l1", "Threat: past mobilization (t-1)"), ("k_x_threat", "K/L x past mobilization")])
write_table(rows, MEAS_LAB, "tab_rob_measures", "Robustness: alternative measures of repression (mobilization-based threat proxy)", "tab:robmeas",
            "Same specification as Table H2 column (2). Higher values mean more repression in every column (FH civil liberties 1-7, PTS 1-5).")
p["k_x_antisys"] = p.k_l1 * p.antisys_l5
X3 = ["k_l1", "antisys_l5", "k_x_antisys"] + CONTROLS
m3 = [twfe(p, y, X3) for y in MEAS]
rows = coef_rows(m3, [("k_l1", "log K/L (t-1)"), ("antisys_l5", "Threat: anti-system movements"), ("k_x_antisys", "K/L x anti-system")])
write_table(rows, MEAS_LAB, "tab_rob_measures_antisys", "Robustness: alternative measures of repression (anti-system-movement threat proxy, 18 countries)",
            "tab:robmeas2", "Same specification as Table H2 column (7). Higher values mean more repression in every column.")

# ------------------------------------------------------------------ sample exclusions and windows
samples = {"baseline": p, "excl. VEN, NIC": p[~p.iso3.isin(["VEN", "NIC"])], "excl. VEN, NIC, CUB-like autocracies (RoW<=1 all years)": p[~p.iso3.isin(["VEN", "NIC"])],
           "1990-2007": p[p.year <= 2007], "2008-2024": p[p.year >= 2008], "PWT years only (<=2019)": p[p.year <= 2019],
           "excl. largest (BRA, MEX)": p[~p.iso3.isin(["BRA", "MEX"])], "democracies only (t-1)": p[p.dem_l1 == 1]}
del samples["excl. VEN, NIC, CUB-like autocracies (RoW<=1 all years)"]
rr = []; rq = []
for lab, s in samples.items():
    a = twfe(s, "rep_phys", X2); b = twfe(s, "abs_red", ["k_l1", "rep_phys_l1", "k_x_rep"] + CONTROLS); c = twfe(s, "labsh_any", ["k_l1"] + CONTROLS)
    rr.append([lab,
               f"{a.params['k_l1']:.3f} ({a.bse['k_l1']:.3f})" if a is not None else "",
               f"{a.params['k_x_threat']:.3f} ({a.bse['k_x_threat']:.3f})" if a is not None else "",
               f"{b.params['rep_phys_l1']:.3f} ({b.bse['rep_phys_l1']:.3f})" if b is not None else "",
               f"{c.params['k_l1']:.3f} ({c.bse['k_l1']:.3f})" if c is not None else "",
               str(int(a.nobs)) if a is not None else ""])
write_table(rr, ["rep. on K/L", "rep. on K/L x threat", "redist. on rep.", "labsh on K/L", "N (rep.)"],
            "tab_rob_samples", "Robustness: sample exclusions and windows", "tab:robsamp",
            "Each cell is the coefficient (country-clustered SE) from the baseline specification re-estimated on the indicated sample.")

# ------------------------------------------------------------------ placebo: pre-robot period for the shift-share share
p60 = load_panel((1970, 1995))
p60["trend_x_manuf"] = p60.w_manuf_use * (p60.year - 1970)
pl = [twfe(p60, "rep_phys", ["trend_x_manuf"] + ["ln_gdppc", "urban_share", "trade_open"]),
      twfe(p60, "labsh_any", ["trend_x_manuf"] + ["ln_gdppc", "urban_share", "trade_open"])]
# placebo outcome: non-automatable sector? use agricultural employment share trend instead of robots (should not respond to z_robot)
pa = [twfe(p, "urban_share", ["z_robot_l1"] + ["ln_gdppc", "trade_open", "resource_rents"]),
      twfe(p, "v2x_corr", ["z_robot_l1"] + CONTROLS)]
rows = [["1970-1995: manufacturing share x trend -> repression", f"{pl[0].params['trend_x_manuf']:.3f} ({pl[0].bse['trend_x_manuf']:.3f})" if pl[0] is not None else "", str(int(pl[0].nobs)) if pl[0] is not None else ""],
        ["1970-1995: manufacturing share x trend -> labour share", f"{pl[1].params['trend_x_manuf']:.3f} ({pl[1].bse['trend_x_manuf']:.3f})" if pl[1] is not None else "", str(int(pl[1].nobs)) if pl[1] is not None else ""],
        ["1990-2024: z_robot -> urban share (placebo outcome)", f"{pa[0].params['z_robot_l1']:.3f} ({pa[0].bse['z_robot_l1']:.3f})" if pa[0] is not None else "", str(int(pa[0].nobs)) if pa[0] is not None else ""],
        ["1990-2024: z_robot -> political corruption (placebo outcome)", f"{pa[1].params['z_robot_l1']:.3f} ({pa[1].bse['z_robot_l1']:.3f})" if pa[1] is not None else "", str(int(pa[1].nobs)) if pa[1] is not None else ""]]
write_table(rows, ["coefficient (SE)", "N"], "tab_rob_placebo", "Placebo tests for the shift-share design", "tab:placebo",
            "Rows 1-2: pre-series trend diagnostic; the 1993-97 share is not predetermined for the 1970-95 period. "
            "Rows 3-4: alternative outcome diagnostics, not definitive exclusion tests; urbanization may respond to industrial change.")
placebo_rows = [row.copy() for row in rows]

# ------------------------------------------------------------------ multiple testing over the declared exploratory family
fam = [("H1a", twfe(p, "labsh_any", ["k_l1"] + CONTROLS), "k_l1"),
       ("H1b low-rep", twfe(p[p.high_rep_l1 == 0], "mob", ["ln_sk_sl", "ln_sk_sl_l1", "ln_sk_sl_l2"] + CONTROLS), "ln_sk_sl"),
       ("H2 rep on K/L", twfe(p, "rep_phys", X2), "k_l1"),
       ("H2 rep on K/L x threat", twfe(p, "rep_phys", X2), "k_x_threat"),
       ("H2 redist on rep", twfe(p, "abs_red", ["k_l1", "rep_phys_l1", "k_x_rep"] + CONTROLS), "rep_phys_l1"),
       ("H3 robots on K/L", twfe(p, "ln_robots_pw", ["k_l1"] + CONTROLS), "k_l1")]
rows = []; pv = []
for lab, res, v in fam:
    if res is None: continue
    pw = wild_cluster_p(res, v, B=999)
    pv.append(pw); rows.append([lab, f"{res.params[v]:.3f}", f"{res.pvalues[v]:.3f}", f"{pw:.3f}"])
pv = np.array(pv); m_ = len(pv)
order = np.argsort(pv); bh = np.empty(m_); prev = 1.0
for rank, i in enumerate(order[::-1]):
    val = min(prev, pv[i] * m_ / (m_ - rank)); bh[i] = val; prev = val
holm = np.minimum(1, np.maximum.accumulate(pv[order] * (m_ - np.arange(m_))))[np.argsort(order)]
for r, b, hh in zip(rows, bh, holm):
    r += [f"{hh:.3f}", f"{b:.3f}"]
write_table(rows, ["coef.", "cluster p", "wild bootstrap p", "Holm p", "BH q"], "tab_rob_multiple",
            "Multiple-testing adjustment over the declared exploratory family of six main tests", "tab:multi",
            "Holm step-down (family-wise) and Benjamini-Hochberg (false discovery rate) adjustments applied to the wild-cluster bootstrap p-values.")
(TABLES / "summary_robustness.json").write_text(json.dumps(dict(samples=rr, multiple=rows, placebo=placebo_rows), indent=1), encoding="utf-8")
print("done: robustness")
