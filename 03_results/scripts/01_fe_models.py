"""H1 and H2: two-way fixed-effects panels, distributed lags, IV.

Tables: tab_h1a_firststage, tab_h1b_mobilization, tab_h2_repression, tab_h2_redistribution,
        tab_h2_choice, tab_iv, tab_pwt_imputation
Figures: fig_coef_h1h2
Run: python 03_results/scripts/01_fe_models.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fuentes TrueType incrustadas en PDF y EPS: las Type3 que matplotlib escribe por omision
# las marca el preflight de la editorial. No altera datos ni ejes.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '02_data' / 'scripts'))
from figlabels import L
from common import (CONTROLS, FIGURES, NAMES, TABLES, coef_rows, driscoll_kraay, iv2sls, load_panel, twfe,
                    wild_cluster_p, write_table, fmt, stars)

p = load_panel((1990, 2024))
p["k_x_threat"] = p.k_l1 * p.threat_mob_l1
p["k_x_antisys"] = p.k_l1 * p.antisys_l5
p["k_x_rep"] = p.k_l1 * p.rep_phys_l1
p["k_x_high"] = p.k_l1 * p.high_rep_l1
p["sksl_x_high"] = p.ln_sk_sl * p.high_rep_l1
p["sksl_l1_x_high"] = p.ln_sk_sl_l1 * p.high_rep_l1
p["sksl_l2_x_high"] = p.ln_sk_sl_l2 * p.high_rep_l1
p["rep_choice"] = (p.rep_phys - p.rep_phys.mean()) / p.rep_phys.std() - (p.abs_red - p.abs_red.mean()) / p.abs_red.std()
p["china_x_manuf"] = p.w_manuf_use * np.log(p.year - 1989)   # placeholder trend x share (manufacturing-share trend; not observed China exposure)

# ------------------------------------------------------------------ H1a: labour share first stage
cols = []
cols.append(twfe(p, "labsh_any", ["k_l1"] + CONTROLS))
cols.append(twfe(p, "labsh_any", ["k_l1", "ln_robots_pw_l1"] + CONTROLS))
cols.append(twfe(p, "labsh_any", ["k_l1", "routine_share_l1"] + CONTROLS))
cols.append(twfe(p, "ilo_labsh", ["k_l1", "ln_robots_pw_l1"] + CONTROLS))
p60 = load_panel((1960, 2024))
cols.append(twfe(p60, "labsh_any", ["k_l1"] + CONTROLS))
# PWT labour share only in years where PWT actually varies (drops the imputed constant stretches)
pv = p.copy(); pv["labsh_var"] = pv.pwt_labsh.where(pv.groupby("iso3").pwt_labsh.diff() != 0)
cols.append(twfe(pv, "labsh_var", ["k_l1", "ln_robots_pw_l1"] + CONTROLS))
labels = [("k_l1", "log K/L (t-1)"), ("ln_robots_pw_l1", "log robot stock per worker (t-1)"),
          ("routine_share_l1", "Routine occupation share (t-1)")]
rows = coef_rows(cols, labels, bootstrap_vars=("k_l1",))
write_table(rows, ["(1) PWT/ILO", "(2) + robots", "(3) + routine", "(4) ILO share", "(5) 1960-2024", "(6) PWT varying"],
            "tab_h1a_firststage",
            "H1a. Labour share and capital per worker (two-way FE)", "tab:h1a",
            "Dependent variable: labour share. All regressors standardized; country and year fixed effects; standard errors clustered by country "
            "in parentheses; wild-cluster bootstrap p-values in brackets (499 Rademacher draws). Controls: log GDP pc, urban share, trade openness, "
            "resource rents. Column (6) keeps only country-years in which the PWT labour share differs from the previous year. "
            "$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$.")
h1a = cols

# per-country slopes for the discussion
slopes = []
for iso, g in p.dropna(subset=["labsh_any", "k"]).groupby("iso3"):
    if len(g) > 10:
        b = np.polyfit(g.k, g.labsh_any, 1)[0]; slopes.append((iso, b, len(g)))
pd.DataFrame(slopes, columns=["iso3", "slope_labsh_on_k", "n"]).to_csv(TABLES / "tab_h1a_country_slopes.csv", index=False)

# PWT imputation diagnostic
diagnostic = load_panel((1959, 2019))
diagnostic['previous_labsh'] = diagnostic.groupby('iso3').pwt_labsh.shift()
diagnostic = diagnostic[diagnostic.year >= 1960]
def repeated_share(g):
    valid = g.pwt_labsh.notna() & g.previous_labsh.notna()
    return float((g.loc[valid, 'pwt_labsh'] == g.loc[valid, 'previous_labsh']).mean()) if valid.any() else np.nan
imp = diagnostic.groupby('iso3').apply(repeated_share, include_groups=False).rename('share_years_labsh_unchanged')
imp.to_csv(TABLES / "tab_pwt_imputation.csv")

# ------------------------------------------------------------------ H1b: mobilization with regime heterogeneity
base_x = ["ln_sk_sl", "ln_sk_sl_l1", "ln_sk_sl_l2"]
het_x = base_x + ["sksl_x_high", "sksl_l1_x_high", "sksl_l2_x_high", "high_rep_l1"]
m = []
m.append(twfe(p, "mob", base_x + CONTROLS))
m.append(twfe(p, "mob", het_x + CONTROLS))
m.append(twfe(p[p.high_rep_l1 == 0], "mob", base_x + CONTROLS))
m.append(twfe(p[p.high_rep_l1 == 1], "mob", base_x + CONTROLS))
m.append(twfe(p, "ln_mmp_protests", het_x + CONTROLS))
m.append(twfe(p, "v2cademmob", het_x + CONTROLS))
labels = [("ln_sk_sl", "log(sK/sL), t"), ("ln_sk_sl_l1", "log(sK/sL), t-1"), ("ln_sk_sl_l2", "log(sK/sL), t-2"),
          ("sksl_x_high", "x high repression (t-1)"), ("sksl_l1_x_high", "t-1 x high repression"),
          ("sksl_l2_x_high", "t-2 x high repression"), ("high_rep_l1", "High repression (t-1)")]
rows = coef_rows(m, labels, bootstrap_vars=("ln_sk_sl",))
# sums of lag coefficients
sum_rows = []
for res in m:
    if res is None:
        sum_rows.append(""); continue
    b = sum(res.params.get(v, 0) for v in base_x)
    sum_rows.append(f"{b:.3f}")
rows.insert(len(rows) - 2, ["Sum of lags 0-2 (low repression)"] + sum_rows)
write_table(rows, ["(1) pooled", "(2) interaction", "(3) low rep.", "(4) high rep.", "(5) MMP protests", "(6) pro-dem. mob."],
            "tab_h1b_mobilization", "H1b. Mobilization and functional inequality, by repression regime", "tab:h1b",
            "Dependent variable: V-Dem mass mobilization (1)-(4), log(1+MMP protest events) (5), V-Dem pro-democracy mobilization (6). "
            "High repression = lagged physical repression above the sample median. Standardized regressors, two-way FE, "
            "country-clustered SEs, wild-cluster bootstrap p in brackets. Observed mobilization is a proxy, not the successful-revolt hazard; lag-sum signs are descriptive comparisons under the stated measurement assumptions.")
h1b = m

# ------------------------------------------------------------------ H2: repression choice
r = []
r.append(twfe(p, "rep_phys", ["k_l1"] + CONTROLS))
r.append(twfe(p, "rep_phys", ["k_l1", "threat_mob_l1", "k_x_threat"] + CONTROLS))
r.append(twfe(p, "rep_phys", ["k_l1", "threat_mob_l1", "k_x_threat", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
r.append(twfe(p, "rep_cso", ["k_l1", "threat_mob_l1", "k_x_threat", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
r.append(twfe(p, "pts_mean", ["k_l1", "threat_mob_l1", "k_x_threat", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
r.append(twfe(p, "rep_index", ["k_l1", "threat_mob_l1", "k_x_threat", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
# anti-system movements as the threat proxy: defined for every country (the mobilization-based proxy drops
# countries that are repressive in every year, a selection on the outcome)
r.append(twfe(p, "rep_phys", ["k_l1", "antisys_l5", "k_x_antisys", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
r.append(twfe(p, "rep_index", ["k_l1", "antisys_l5", "k_x_antisys", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
labels = [("k_l1", "log K/L (t-1)"), ("threat_mob_l1", "Threat: past mobilization (t-1)"), ("k_x_threat", "K/L x past mobilization"),
          ("antisys_l5", "Threat: anti-system movements (t-5..t-1)"), ("k_x_antisys", "K/L x anti-system"),
          ("milexp_gdp_l1", "Military expenditure % GDP (t-1)"), ("mil_support_l1", "Military is regime's main support (t-1)")]
rows = coef_rows(r, labels, bootstrap_vars=("k_l1", "k_x_threat", "k_x_antisys"))
write_table(rows, ["(1) physical", "(2) physical", "(3) physical", "(4) CSO", "(5) PTS", "(6) composite", "(7) physical", "(8) composite"],
            "tab_h2_repression", "H2. Repression, capital per worker and the threat of revolt", "tab:h2rep",
            "Dependent variables: physical repression (1 - v2x_clphy), CSO repression (-v2csreprss), Political Terror Scale mean, composite index. "
            "Threat proxies: columns (2)-(6) lagged 10-year mean of mass mobilization in low-repression years (defined for 11 countries); "
            "columns (7)-(8) lagged 5-year mean of V-Dem anti-system movement strength (all 18 countries). Standardized regressors, two-way FE, "
            "country-clustered SEs, wild-cluster bootstrap p in brackets. Linear proxy associations are not literal coefficient restrictions from Proposition 6; main effects refer to raw moderator zero when interactions are present.")
h2r = r

# H2: redistribution and the substitutability test
q = []
q.append(twfe(p, "abs_red", ["k_l1"] + CONTROLS))
q.append(twfe(p, "abs_red", ["k_l1", "rep_phys_l1", "k_x_rep"] + CONTROLS))
q.append(twfe(p, "socexp_gdp", ["k_l1", "rep_phys_l1", "k_x_rep"] + CONTROLS))
q.append(twfe(p, "tax_gdp", ["k_l1", "rep_phys_l1", "k_x_rep"] + CONTROLS))
q.append(twfe(p[p.high_rep_l1 == 0], "abs_red", ["k_l1"] + CONTROLS))
q.append(twfe(p[p.high_rep_l1 == 1], "abs_red", ["k_l1"] + CONTROLS))
labels = [("k_l1", "log K/L (t-1)"), ("rep_phys_l1", "Physical repression (t-1)"), ("k_x_rep", "K/L x repression")]
rows = coef_rows(q, labels, bootstrap_vars=("rep_phys_l1",))
write_table(rows, ["(1) redistribution", "(2) redistribution", "(3) social exp.", "(4) tax revenue", "(5) low rep.", "(6) high rep."],
            "tab_h2_redistribution", "H2. Redistribution, capital and repression: the substitutability test", "tab:h2red",
            "Dependent variables: SWIID absolute redistribution (Gini market minus Gini disposable), social expenditure % GDP, tax revenue % GDP. "
            "Conditional redistribution associations; noisy political indices and fiscal measures need not obey the model’s structural policy-regime restriction. "
            "Low-repression observations are not identified structural redistribution regimes.")
h2q = q

# relative choice
c = []
c.append(twfe(p, "rep_choice", ["k_l1"] + CONTROLS))
c.append(twfe(p, "rep_choice", ["k_l1", "threat_mob_l1", "k_x_threat"] + CONTROLS))
c.append(twfe(p, "rep_choice", ["k_l1", "threat_mob_l1", "k_x_threat", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
c.append(twfe(p, "rep_choice", ["k_l1", "antisys_l5", "k_x_antisys", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS))
dk = driscoll_kraay(p, "rep_choice", ["k_l1", "antisys_l5", "k_x_antisys", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS)
labels = [("k_l1", "log K/L (t-1)"), ("threat_mob_l1", "Threat: past mobilization (t-1)"), ("k_x_threat", "K/L x past mobilization"),
          ("antisys_l5", "Threat: anti-system movements"), ("k_x_antisys", "K/L x anti-system"),
          ("milexp_gdp_l1", "Military expenditure % GDP (t-1)"), ("mil_support_l1", "Military support (t-1)")]
rows = coef_rows(c, labels, bootstrap_vars=("k_l1", "k_x_threat", "k_x_antisys"))
rows.insert(len(rows) - 2, ["Driscoll-Kraay SE, log K/L"] + ["", "", "", f"({dk.bse['k_l1']:.3f})"])
write_table(rows, ["(1)", "(2)", "(3)", "(4)"], "tab_h2_choice",
            "H2. Relative choice: standardized repression minus standardized redistribution", "tab:h2choice",
            "Dependent variable: z(physical repression) - z(absolute redistribution). Positive coefficients mean the policy mix tilts toward repression. "
            "Two-way FE, country-clustered SEs; Driscoll-Kraay SE (3 lags) reported for column (4). Columns (2)-(3) use the mobilization-based "
            "threat proxy (11 countries), column (4) the anti-system-movement proxy (18 countries).")
h2c = c

# ------------------------------------------------------------------ IV
iv = []
# first-stage search over instrument timing (reported in full: the level, 5-year mean and 5-year change of the commodity index)
fs_rows = []
for z in ["z_commodity_l1", "z_commodity_ma5_l1", "z_commodity_d5_l1"]:
    for xx in ["k_l1", "dk"]:
        t = twfe(p, xx, [z] + CONTROLS)
        if t is not None:
            fs_rows.append([f"{xx} on {z}", fmt(t.params[z], t.bse[z], t.pvalues[z]), f"{(t.tvalues[z] ** 2):.1f}", str(int(t.nobs))])
t = twfe(p, "ln_robots_pw_l1", ["z_robot_l1"] + CONTROLS)
fs_rows.append(["ln robots pw on z_robot", fmt(t.params["z_robot_l1"], t.bse["z_robot_l1"], t.pvalues["z_robot_l1"]), f"{(t.tvalues['z_robot_l1'] ** 2):.1f}", str(int(t.nobs))])
write_table(fs_rows, ["coefficient (SE)", "F (= t^2)", "N"], "tab_iv_firststages", "First stages for every instrument timing tried", "tab:fs",
            "Two-way FE regressions of the endogenous regressor on the instrument and controls; F is the cluster-robust t-statistic squared. "
            "All attempted timings are reported. Selecting the largest first-stage F after inspection remains exploratory.")
iv.append(iv2sls(p, "rep_phys", "k_l1", ["z_commodity_d5_l1"], CONTROLS))
iv.append(iv2sls(p, "rep_choice", "k_l1", ["z_commodity_d5_l1"], CONTROLS))
iv.append(iv2sls(p, "labsh_any", "k_l1", ["z_commodity_d5_l1"], CONTROLS))
iv.append(iv2sls(p, "rep_phys", "ln_robots_pw_l1", ["z_robot_l1"], CONTROLS))
iv.append(iv2sls(p, "labsh_any", "ln_robots_pw_l1", ["z_robot_l1"], CONTROLS))
iv.append(iv2sls(p, "rep_phys", "ln_robots_pw_l1", ["z_robot_l1"], CONTROLS + ["china_x_manuf"]))
rows = []
r1 = ["Instrumented regressor"]
for res in iv:
    if res is None:
        r1.append("")
        continue
    v = [x for x in res.params.index if x in ("k_l1", "ln_robots_pw_l1")][0]
    r1.append(fmt(res.params[v], res.std_errors[v], res.pvalues[v]))
rows.append(r1)
rows.append(["First-stage F (cluster-robust)"] + [f"{res._fsF:.1f}" if res is not None else "" for res in iv])
rows.append(["Observations"] + [str(res._n) if res is not None else "" for res in iv])
# reduced forms
rf = [twfe(p, "rep_phys", ["z_commodity_d5_l1"] + CONTROLS), twfe(p, "rep_choice", ["z_commodity_d5_l1"] + CONTROLS),
      twfe(p, "labsh_any", ["z_commodity_d5_l1"] + CONTROLS), twfe(p, "rep_phys", ["z_robot_l1"] + CONTROLS),
      twfe(p, "labsh_any", ["z_robot_l1"] + CONTROLS), twfe(p, "rep_phys", ["z_robot_l1", "china_x_manuf"] + CONTROLS)]
rr = ["Reduced form (instrument)"]
for res in rf:
    if res is None:
        rr.append(""); continue
    v = [x for x in res.params.index if x.startswith("z_")][0]
    rr.append(fmt(res.params[v], res.bse[v], res.pvalues[v]))
rows.append(rr)
write_table(rows, ["(1) rep. | K/L", "(2) choice | K/L", "(3) labsh | K/L", "(4) rep. | robots", "(5) labsh | robots", "(6) rep. | robots + manuf. trend"],
            "tab_iv", "Instrumental-variable estimates", "tab:iv",
            "2SLS with two-way FE and country-clustered SEs. Columns (1)-(3) instrument log K/L (t-1) with the lagged 5-year change of the commodity price index "
            "(initial export shares x world real prices), the only timing with a first-stage F above 10 (see the first-stage table); its first-stage sign is negative. Columns (4)-(6) instrument the log robot stock per worker with the shift-share "
            "(1993-97 manufacturing share x donor robot stock per worker). Column (6) adds manufacturing share x log trend, not observed China exposure. "
            "Standardized regressors and instruments.")

# ------------------------------------------------------------------ coefficient plot
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
def cplot(ax, res_list, var, names, title):
    ys = np.arange(len(res_list))
    for i, res in enumerate(res_list):
        if res is None or var not in res.params: continue
        b, se = res.params[var], res.bse[var]
        ax.errorbar(b, i, xerr=1.96 * se, fmt="o", color="#1f4e79", capsize=2)
    ax.set_yticks(ys); ax.set_yticklabels([L(n) for n in names]); ax.axvline(0, color="k", lw=0.6); ax.set_title(L(title)); ax.invert_yaxis()
cplot(axes[0], h1a, "k_l1", ["PWT/ILO", "+robots", "+routine", "ILO share", "1960-2024", "PWT varying"], "H1a: labour share on log K/L")
cplot(axes[1], h2r, "k_l1", ["physical", "physical+threat", "physical+R", "CSO", "PTS", "composite", "physical (anti-sys.)", "composite (anti-sys.)"], "H2: repression on log K/L")
cplot(axes[2], h2q[1:4] + h2q[4:6], "rep_phys_l1", ["redistribution", "social exp.", "tax", "low rep. (K/L)", "high rep. (K/L)"], "H2: redistribution on lagged repression")
for ax in axes: ax.set_xlabel(L("outcome units per SD of regressor, 95% CI"))
fig.tight_layout(); fig.savefig(FIGURES / "fig_coef_h1h2.pdf"); fig.savefig(FIGURES / "fig_coef_h1h2.svg")

# summary json for the manuscript
import json
summary = {}
def grab(res, v):
    return None if (res is None or v not in res.params) else dict(b=float(res.params[v]), se=float(res.bse[v]), p=float(res.pvalues[v]), n=int(res.nobs))
summary["h1a"] = [grab(x, "k_l1") for x in h1a]
summary["h1a_robots"] = [grab(x, "ln_robots_pw_l1") for x in h1a]
summary["h1b_lowrep_sum"] = sum_rows
summary["h2_rep_k"] = [grab(x, "k_l1") for x in h2r]
summary["h2_rep_threat"] = [grab(x, "threat_mob_l1") for x in h2r]
summary["h2_rep_kxthreat"] = [grab(x, "k_x_threat") for x in h2r]
summary["h2_red_rep"] = [grab(x, "rep_phys_l1") for x in h2q]
summary["h2_red_k"] = [grab(x, "k_l1") for x in h2q]
summary["h2_choice_k"] = [grab(x, "k_l1") for x in h2c]
summary["iv"] = [None if res is None else dict(b=float(res.params.iloc[-1]) if False else float(res.params[[x for x in res.params.index if x in ('k_l1','ln_robots_pw_l1')][0]]),
                                          F=float(res._fsF), n=int(res._n)) for res in iv]
(TABLES / "summary_fe.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("done: H1/H2 tables written")

# ------------------------------------------------------------------ H3b (T7): mobilization hazard rising with capital in redistributive regimes
hb = [twfe(p[p.high_rep_l1 == 0], "mob", ["k_l1"] + CONTROLS), twfe(p[p.high_rep_l1 == 1], "mob", ["k_l1"] + CONTROLS),
      twfe(p, "mob", ["k_l1", "k_x_high", "high_rep_l1"] + CONTROLS),
      twfe(p[p.high_rep_l1 == 0], "ln_mmp_protests", ["k_l1"] + CONTROLS), twfe(p[p.high_rep_l1 == 1], "ln_mmp_protests", ["k_l1"] + CONTROLS)]
rows = coef_rows(hb, [("k_l1", "log K/L (t-1)"), ("k_x_high", "K/L x high repression (t-1)"), ("high_rep_l1", "High repression (t-1)")], bootstrap_vars=("k_l1",))
write_table(rows, ["(1) mob., low rep.", "(2) mob., high rep.", "(3) mob., interaction", "(4) MMP, low rep.", "(5) MMP, high rep."],
            "tab_h3b_hazard", "H3b. Mobilization and capital per worker by repression regime", "tab:h3b",
            "Descriptive observed-mobilization comparisons; the successful-revolt hazard is not observed, and index-based groups are not identified structural regimes. "
            "Two-way FE, standardized regressors, country-clustered SEs, wild-cluster bootstrap p in brackets.")
summary["h3b_k"] = [grab(x, "k_l1") for x in hb]
(TABLES / "summary_fe.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("done: H3b")
