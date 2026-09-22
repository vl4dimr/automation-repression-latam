"""H3: causal forest (Athey-Tibshirani-Wager generalized random forest via econml.CausalForestDML).

Treatment: log K/L (t-1), continuous. Outcome: physical repression. Nuisances residualized on
country and year means before the forest (two-way FE analogue), then cross-fitted with random forests.
Outputs: tab_cate_country (.tex/.csv), fig_cate_country, fig_cate_by_k, fig_cate_by_threat, tab_cf_calibration
Run: python 03_results/scripts/03_causal_forest.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '02_data' / 'scripts'))
from figlabels import L
from common import CONTROLS, FIGURES, NAMES, SEED, TABLES, load_panel, write_table

from econml.dml import CausalForestDML

# Fuentes TrueType incrustadas en PDF y EPS: las Type3 que matplotlib escribe por omision
# las marca el preflight de la editorial. No altera datos ni ejes.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

p = load_panel((1990, 2024))
# the mobilization-based threat proxy is undefined for always-repressive countries (selection on the outcome), so the
# forest uses the anti-system-movement proxy, defined for all 18 countries
MODS = ["k_l2", "antisys_l5", "milexp_gdp_l1", "dem_l1", "ln_gdppc", "resource_rents", "urban_share", "decade",
        "v2x_polyarchy_l1", "abs_red_l1"]
d = p.dropna(subset=["rep_phys", "k_l1"] + MODS).copy()


def demean2(s, df):
    from common import fe_residualize
    return fe_residualize(s, df)


Y = demean2(d.rep_phys, d)
T = demean2(d.k_l1, d)
X = d[MODS].astype(float).values
W = None

cf = CausalForestDML(model_y=RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEED),
                     model_t=RandomForestRegressor(n_estimators=300, min_samples_leaf=10, random_state=SEED),
                     n_estimators=2000, min_samples_leaf=10, max_depth=None, honest=True, inference=False,
                     cv=GroupKFold(n_splits=5), random_state=SEED, discrete_treatment=False)
cf.fit(Y, T, X=X, W=W, groups=d.iso3.to_numpy())
cate = cf.effect(X)
lb = ub = np.full(len(d), np.nan)
ate = float(np.mean(cate)); ate_lb = ate_ub = float("nan")
d["cate"] = cate; d["cate_lb"] = lb; d["cate_ub"] = ub

# ---------------- CATE by country
bc = d.groupby("iso3").agg(cate=("cate", "mean"), lb=("cate_lb", "mean"), ub=("cate_ub", "mean"), n=("cate", "size"),
                           k=("k_l1", "mean"), threat=("antisys_l5", "mean")).sort_values("cate")
rows = [[NAMES[i], f"{r.cate:.3f}", "not estimated", f"{r.k:.2f}", f"{r.threat:.2f}", str(int(r.n))] for i, r in bc.iterrows()]
rows.append(["Average fitted slope", f"{ate:.3f}", "not estimated", "", "", str(len(d))])
write_table(rows, ["CATE", "Cluster-valid interval", "mean log K/L", "mean threat", "N"], "tab_cate_country",
            "H3. Exploratory forest slopes by country (not identified causal effects)", "tab:cate",
            "CausalForestDML, 2,000 honest trees; nuisance folds keep countries together. "
            "Exact full-sample country/year projection precedes fitting, so this is exploratory panel preprocessing, "
            "not fully cross-fitted fixed-effect estimation. Capital t-2 replaces the treatment itself among modifiers. "
            "Slopes are index points per log point of lagged capital. Country point estimates average fitted slopes; "
            "no cluster-valid forest confidence intervals or country ranking significance are claimed.")

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(bc.cate, np.arange(len(bc)), "o", color="#1f4e79")
ax.set_yticks(np.arange(len(bc))); ax.set_yticklabels([NAMES[i] for i in bc.index]); ax.axvline(0, color="k", lw=0.6)
ax.axvline(ate, color="#c0504d", ls="--", lw=0.8, label=L("Mean fitted slope") + f" = {ate:.3f}")
ax.set_xlabel(L("Exploratory slope: index points per log point of capital")); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIGURES / "fig_cate_country.pdf"); fig.savefig(FIGURES / "fig_cate_country.svg")

# ---------------- CATE against capital and threat (the model's threshold curve)
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, v, lab in zip(axes, ["k_l1", "antisys_l5", "milexp_gdp_l1"], ["log K/L (t-1)", "anti-system movements (t-5..t-1)", "military expenditure % GDP (t-1)"]):
    dd = d.sort_values(v)
    dd["bin"] = pd.qcut(dd[v], 10, duplicates="drop")
    b = dd.groupby("bin", observed=True).agg(x=(v, "median"), y=("cate", "mean"), lo=("cate_lb", "mean"), hi=("cate_ub", "mean"))
    ax.fill_between(b.x, b.lo, b.hi, color="#1f4e79", alpha=0.15); ax.plot(b.x, b.y, "o-", color="#1f4e79")
    ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(lab); ax.set_ylabel(L("Exploratory fitted slope"))
axes[0].set_title(L("Capital profile (descriptive)"), fontsize=9); axes[1].set_title(L("Threat profile (descriptive)"), fontsize=9); axes[2].set_title(L("Military spending profile (descriptive)"), fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES / "fig_cate_by_k.pdf"); fig.savefig(FIGURES / "fig_cate_by_k.svg")

# The previous row-split calibration was not a cluster-held-out orthogonal BLP.
write_table([["Calibration inference", "not estimated", "not estimated"]],
            ["coefficient (SE)", "p-value"], "tab_cf_calibration",
            "Forest calibration inference withdrawn", "tab:cfblp",
            "The earlier row-split test reused countries and did not residualize evaluation observations "
            "on independently trained nuisance functions. Its p-value is not used.")

# variable importance
imp = pd.Series(cf.feature_importances_, index=MODS).sort_values(ascending=False)
imp.to_csv(TABLES / "tab_cf_importance.csv")
d[["iso3", "year", "cate", "cate_lb", "cate_ub"] + MODS].to_csv(TABLES / "cate_country_year.csv", index=False)
(TABLES / "summary_cf.json").write_text(json.dumps(dict(ate=float(ate), ate_ci=[float(ate_lb), float(ate_ub)], n=int(len(d)),
                                                        blp_beta2=None, blp_p2=None,
                                                        top_importance=imp.head(4).round(3).to_dict(),
                                                        country=bc[["cate", "lb", "ub"]].round(3).to_dict("index")), indent=1), encoding="utf-8")
print("done: exploratory forest; mean slope", ate)
