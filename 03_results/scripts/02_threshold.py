"""H3: joint response of automation and repression to capital; Hansen (1999) panel threshold.

Tables: tab_h3_joint, tab_h3_threshold
Figures: fig_threshold_lr (likelihood-ratio profile), fig_h3_binned
Run: python 03_results/scripts/02_threshold.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '02_data' / 'scripts'))
from figlabels import L
from common import CONTROLS, FIGURES, RNG, SEED, TABLES, coef_rows, load_panel, twfe, write_table, zscore

# Fuentes TrueType incrustadas en PDF y EPS: las Type3 que matplotlib escribe por omision
# las marca el preflight de la editorial. No altera datos ni ejes.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

p = load_panel((1990, 2024))

# ------------------------------------------------------------------ joint response (eq. 5)
j = [twfe(p, "ln_robots_pw", ["k_l1"] + CONTROLS), twfe(p, "routine_share", ["k_l1"] + CONTROLS),
     twfe(p, "rep_phys", ["k_l1"] + CONTROLS), twfe(p, "rep_index", ["k_l1"] + CONTROLS)]
rows = coef_rows(j, [("k_l1", "log K/L (t-1)")], bootstrap_vars=("k_l1",))
# residual correlation between the automation and repression equations
d1, d2 = j[0]._d, j[2]._d
common_idx = d1.index.intersection(d2.index)
rc = np.corrcoef(j[0].resid.loc[common_idx], j[2].resid.loc[common_idx])[0, 1]
rows.insert(len(rows) - 2, ["Residual corr. (robots, repression)", f"{rc:.3f}", "", "", ""])
write_table(rows, ["(1) log robots pw", "(2) routine share", "(3) physical rep.", "(4) rep. composite"],
            "tab_h3_joint", "H3. Automation and repression respond jointly to capital per worker", "tab:h3joint",
            "Two-way FE, standardized regressors, country-clustered SEs, wild-cluster bootstrap p in brackets. "
            "Residual correlation is descriptive; its sign is not a direct implication of the structural model.")


# ------------------------------------------------------------------ Hansen (1999) threshold
def within(df, cols):
    from common import fe_residualize
    return pd.DataFrame(fe_residualize(df[cols], df), index=df.index, columns=cols)


def hansen(df, y, x, q, controls, grid_q=(0.15, 0.85), n_grid=60, B=999, seed=SEED):
    # Construct regime-specific regressors BEFORE removing the fixed effects.
    cols = list(dict.fromkeys([y, x, q] + controls))
    d = df.dropna(subset=cols + ['iso3', 'year']).copy()
    qv = d[q].to_numpy().copy()  # original log-capital / threat units
    xv = zscore(d[x]).to_numpy()
    C = d[controls].apply(zscore).to_numpy()
    F = np.column_stack([np.ones(len(d)), pd.get_dummies(d.iso3, drop_first=True),
                         pd.get_dummies(d.year, drop_first=True)]).astype(float)
    Q, _ = np.linalg.qr(F, mode='reduced')
    def residual(a):
        return a - Q @ (Q.T @ a)
    yv = residual(d[y].to_numpy())
    X0 = residual(np.column_stack([xv, C]))
    gammas = np.unique(np.quantile(qv, np.linspace(*grid_q, n_grid)))
    designs = [residual(np.column_stack([xv * (qv <= g), xv * (qv > g), C])) for g in gammas]
    inverses = [np.linalg.pinv(X) for X in designs]
    def scores(yy):
        return np.array([np.sum((yy - X @ (pinv @ yy))**2) for X, pinv in zip(designs, inverses)])
    S = scores(yv); best = int(S.argmin()); S1 = S[best]
    X_hat = designs[best]
    b0 = np.linalg.lstsq(X0, yv, rcond=None)[0]
    e0 = yv - X0 @ b0; S0 = float(e0 @ e0)
    n = len(d); F1 = n * (S0 - S1) / S1
    rng = np.random.default_rng(seed)
    groups, codes = np.unique(d.iso3, return_inverse=True)
    Fb = []
    for _ in range(B):
        yb = X0 @ b0 + residual(e0 * rng.choice([-1., 1.], size=len(groups))[codes])
        S1b = scores(yb).min()
        e0b = yb - X0 @ np.linalg.lstsq(X0, yb, rcond=None)[0]
        Fb.append(n * (float(e0b @ e0b) - S1b) / S1b)
    import statsmodels.api as sm
    # Full dummy regression gives CR1 its correct parameter count.
    ghat = gammas[best]
    full = np.column_stack([xv * (qv <= ghat), xv * (qv > ghat), C, F])
    if np.linalg.matrix_rank(full) != full.shape[1]:
        raise ValueError('Threshold design is not identified')
    res = sm.OLS(d[y].to_numpy(), full).fit(cov_type='cluster', cov_kwds={'groups': d.iso3})
    LR = n * (S - S1) / S1
    return dict(gamma=float(ghat), ci=(float('nan'), float('nan')),
                F=float(F1), p=float((1 + np.sum(np.array(Fb) >= F1))/(B+1)),
                b_low=float(res.params[0]), se_low=float(res.bse[0]),
                b_high=float(res.params[1]), se_high=float(res.bse[1]), n=n,
                gammas=gammas, LR=LR, q_share_high=float((qv > ghat).mean()))


specs = [("rep_phys", "k_l1", "k_l1", "Physical rep., threshold in K/L"),
         ("rep_index", "k_l1", "k_l1", "Composite rep., threshold in K/L"),
         ("rep_phys", "k_l1", "threat_mob_l1", "Physical rep., threshold in threat"),
         ("ln_robots_pw", "k_l1", "k_l1", "Robots, threshold in K/L"),
         ("abs_red", "k_l1", "k_l1", "Redistribution, threshold in K/L")]
out = []; profiles = {}
for y, x, q, lab in specs:
    r = hansen(p, y, x, q, CONTROLS)
    out.append([lab, f"{r['gamma']:.3f}", "not reported", f"{r['q_share_high']:.2f}",
                f"{r['b_low']:.3f} ({r['se_low']:.3f})", f"{r['b_high']:.3f} ({r['se_high']:.3f})",
                f"{r['F']:.1f}", f"{r['p']:.3f}", str(r["n"])])
    profiles[lab] = (r["gammas"], r["LR"])
write_table(out, ["threshold", "threshold interval", "share above", "slope below", "slope above", "F", "bootstrap p", "N"],
            "tab_h3_threshold", "H3. Exploratory two-way fixed-effects threshold regressions", "tab:h3thr",
            "Regime-specific regressors are formed before exact country/year projection. Thresholds are in original "
            "log-capital or threat units; slopes are per SD of log capital, outcomes in original units. "
            "Grid: 15th-85th percentiles; 999 country-level wild draws under the no-threshold null. "
            "Slope SEs are conditional on the selected threshold and exclude threshold-selection uncertainty. "
            "No threshold confidence interval is claimed: Hansen's original reference distribution is not established "
            "for this unbalanced, two-way FE, clustered extension. These are exploratory slope-break tests, not a test of a structural regime switch.")

fig, axes = plt.subplots(3, 2, figsize=(7, 7))
for ax, (lab, (gm, lr)) in zip(axes.flat, profiles.items()):
    ax.plot(gm, lr, color="#1f4e79")
    title = lab.replace(", threshold in ", "\nThreshold in ")
    ax.set_title(L(title), fontsize=10)
    ax.set_xlabel(L("Threshold"), fontsize=9); ax.set_ylabel(L("LR"), fontsize=9)
    ax.tick_params(axis="both", labelsize=9)
axes.flat[-1].set_visible(False)
fig.tight_layout(pad=1.15, h_pad=1.7, w_pad=1.5)
fig.savefig(FIGURES / "fig_threshold_lr.pdf"); fig.savefig(FIGURES / "fig_threshold_lr.svg")

# binned within-country relationship (the model's step)
fig, ax = plt.subplots(figsize=(5, 3.6))
d = p.dropna(subset=["rep_phys", "k_l1"]).copy()
d["rep_w"] = within(d, ["rep_phys"])["rep_phys"]
d["kbin"] = pd.qcut(d.k_l1, 12)
b = d.groupby("kbin", observed=True).agg(x=("k_l1", "median"), y=("rep_w", "mean"), se=("rep_w", lambda s: s.std() / np.sqrt(len(s))))
ax.errorbar(b.x, b.y, yerr=1.96 * b.se, fmt="o", color="#1f4e79", capsize=2)
ax.axhline(0, color="k", lw=0.5); ax.set_xlabel(L("log K/L (t-1), 12 bins")); ax.set_ylabel(L("physical repression, two-way demeaned"))
ax.set_title(L("Within-country repression against capital per worker"), fontsize=9)
fig.tight_layout(); fig.savefig(FIGURES / "fig_h3_binned.pdf"); fig.savefig(FIGURES / "fig_h3_binned.svg")
(TABLES / "summary_threshold.json").write_text(json.dumps({s[3]: dict(zip(["threshold", "ci", "share_above", "slope_low", "slope_high", "F", "p", "N"], o[1:])) for s, o in zip(specs, out)}, indent=1), encoding="utf-8")
print("done: threshold tables")
