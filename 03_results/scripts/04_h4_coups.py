"""H4: coups in electoral regimes, 1960-2024; event studies around coups and autocratization onsets.

Tables: tab_h4_hazard, tab_h4_events
Figures: fig_event_study
Run: python 03_results/scripts/04_h4_coups.py
"""
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '02_data' / 'scripts'))
from figlabels import L
from common import CONTROLS, FIGURES, NAMES, TABLES, coef_rows, load_panel, twfe, write_table, zscore

# Fuentes TrueType incrustadas en PDF y EPS: las Type3 que matplotlib escribe por omision
# las marca el preflight de la editorial. No altera datos ni ejes.
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

p = load_panel((1960, 2024))
p["tax_any_l1"] = p.groupby("iso3").tax_any.shift(1)
p["k_x_tax"] = p.k_l1 * p.tax_any_l1
p["poly_l1"] = p.v2x_polyarchy_l1
CTRL60 = ["ln_gdppc", "urban_share", "trade_open"]   # resource rents start 1970 in WDI

# ------------------------------------------------------------------ hazard (LPM with two-way FE)
samples = {"multiparty": p[p.multiparty_l1 == 1], "polyarchy>0.4": p[p.poly_l1 > 0.4], "RoW democracy": p[p.dem_l1 == 1]}
h = []
h.append(twfe(samples["multiparty"], "coup", ["k_l1"] + CTRL60))
h.append(twfe(samples["multiparty"], "coup", ["k_l1", "tax_any_l1", "k_x_tax"] + CTRL60))
h.append(twfe(samples["multiparty"], "coup", ["k_l1", "tax_any_l1", "k_x_tax", "milexp_gdp_l1", "mil_support_l1"] + CTRL60))
h.append(twfe(samples["multiparty"], "coup_att", ["k_l1", "tax_any_l1", "k_x_tax"] + CTRL60))
h.append(twfe(samples["polyarchy>0.4"], "coup", ["k_l1", "tax_any_l1", "k_x_tax"] + CTRL60))
h.append(twfe(samples["multiparty"][samples["multiparty"].year >= 1990], "aut_onset", ["k_l1", "tax_any_l1", "k_x_tax"] + CTRL60))
mp = samples["multiparty"]
h.append(twfe(mp[mp.year <= 1990], "coup", ["k_l1", "tax_any_l1", "k_x_tax"] + CTRL60))
h.append(twfe(mp[mp.year > 1990], "coup_att", ["k_l1", "tax_any_l1", "k_x_tax"] + CTRL60))
labels = [("k_l1", "log K/L (t-1)"), ("tax_any_l1", "Tax revenue % GDP (t-1)"), ("k_x_tax", "K/L x tax"),
          ("milexp_gdp_l1", "Military expenditure % GDP (t-1)"), ("mil_support_l1", "Military support (t-1)")]
rows = coef_rows(h, labels, bootstrap_vars=("k_l1", "tax_any_l1"))
ev = ["Events"] + [str(int(res._d[res._y].sum())) if res is not None else "" for res in h]
rows.insert(len(rows) - 2, ev)
write_table(rows, ["(1) coup", "(2) coup", "(3) coup", "(4) attempt", "(5) coup, poly>0.4", "(6) aut. onset 1990+", "(7) coup, <=1990", "(8) attempt, >1990"],
            "tab_h4_hazard", "H4. Coups in regimes with prior multiparty institutions, 1960-2024 (linear probability, two-way FE)", "tab:h4",
            "Sample: country-years with multiparty elections in t-1 (V-Dem RoW >= 1), column (5) electoral democracy index > 0.4 in t-1, "
            "column (6) ERT autocratization onset 1990-2024, columns (7)-(8) Cold War and post-Cold War sub-periods. Coefficients x 100 give percentage points per standard deviation. Tax revenue: CEPALSTAT, "
            "Andersson & Brambor, WDI spliced. Country-clustered SEs, wild-cluster bootstrap p in brackets. These all-coup associations do not identify capitalist initiators; actual tax revenue is not the model’s feasible tax ceiling. Interaction contrasts govern marginal interpretations.")

# conditional logit check (countries with events only)
from statsmodels.discrete.conditional_models import ConditionalLogit
s = samples["multiparty"].dropna(subset=["coup", "k_l1", "tax_any_l1"] + CTRL60).copy()
s = s[s.groupby("iso3").coup.transform("sum") > 0]
try:
    Xc = s[["k_l1", "tax_any_l1"] + CTRL60].apply(zscore)
    cl = ConditionalLogit(s.coup.values, Xc.values, groups=s.iso3.values).fit(disp=0)
    clog = dict(k=float(cl.params[0]), k_se=float(cl.bse[0]), tax=float(cl.params[1]), tax_se=float(cl.bse[1]), n=int(len(s)), countries=int(s.iso3.nunique()))
except Exception as e:  # noqa
    clog = dict(error=str(e))

# ------------------------------------------------------------------ event studies (Sun-Abraham interaction-weighted)
def event_profile(df, y, event_col, pre=6, post=4, min_year=1960):
    # First observed event per country; complete windows and no repeat event in window.
    source = df[df.year >= min_year].copy()
    event_years = source[source[event_col] == 1].groupby('iso3').year.apply(list)
    never_ids = set(source.iso3.unique()) - set(event_years.index)
    d = source.dropna(subset=[y]).copy()
    frames = []
    for iso, years in event_years.items():
        e = min(years)
        if any(e < t <= e + post for t in years):
            continue
        g = d[(d.iso3 == iso) & d.year.between(e-pre, e+post)].copy()
        if set(g.year) != set(range(e-pre, e+post+1)):
            continue
        g['rel'] = g.year - e
        frames.append(g)
    if len(frames) < 5:
        raise ValueError(f'Only {len(frames)} eligible event countries; inference withheld')
    treated = pd.concat(frames)
    never = d[d.iso3.isin(never_ids) & d.year.isin(treated.year)].copy()
    if never.iso3.nunique() < 2:
        raise ValueError('Fewer than two never-event control countries')
    never['rel'] = np.nan
    dd = pd.concat([treated, never], ignore_index=True)
    rel_vals = [r for r in range(-pre, post+1) if r != -1]
    for r in rel_vals:
        dd[f'event_{r}'] = (dd.rel == r).astype(float)
    res = twfe(dd, y, [f'event_{r}' for r in rel_vals], std=False)
    out = {r: (res.params[f'event_{r}'], res.bse[f'event_{r}']) for r in rel_vals}
    out[-1] = (0., 0.)
    return pd.DataFrame(out, index=['b', 'se']).T.sort_index(), len(frames)


events = [("coup", "Successful coup (1960-2024)", 1960), ("aut_onset", "Autocratization onset (1990-2024)", 1990)]
outcomes = [("k", "log K/L"), ("ln_robots_pw", "log robots per worker"), ("labsh_any", "labour share"), ("tax_any", "tax revenue % GDP"),
            ("abs_red", "redistribution"), ("rep_phys", "physical repression")]
fig, axes = plt.subplots(2, 3, figsize=(7, 5.5))
# El titulo va en el pie numerado del manuscrito, no sobre la figura (guia de la revista).
es_rows = []
for i, (ecol, elab, y0) in enumerate(events):
    for j, (y, ylab) in enumerate(outcomes):
        # Keep calculations and the table for both event types. The figure shows
        # only the six estimated autocratization outcomes, at readable page size.
        ax = axes.flat[j] if ecol == "aut_onset" else None
        try:
            tab, nev = event_profile(p, y, ecol, min_year=y0)
            if ax is not None:
                ax.errorbar(tab.index, tab.b, yerr=1.96 * tab.se, fmt="o-", color="#1f4e79", capsize=2, ms=3)
            pre_mean = tab.loc[tab.index < -1, "b"].mean()
            es_rows.append([elab, ylab, f"{pre_mean:.3f}", f"{tab.loc[-6, 'b']:.3f} ({tab.loc[-6, 'se']:.3f})", f"{tab.loc[-3, 'b']:.3f} ({tab.loc[-3, 'se']:.3f})",
                            f"{tab.loc[2, 'b']:.3f} ({tab.loc[2, 'se']:.3f})", str(nev)])
        except ValueError as e:
            es_rows.append([elab, ylab, "not estimated", "not estimated", "not estimated", "not estimated", str(e)])
            if ax is not None:
                ax.text(0.5, 0.5, textwrap.fill(str(e), 22), transform=ax.transAxes, ha="center", va="center", fontsize=9)
                ax.set_title(L(ylab), fontsize=10)
                ax.set_axis_off()
            continue
        if ax is not None:
            ax.axvline(-0.5, color="k", lw=0.6, ls="--"); ax.axhline(0, color="k", lw=0.5)
            ax.set_title(L(ylab) + f"\n{nev} " + L("event countries"), fontsize=10)
            ax.set_xlabel(L("Years to onset"), fontsize=9)
            ax.set_xticks([-6, -3, 0, 3])
            ax.tick_params(axis="both", labelsize=9)
fig.tight_layout(pad=1.0, h_pad=1.4, w_pad=1.1)
fig.savefig(FIGURES / "fig_event_study.pdf"); fig.savefig(FIGURES / "fig_event_study.svg")
write_table(es_rows, ["outcome", "mean pre (t-6..t-2)", "t-6", "t-3", "t+2", "events"], "tab_h4_events",
            "Descriptive event profiles around first observed coups and autocratization onsets", "tab:events",
            "First event per country in the analysis window; complete -6 to +4 outcome windows without another event. "
            "Country and year FE, never-event countries as controls; clustered SEs. At least five event countries and two controls required. "
            "Pooled descriptive profiles, not the Sun-Abraham estimator, not causal effects; event selection and heterogeneous paths remain concerns.")
(TABLES / "summary_h4.json").write_text(json.dumps(dict(
    lpm=[None if r is None else dict(k=float(r.params.get("k_l1", np.nan)), k_se=float(r.bse.get("k_l1", np.nan)), k_p=float(r.pvalues.get("k_l1", np.nan)),
                                     tax=float(r.params.get("tax_any_l1", np.nan)), tax_se=float(r.bse.get("tax_any_l1", np.nan)), tax_p=float(r.pvalues.get("tax_any_l1", np.nan)),
                                     kxt=float(r.params.get("k_x_tax", np.nan)), kxt_p=float(r.pvalues.get("k_x_tax", np.nan)),
                                     n=int(r.nobs), events=int(r._d[r._y].sum())) for r in h],
    clogit=clog, events=es_rows), indent=1), encoding="utf-8")
print("done: H4")
