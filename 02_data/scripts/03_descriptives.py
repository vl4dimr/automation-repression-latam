"""Descriptive statistics and the six descriptive figures.

Outputs:
  03_results/figures/fig1..fig6 (.pdf and .svg)
  03_results/tables/tab_descriptives.tex / .csv
  02_descriptives.pdf (all figures + summary table, one document)
Run: python 02_data/scripts/03_descriptives.py
"""
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLEAN, FIGURES, ISO3, NAMES, ROOT, TABLES
from figlabels import L

plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 120, "savefig.bbox": "tight", "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "legend.fontsize": 7.5, "lines.linewidth": 1.2,
                     # TrueType incrustada: el preflight de la editorial marca las Type3
                     "pdf.fonttype": 42, "ps.fonttype": 42})
C = {"a": "#1f4e79", "b": "#c0504d", "c": "#7f7f7f", "d": "#e6a532", "e": "#4f9d69"}

p = pd.read_parquet(CLEAN / "panel.parquet")
p = p[p.year <= 2024].copy()
sys.path.insert(0, str(ROOT / "03_results/scripts"))
from common import fe_residualize
pw = p[(p.year >= 1990) & (p.year <= 2024)].copy()
from figlabels import LANG as _FIGLANG
pdf = PdfPages(ROOT / ("02_descriptives_es.pdf" if _FIGLANG == "es" else "02_descriptives.pdf"))


def save(fig, name, title=None, y=1.01):
    """Guarda la figura para la revista y, aparte, para el PDF de trabajo.

    World Development pide que el titulo de una figura no aparezca sobre la figura misma: va en el
    pie numerado del manuscrito. Los archivos que se envian se graban limpios. El titulo se anade
    despues, solo a la copia que se acumula en 02_descriptives.pdf, que es un documento de trabajo.
    """
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.svg")
    if title is not None:
        fig.suptitle(title, y=y)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig 1: labour share vs capital per worker
fig, axes = plt.subplots(3, 6, figsize=(13, 6.5), sharey=True)
for ax, iso in zip(axes.ravel(), ISO3):
    g = p[(p.iso3 == iso) & (p.year >= 1960)]
    ax.plot(g.year, g.labsh_any, color=C["a"], label=L("labour share (left)"))
    ax2 = ax.twinx(); ax2.plot(g.year, g.k, color=C["b"], lw=0.9, label=L("log K/L (right)"))
    ax2.tick_params(labelsize=6); ax2.spines["top"].set_visible(False)
    ax.set_title(L(NAMES[iso])); ax.set_ylim(0.2, 0.8); ax.tick_params(labelsize=6)
    ax.axvspan(1990, 2024, color="#eeeeee", zorder=-1)
h1, l1 = axes[0, 0].get_legend_handles_labels()
fig.legend(h1 + [plt.Line2D([], [], color=C["b"])], l1 + [L("log K/L (right axis)")], loc="lower center", ncol=2, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig1_labsh_k",
     L("Labour share and capital per worker, 1960–2024 (shaded: baseline window). "
       "Sources: PWT 10.01, ILO, WDI."))

# ---------------------------------------------------------------- Fig 2: robot stock per worker
fig, ax = plt.subplots(figsize=(8, 4.5))
if "robots_pw" in pw:
    last = pw[pw.year <= 2024].groupby("iso3").robots_pw.last().sort_values(ascending=False)
    for i, iso in enumerate(last.index):
        g = pw[pw.iso3 == iso]
        ax.plot(g.year, g.robots_pw, color=plt.cm.viridis(i / 18), label=L(NAMES[iso]) if i < 8 else None)
    ax.set_ylabel(L("Robot-import-value stock per worker (2015 US$)"))
    ax.legend(frameon=False, ncol=2)
    ax.set_yscale("symlog", linthresh=1)
else:
    ax.text(0.5, 0.5, L("Comtrade pull not complete"), ha="center")
save(fig, "fig2_robots", L("Robot import-value proxy, 1996–2024\nSource: UN Comtrade; PIM with 10% depreciation"))

# ---------------------------------------------------------------- Fig 3: repression by country, with regime
fig, axes = plt.subplots(3, 6, figsize=(13, 6.5), sharey=True, sharex=True)
for ax, iso in zip(axes.ravel(), ISO3):
    g = p[(p.iso3 == iso) & (p.year >= 1960)]
    ax.fill_between(g.year, 0, 1, where=(g.dem == 0), color="#f6d5d5", step="mid", label=L("non-democracy (RoW)"))
    ax.plot(g.year, g.rep_phys, color=C["a"], label=L("physical repression (1 - v2x_clphy)"))
    ax.plot(g.year, (g.pts_mean - 1) / 4, color=C["d"], lw=0.9, label=L("PTS (rescaled 0-1)"))
    cp = g[g.coup == 1]
    ax.scatter(cp.year, [0.95] * len(cp), marker="v", color="k", s=12, label=L("successful coup"))
    ax.set_title(L(NAMES[iso])); ax.set_ylim(0, 1); ax.tick_params(labelsize=6)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="lower center", ncol=4, frameon=False)
fig.tight_layout(rect=(0, 0.04, 1, 1))
save(fig, "fig3_repression",
     L("Repression, regime and coups, 1960–2024. Sources: V-Dem v16, PTS-2025, Powell & Thyne."))

# ---------------------------------------------------------------- Fig 4: repression vs redistribution, conditional on k
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
s = pw.dropna(subset=["rep_phys", "abs_red", "k"]).copy()
ax = axes[0]
sc = ax.scatter(s.rep_phys, s.abs_red, c=s.k, cmap="viridis", s=12, alpha=0.8)
ax.set_xlabel(L("Physical repression (1 - v2x_clphy)")); ax.set_ylabel(L("Absolute redistribution (Gini market - Gini disposable)"))
ax.set_title(L("(a) Country-years 1990–2024, colour = log K/L"))
plt.colorbar(sc, ax=ax, label=L("log K/L"))
# within-country residuals
ax = axes[1]
for v in ["rep_phys", "abs_red"]:
    s[v + "_w"] = fe_residualize(s[v], s)
ax.scatter(s.rep_phys_w, s.abs_red_w, s=12, alpha=0.7, color=C["c"])
b = np.polyfit(s.rep_phys_w, s.abs_red_w, 1)
xx = np.linspace(s.rep_phys_w.min(), s.rep_phys_w.max(), 20)
ax.plot(xx, np.polyval(b, xx), color=C["b"], label=L("slope") + f" = {b[0]:.2f}")
ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(L("Repression, within-country and within-year deviation")); ax.set_ylabel(L("Redistribution, same deviation"))
ax.set_title(L("(b) Exact country and year fixed-effect residuals")); ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig4_rep_vs_redist", L("Repression and redistribution: descriptive associations\nSources: V-Dem and SWIID 9.92"), y=1.02)

# ---------------------------------------------------------------- Fig 5: regime transition matrix conditional on capital growth
q = p[(p.year >= 1960) & p.rep_phys.notna() & p.k.notna()].copy()
q = q.sort_values(["iso3", "year"])
q["state"] = pd.qcut(q.rep_phys, 3, labels=["low", "mid", "high"])
q["state_next"] = q.groupby("iso3").state.shift(-1)
q["dk"] = q.groupby("iso3").k.shift(-1) - q.k
q = q[q.groupby("iso3").year.shift(-1).eq(q.year + 1)].copy()
q = q.dropna(subset=["state_next", "dk"])
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, (lab, sub) in zip(axes, [(L("capital per worker rising (dk > 0)"), q[q.dk > 0]), (L("capital per worker falling (dk <= 0)"), q[q.dk <= 0])]):
    m = pd.crosstab(sub.state, sub.state_next, normalize="index").reindex(index=["low", "mid", "high"], columns=["low", "mid", "high"])
    im = ax.imshow(m.values, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m.values[i, j]:.2f}", ha="center", va="center", color="k" if m.values[i, j] < 0.6 else "w")
    ax.set_xticks(range(3)); ax.set_xticklabels([L(str(c)) for c in m.columns]); ax.set_yticks(range(3)); ax.set_yticklabels([L(str(i)) for i in m.index])
    ax.set_xlabel(L("repression tercile, t+1")); ax.set_ylabel(L("repression tercile, t"))
    ax.set_title(f"{lab}\nN = {len(sub)}")
fig.tight_layout()
save(fig, "fig5_transitions",
     L("Annual transitions between repression terciles, 1960–2024"), y=1.03)
exits = q[(q.state == "high") & (q.state_next != "high") & (q.dk > 0)]
exits_tab = exits.groupby("iso3").size()

# ---------------------------------------------------------------- Fig 6: coups over capital and fiscal capacity
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
# condition on democracy in t-1: V-Dem codes the regime at year end, so the coup year itself is already autocratic
p["dem_l1"] = p.groupby("iso3").dem.shift(1)
h4 = p[(p.year >= 1960) & (p.dem_l1 == 1)].copy()
ax = axes[0]
h4["k_bin"] = pd.qcut(h4.k, 6, duplicates="drop")
rate = h4.groupby("k_bin", observed=True).agg(coup=("coup", "mean"), att=("coup_att", "mean"), n=("coup", "size"), kmid=("k", "median"))
ax.plot(rate.kmid, rate.att * 100, "o-", color=C["a"], label=L("coup attempt"))
ax.plot(rate.kmid, rate.coup * 100, "s-", color=C["b"], label=L("successful coup"))
ax.set_xlabel(L("log capital per worker (sextile medians), democracies only")); ax.set_ylabel(L("% of democracy-years with a coup"))
ax.set_title(L("(a) Coup risk against capital per worker")); ax.legend(frameon=False)
ax = axes[1]
tt = h4.dropna(subset=["tax_gdp"]).copy()
tt["tax_bin"] = pd.qcut(tt.tax_gdp, 5, duplicates="drop")
r2 = tt.groupby("tax_bin", observed=True).agg(coup=("coup", "mean"), att=("coup_att", "mean"), n=("coup", "size"), tmid=("tax_gdp", "median"))
ax.plot(r2.tmid, r2.att * 100, "o-", color=C["a"], label=L("coup attempt"))
ax.plot(r2.tmid, r2.coup * 100, "s-", color=C["b"], label=L("successful coup"))
ax.set_xlabel(L("Tax revenue, % of GDP (quintile medians), democracies only")); ax.set_ylabel(L("% of democracy-years with a coup"))
ax.set_title(L("(b) Coup risk against fiscal capacity")); ax.legend(frameon=False)
fig.tight_layout()
save(fig, "fig6_coups",
     L("Coups in democracies, 1960–2024. Sources: Powell & Thyne via V-Dem, PWT, CEPALSTAT."), y=1.02)

# ---------------------------------------------------------------- descriptive table
vars_ = [("rep_phys", "Physical repression (1 - v2x_clphy)"), ("rep_cso", "CSO repression (-v2csreprss)"),
         ("rep_expr", "Expression repression"), ("pts_mean", "Political Terror Scale (mean)"),
         ("mob", "Mass mobilization (V-Dem)"), ("mmp_protests", "Protest events (MMP)"),
         ("labsh_any", "Labour share"), ("k", "log capital per worker"), ("kl_level", "Capital per worker (2017 US$)"),
         ("robots_pw", "Robot import-value stock per worker (2015 US$)"), ("routine_share", "Routine occupation share"),
         ("abs_red", "Absolute redistribution (Gini pts)"), ("socexp_gdp", "Social expenditure, % GDP"),
         ("tax_gdp", "Tax revenue, % GDP"), ("dem", "Democracy (RoW >= 2)"), ("coup", "Successful coup"),
         ("coup_att", "Coup attempt"), ("ln_gdppc", "log GDP per capita"), ("resource_rents", "Resource rents, % GDP"),
         ("milexp_gdp", "Military expenditure, % GDP"), ("z_commodity", "Commodity price instrument"),
         ("z_robot", "Robot shift-share instrument")]
rows = []
for v, lab in vars_:
    if v not in pw:
        continue
    x = pw[v]
    rows.append(dict(Variable=lab, N=int(x.notna().sum()), Countries=int(pw.loc[x.notna(), "iso3"].nunique()),
                     Mean=x.mean(), SD=x.std(), Min=x.min(), Max=x.max(),
                     Within_SD=(x - pw.groupby("iso3")[v].transform("mean")).std()))
tab = pd.DataFrame(rows)
tab.to_csv(TABLES / "tab_descriptives.csv", index=False)
with open(TABLES / "tab_descriptives.tex", "w", encoding="utf-8") as f:
    f.write("\\begin{table}[htbp]\\centering\n\\caption{Descriptive statistics, 18 Latin American countries, 1990--2024}\n"
            "\\label{tab:desc}\n\\small\n\\begin{tabular}{lrrrrrrr}\n\\toprule\n"
            "Variable & N & Countries & Mean & SD & Within SD & Min & Max \\\\\n\\midrule\n")
    for r in rows:
        f.write(f"{r['Variable']} & {r['N']} & {r['Countries']} & {r['Mean']:.3f} & {r['SD']:.3f} & {r['Within_SD']:.3f} & {r['Min']:.3f} & {r['Max']:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n\\begin{tablenotes}\\footnotesize Within SD is the standard deviation after removing country means. "
            "Sources: V-Dem v16, PTS-2025, MMP v5, PWT 10.01, ILOSTAT, UN Comtrade, SWIID 9.92, CEPALSTAT, WDI, World Bank Pink Sheet.\\end{tablenotes}\n\\end{table}\n")

# table page in the PDF
fig, ax = plt.subplots(figsize=(11, 0.5 * len(rows) + 1.2))
ax.axis("off")
cell = [[textwrap.fill(r["Variable"], 32), r["N"], r["Countries"], f"{r['Mean']:.3f}", f"{r['SD']:.3f}", f"{r['Within_SD']:.3f}", f"{r['Min']:.2f}", f"{r['Max']:.2f}"] for r in rows]
t = ax.table(cellText=cell, colLabels=["Variable", "N", "Countries", "Mean", "SD", "Within SD", "Min", "Max"], loc="center", cellLoc="right", colWidths=[.32,.05,.07,.112,.112,.112,.112,.112])
t.auto_set_font_size(False); t.set_fontsize(7.5); t.scale(1, 2.0)
for (i, j), c in t.get_celld().items():
    if j == 0:
        c.set_text_props(ha="left")
ax.set_title(L("Table 1. Descriptive statistics, 1990–2024"), loc="left")
pdf.savefig(fig); plt.close(fig)

# notes page
fig, ax = plt.subplots(figsize=(11, 5)); ax.axis("off")
txt = ("Notes to the descriptive figures\n\n"
       f"Panel: {pw.iso3.nunique()} countries, {pw.year.min()}–{pw.year.max()}, {len(pw)} country-years.\n"
       f"Exits from the top repression tercile while capital per worker was rising (Figure 5), by country: "
       f"{', '.join(f'{NAMES[i]} ({n})' for i, n in exits_tab.items()) if len(exits_tab) else 'none'}.\n"
       f"Two-way demeaned slope of redistribution on repression (Figure 4b): {b[0]:.3f}.\n"
       "El Salvador uses the ILO labour income share (no PWT labsh). Capital per worker for 2020+ is a perpetual-inventory extension "
       "of PWT with WDI real investment growth; robot stocks are perpetual-inventory sums of Comtrade HS 847950 import values deflated by the US CPI.")
ax.text(0.01, 0.98, txt, va="top", wrap=True, fontsize=9)
pdf.savefig(fig); plt.close(fig)
pdf.close()
print("saved 02_descriptives.pdf and figures; exits with rising k:", exits_tab.to_dict())
print("Exact FE descriptive slope:", b[0])
