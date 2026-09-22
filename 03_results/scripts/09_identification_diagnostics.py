"""Weak-IV, sector-exposure and coup-sample diagnostics using archived data only.

Run: python 03_results/scripts/09_identification_diagnostics.py
Outputs: tab_identification_*.csv/.tex and identification_completion.md.
No causal exclusion restriction or coup actor classification is supplied by these tests.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CONTROLS, TABLES, load_panel, write_table, iv2sls

ROOT = Path(__file__).resolve().parents[2]
SEED = 20260907
DRAWS = "enumerate"


def design(df, controls):
    """Full-rank nuisance design, standardized controls and explicit two-way FE."""
    w = df[controls].astype(float).copy()
    for c in controls:
        if w[c].nunique() > 2:
            w[c] = (w[c] - w[c].mean()) / w[c].std()
    w = pd.concat([w, pd.get_dummies(df.iso3, prefix="country", drop_first=True, dtype=float),
                   pd.get_dummies(df.year, prefix="year", drop_first=True, dtype=float)], axis=1)
    w = sm.add_constant(w).to_numpy()
    if np.linalg.matrix_rank(w) != w.shape[1]:
        raise ValueError("Nuisance design is rank deficient")
    return w


class RestrictedClusterAR:
    """Single-instrument AR = CR1 t-test of z in (y-beta*x) ~ W + z.

    The restricted bootstrap regresses y-beta*x on W, multiplies its residual
    by one Rademacher weight per country, and refits W+z in each draw. Both
    nuisance coefficients and unrestricted CR1 studentization are refitted.
    Linearity permits caching y and x contributions for the entire grid.
    """
    def __init__(self, df, y, x, z, controls, draws=DRAWS):
        self.df = df
        self.y = df[y].to_numpy(float)
        self.x = np.zeros(len(df)) if x is None else ((df[x] - df[x].mean()) / df[x].std()).to_numpy(float)
        self.z = ((df[z] - df[z].mean()) / df[z].std()).to_numpy(float)
        self.w = design(df, controls)
        self.full = np.column_stack([self.w, self.z])
        if np.linalg.matrix_rank(self.full) != self.full.shape[1]:
            raise ValueError("Excluded instrument is absorbed")
        self.qw = np.linalg.qr(self.w, mode="reduced")[0]
        self.qf = np.linalg.qr(self.full, mode="reduced")[0]
        self.zr = self.z - self.qw @ (self.qw.T @ self.z)
        self.influence = self.zr / (self.zr @ self.zr)
        self.uy = self.y - self.qw @ (self.qw.T @ self.y)
        self.ux = self.x - self.qw @ (self.qw.T @ self.x)
        self.groups, self.codes = np.unique(df.iso3, return_inverse=True)
        self.G = len(self.groups)
        self.N, self.K = self.full.shape
        self.cr1 = self.G / (self.G - 1) * (self.N - 1) / (self.N - self.K)
        self.enumerated = draws == "enumerate"
        if self.enumerated:
            # A two-sided studentized statistic is invariant to reversing every sign.
            # Fix the first country's weight to +1 and enumerate all remaining signs.
            self.draws = 2 ** (self.G - 1)
            if self.G > 20:
                raise ValueError("Enumeration intended for at most 20 clusters")
            codes = np.arange(self.draws, dtype=np.uint32)
            self.weights = np.vstack([np.ones(self.draws),
                                      1. - 2. * ((codes[None, :] >> np.arange(self.G - 1)[:, None]) & 1)])
        else:
            self.draws = draws
            self.weights = np.random.default_rng(SEED).choice([-1., 1.], size=(self.G, draws))
        self.by, self.sy = self.bootstrap_basis(self.uy)
        self.bx, self.sx = self.bootstrap_basis(self.ux)
        self.vy = self.cr1 * np.sum(self.sy ** 2, axis=0)
        self.vyx = self.cr1 * np.sum(self.sy * self.sx, axis=0)
        self.vx = self.cr1 * np.sum(self.sx ** 2, axis=0)

    def group_scores(self, residual):
        return np.asarray([self.influence[self.codes == g] @ residual[self.codes == g]
                           for g in range(self.G)])

    def bootstrap_basis(self, restricted_residual):
        # Group sufficient statistics avoid allocating an N-by-number-of-draws matrix.
        a = self.group_scores(restricted_residual)
        projection = np.column_stack([self.qf[self.codes == g].T @ restricted_residual[self.codes == g]
                                      for g in range(self.G)])
        score_projection = self.group_scores(self.qf)
        # This is algebraically q_g' M_full (u * w), so every nuisance coefficient
        # and the instrument coefficient are refitted in every draw.
        return a @ self.weights, (np.diag(a) - score_projection @ projection) @ self.weights

    def test(self, beta):
        adjusted = self.uy - beta * self.ux
        coef = self.influence @ adjusted
        residual = adjusted - self.zr * coef
        se = np.sqrt(self.cr1 * np.sum(self.group_scores(residual) ** 2))
        t_obs = coef / se
        t_boot = (self.by - beta * self.bx) / np.sqrt(np.maximum(self.vy - 2 * beta * self.vyx + beta ** 2 * self.vx, 0))
        # Numerical tolerance includes the observed all-positive sign pattern.
        exceed = np.sum(np.abs(t_boot) >= abs(t_obs) - 1e-10)
        p_wild = exceed / self.draws if self.enumerated else (1 + exceed) / (self.draws + 1)
        return dict(coef=float(coef), se=float(se), t=float(t_obs),
                    p_cluster=float(2 * student_t.sf(abs(t_obs), self.G - 1)), p_wild=float(p_wild))

    def verify(self):
        """Independent dummy-OLS/CR1 checks, including one full bootstrap refit."""
        for beta in [-.7, 0., .9]:
            fit = sm.OLS(self.y - beta * self.x, self.full).fit(
                cov_type="cluster", cov_kwds={"groups": self.df.iso3})
            result = self.test(beta)
            np.testing.assert_allclose([result["coef"], result["se"]],
                                       [fit.params[-1], fit.bse[-1]], rtol=1e-8, atol=1e-9)
        beta = .37
        adjusted = self.y - beta * self.x
        fitted = self.qw @ (self.qw.T @ adjusted)
        draw = 7  # nontrivial cluster sign flips, not just the observed all-positive draw
        boot_y = fitted + (adjusted - fitted) * self.weights[self.codes, draw]
        fit = sm.OLS(boot_y, self.full).fit(cov_type="cluster", cov_kwds={"groups": self.df.iso3})
        coef = self.by[draw] - beta * self.bx[draw]
        se = np.sqrt(self.cr1 * np.sum((self.sy[:, draw] - beta * self.sx[:, draw]) ** 2))
        np.testing.assert_allclose([coef, se], [fit.params[-1], fit.bse[-1]], rtol=1e-8, atol=1e-9)


def accepted_segments(grid, accepted):
    """List contiguous accepted grid points; never fill gaps or extrapolate tails."""
    indices = np.flatnonzero(accepted)
    if not len(indices):
        return [], "none on grid"
    pieces = np.split(indices, np.where(np.diff(indices) > 1)[0] + 1)
    spans = [[float(grid[g[0]]), float(grid[g[-1]])] for g in pieces]
    return spans, " U ".join(f"[{a:.3f}, {b:.3f}]" for a, b in spans)


def iv_diagnostics(p):
    specs = [
        ("commodity_repression", "rep_phys", "k_l1", "z_commodity_d5_l1", CONTROLS, 2., .005),
        ("commodity_choice", "rep_choice", "k_l1", "z_commodity_d5_l1", CONTROLS, 20., .025),
        ("commodity_labour", "labsh_any", "k_l1", "z_commodity_d5_l1", CONTROLS, 1., .0025),
        ("robot_repression", "rep_phys", "ln_robots_pw_l1", "z_robot_l1", CONTROLS, 5., .01),
        ("robot_labour", "labsh_any", "ln_robots_pw_l1", "z_robot_l1", CONTROLS, 2., .005),
        ("robot_repression_trend", "rep_phys", "ln_robots_pw_l1", "z_robot_l1", CONTROLS + ["china_x_manuf"], 5., .01),
    ]
    summaries, grids, members, displays = [], [], [], []
    for name, y, x, z, controls, extent, step in specs:
        df = p.dropna(subset=[y, x, z] + controls).copy()
        ar = RestrictedClusterAR(df, y, x, z, controls)
        ar.verify()
        iv = iv2sls(p, y, x, [z], controls)
        point = float((ar.influence @ ar.uy) / (ar.influence @ ar.ux))
        np.testing.assert_allclose(point, iv.params[x], rtol=1e-8, atol=1e-8)
        grid = np.round(np.arange(-extent, extent + step / 2, step), 8)
        tests = [ar.test(beta) for beta in grid]
        aw = np.array([t["p_wild"] >= .05 for t in tests])
        ac = np.array([t["p_cluster"] >= .05 for t in tests])
        spans, display = accepted_segments(grid, aw)
        cspans, _ = accepted_segments(grid, ac)
        zero = ar.test(0.)
        for beta, test, wild_ok, cluster_ok in zip(grid, tests, aw, ac):
            grids.append(dict(specification=name, beta=beta, **test, accepted_wild_95=wild_ok,
                              accepted_cluster_95=cluster_ok))
        summary = dict(specification=name, outcome=y, endogenous=x, instrument=z,
                       n=ar.N, countries=ar.G, start=int(df.year.min()), end=int(df.year.max()),
                       iv_coefficient=point, first_stage_F=float(iv._fsF),
                       ar_zero_t=zero["t"], ar_zero_p_cluster=zero["p_cluster"],
                       ar_zero_p_wild=zero["p_wild"], grid_lower=-extent, grid_upper=extent,
                       grid_step=step, accepted_wild_segments=json.dumps(spans),
                       accepted_cluster_segments=json.dumps(cspans),
                       touches_lower=bool(aw[0]), touches_upper=bool(aw[-1]),
                       all_grid_accepted=bool(aw.all()), bootstrap_draws=ar.draws,
                       resampling="all Rademacher sign pairs enumerated; first country sign fixed positive",
                       unit="outcome units per estimation-sample SD of endogenous regressor")
        summaries.append(summary)
        displays.append([name.replace("_", " "), f"{point:.3f}", f"{iv._fsF:.2f}",
                         f"{zero['p_wild']:.4f}", display, str(ar.N)])
        for iso, year in df[["iso3", "year"]].itertuples(index=False, name=None):
            members.append(dict(specification=name, iso3=iso, year=year))
        print(f"AR {name}: N={ar.N}, p(0)={zero['p_wild']:.4f}, grid={display}", flush=True)
    pd.DataFrame(summaries).to_csv(TABLES / "tab_identification_ar_summary.csv", index=False)
    pd.DataFrame(grids).to_csv(TABLES / "tab_identification_ar_grid.csv", index=False)
    pd.DataFrame(members).to_csv(TABLES / "tab_identification_iv_samples.csv", index=False)
    write_table(displays, ["2SLS", "First-stage F", "AR wild p(0)", "95% accepted grid points", "N"],
                "tab_identification_ar", "Weak-instrument robust null tests and finite-grid inversion",
                "tab:identification_ar", "Each row reproduces the corresponding existing IV estimation sample. "
                "AR regresses y-beta*x on controls, country/year FE and the excluded instrument. "
                "Country CR1 studentization; all distinct Rademacher sign pairs enumerated (131,072 for 18 countries). "
                "Brackets summarize consecutive accepted grid points, not interpolated confidence bounds. "
                "Sets reaching a grid edge cannot be called bounded confidence intervals. "
                "Units: original outcome per sample SD of the endogenous regressor. Exclusion remains unverified.")
    return pd.DataFrame(summaries)


def sector_diagnostics(p):
    eco = pd.read_csv(ROOT / "02_data/raw/ilo_emp_eco_modelled.csv", low_memory=False)
    codes = ["ECO_ISIC4_TOTAL", "ECO_ISIC4_C", "ECO_ISIC4_P", "ECO_ISIC4_Q"]
    eco = eco[(eco.SEX == "SEX_T") & (eco.FREQ == "A") & eco.ECO.isin(codes)].copy()
    eco = eco[eco.TIME_PERIOD.astype(str).str.fullmatch(r"\d{4}")]
    eco["year"] = eco.TIME_PERIOD.astype(int)
    if eco.duplicated(["REF_AREA", "year", "ECO"]).any():
        raise ValueError("Ambiguous sector series; no silent source selection")
    wide = eco.pivot(index=["REF_AREA", "year"], columns="ECO", values="OBS_VALUE")
    shares = pd.DataFrame(index=wide.index)
    for short, code in [("education", "P"), ("health_social", "Q"), ("manufacturing", "C")]:
        shares[short + "_share"] = 100 * wide[f"ECO_ISIC4_{code}"] / wide.ECO_ISIC4_TOTAL
    shares["education_health_share"] = shares.education_share + shares.health_social_share
    shares = shares.reset_index().rename(columns={"REF_AREA": "iso3"})
    outcomes = ["education_share", "health_social_share", "education_health_share", "manufacturing_share"]
    merged = p.merge(shares, on=["iso3", "year"], how="left", validate="one_to_one")
    # All four outcomes and the two trend variants use exactly the same observations.
    common = merged[merged.year >= 1998].dropna(subset=outcomes + ["z_robot_l1", "china_x_manuf"] + CONTROLS).copy()
    if not ((common[outcomes] >= 0) & (common[outcomes] <= 100)).all().all():
        raise ValueError("Invalid employment share")
    common[["iso3", "year"] + outcomes].to_csv(TABLES / "tab_identification_sector_data.csv", index=False)
    rows, displays = [], []
    for y in outcomes:
        for trend in [False, True]:
            controls = CONTROLS + (["china_x_manuf"] if trend else [])
            model = RestrictedClusterAR(common, y, None, "z_robot_l1", controls)
            result = model.test(0.)
            rows.append(dict(outcome=y, manufacturing_trend=trend, **result, n=model.N, countries=model.G,
                             start=int(common.year.min()), end=int(common.year.max()),
                             instrument="z_robot_l1", unit="employment percentage points per instrument SD",
                             status="sector-exposure diagnostic; not a validated negative-control outcome"))
            displays.append([y.replace("_", " ") + (" + share x log trend" if trend else ""),
                             f"{result['coef']:.3f} ({result['se']:.3f})", f"{result['p_wild']:.4f}", str(model.N)])
    results = pd.DataFrame(rows)
    results.to_csv(TABLES / "tab_identification_sector_summary.csv", index=False)
    write_table(displays, ["Coefficient (CR1 SE)", "Wild p", "N"], "tab_identification_sectors",
                "Employment-sector exposure diagnostics for the robot instrument", "tab:identification_sectors",
                "ILO modelled total-sex employment, ISIC Rev.4 P (education), Q (human health/social work), C (manufacturing). "
                "Sector employment / ISIC4 total, in percent. Common complete-case sample starts in 1998, "
                "after the 1993-97 exposure-share window. Original controls plus country/year FE; "
                "Restricted country wild bootstrap with all distinct Rademacher sign pairs enumerated. Education and health may respond to technology, "
                "income, fiscal policy and labour reallocation, so these are not valid zero-effect placebos by assumption.")
    return results


def coup_roster():
    p = load_panel(None)
    p["tax_source"] = np.select([p.tax_gdp.notna(), p.tax_ab_gdp.notna(), p.tax_rev_wdi.notna()],
                                ["CEPALSTAT", "Andersson-Brambor", "WDI"], default="unavailable")
    for variable in ["tax_any", "tax_source", "k_extended"]:
        p[variable + "_l1"] = p.groupby("iso3")[variable].shift(1)
    p = p[p.year.between(1960, 2024)].copy()
    ctrl = ["ln_gdppc", "urban_share", "trade_open"]
    p["controls_available"] = p[ctrl].notna().all(axis=1)
    p["k_available_l1"] = p.k_l1.notna()
    p["tax_available_l1"] = p.tax_any_l1.notna()
    p["h4_column1_eligible"] = p.multiparty_l1.eq(1) & p.controls_available & p.k_available_l1 & p.coup.notna()
    p["h4_column2_eligible"] = p.h4_column1_eligible & p.tax_available_l1
    p["h4_column3_eligible"] = p.h4_column2_eligible & p[["milexp_gdp_l1", "mil_support_l1"]].notna().all(axis=1)
    regime_names = {0: "closed autocracy", 1: "electoral autocracy", 2: "electoral democracy", 3: "liberal democracy"}
    p["prior_regime"] = p.v2x_regime_l1.map(regime_names).fillna("unavailable")
    p["actor_classification"] = "unclassified: no event-actor attribution in archived panel"
    columns = ["iso3", "country", "year", "e_pt_coup", "coup", "e_pt_coup_attempts", "v2x_regime_l1",
               "prior_regime", "multiparty_l1", "dem_l1", "v2x_polyarchy_l1", "k_l1", "k_available_l1",
               "k_extended_l1", "tax_any_l1", "tax_available_l1", "tax_source_l1", "controls_available",
               "h4_column1_eligible", "h4_column2_eligible", "h4_column3_eligible", "actor_classification"]
    roster = p.loc[p.coup.eq(1), columns].sort_values(["year", "iso3"])
    roster.to_csv(TABLES / "tab_identification_coup_roster.csv", index=False)
    sample_rows = []
    for label, mask in [("All 1960-2024", p.coup.notna()), ("Prior multiparty regime", p.multiparty_l1.eq(1)),
                        ("Prior RoW democracy", p.dem_l1.eq(1)),
                        ("H4 column 1", p.h4_column1_eligible), ("H4 column 2", p.h4_column2_eligible),
                        ("H4 column 3", p.h4_column3_eligible)]:
        frame = p.loc[mask]
        sample_rows.append(dict(sample=label, observations=len(frame), countries=frame.iso3.nunique(),
                                coup_country_years=int(frame.coup.sum()), reported_coup_count=int(frame.e_pt_coup.sum()),
                                countries_with_coups=frame.loc[frame.coup.eq(1), "iso3"].nunique()))
    summary = pd.DataFrame(sample_rows)
    summary.to_csv(TABLES / "tab_identification_coup_summary.csv", index=False)
    display = [[r.sample, str(r.observations), str(r.countries), str(r.coup_country_years),
                str(r.reported_coup_count), str(r.countries_with_coups)] for r in summary.itertuples()]
    write_table(display, ["N", "Countries", "Coup country-years", "Source coup count", "Event countries"],
                "tab_identification_coup_samples", "Coup-event coverage and complete-case selection", "tab:identification_coups",
                "Exact archived Powell-Thyne series via V-Dem; annual indicators merge multiple events in one country-year. "
                "Multiparty includes electoral autocracies (RoW>=1); democracy is RoW>=2. "
                "Prior-year regime, capital and tax availability appear in the companion roster CSV. "
                "These data do not classify capital-led coups or identify individual actors.")
    return roster, summary


def report(ar, sectors, roster, coup_summary):
    lines = ["# Identification diagnostics completed on archived inputs", "", "Date: 7 September 2026.", "",
             "## What is completed", "",
             "The six IV specifications now have weak-instrument robust Anderson-Rubin null tests and reproducible finite-grid inversions. "
             "Employment-sector diagnostics use archived ILO observations. An exact annual coup roster documents regime and missing-data selection. "
             "These additions complete feasible diagnostics; they do not supply an exclusion restriction, event-actor coding or causal identification.", "",
             "## Anderson-Rubin procedure and results", "",
             "For each candidate beta, regress y - beta*x on the original exogenous controls, country/year dummies and the single excluded instrument. "
             "x is standardized within that IV estimation sample; y keeps its original units. The AR statistic is the instrument's CR1 t-statistic "
             "(or its square), so no first-stage coefficient divides the null test. Cluster reference p-values use t(G-1). "
             "The reported bootstrap imposes the zero-instrument-coefficient null: residuals from y-beta*x on nuisance controls are multiplied "
             "by one Rademacher weight per country, all coefficients are refitted, and each draw is studentized with its own full-model CR1 variance. "
             "All 2^(G-1) distinct Rademacher sign pairs are enumerated (131,072 for 18 countries), fixing the first sign positive because the two-sided statistic "
             "is invariant to a global sign reversal. The p-value is the fraction at least as extreme as observed; the all-positive observed pattern is included. "
             "This eliminates Monte Carlo noise; it does not make the inferential approximation an exact finite-sample test. Country dependence is allowed within clusters, "
             "but residual dependence across countries, 18-cluster limitations and invalid instruments remain concerns.", "",
             "Implementation verification compares the projected coefficient and CR1 SE with statsmodels full-dummy OLS at three null values "
             "and one explicit restricted bootstrap refit for every IV specification. The reduced-form/first-stage ratio also matches the existing 2SLS coefficient.", "",
             "| Specification | N | First-stage F | AR wild p(beta=0) | Accepted grid points at 95% | Edge reached? |",
             "|---|---:|---:|---:|---|---|"]
    for row in ar.itertuples():
        spans = json.loads(row.accepted_wild_segments)
        text = " U ".join(f"[{a:.3f}, {b:.3f}]" for a, b in spans) or "none"
        lines.append(f"| {row.specification} | {row.n} | {row.first_stage_F:.2f} | {row.ar_zero_p_wild:.4f} | {text} | {row.touches_lower or row.touches_upper} |")
    lines += ["", "These are consecutive accepted *grid points*, not interpolated full confidence intervals. "
              "All grid coordinates, p-values, steps and boundaries are archived. An edge hit means the extent beyond that boundary "
              "has not been determined; accepting all inspected points does not prove the entire real line is accepted. "
              "Commodity instrument timing was explored earlier, so these tests remain exploratory and do not correct for that search. "
              "Rejecting beta=0 is conditional on valid exclusion/exogeneity; AR cannot test or repair that assumption.", "",
              "Method sources: [Stata's IV postestimation manual](https://www.stata.com/manuals/rivregresspostestimation.pdf), "
              "[restricted wild-cluster bootstrap manual](https://www.stata.com/manuals/rwildbootstrap.pdf), "
              "and [Davidson and MacKinnon on AR-set inversion](https://www.econ.queensu.ca/research/working-papers/1257). "
              "The implementation is the restricted auxiliary-regression bootstrap described above, not an assertion that all other IV bootstrap procedures are equivalent.", "",
              "## Sector diagnostic and placebo feasibility", "",
              "The archive contains ILO modelled employment for ISIC Rev.4 P (education), Q (human health and social work), and C (manufacturing), "
              "with consistent ISIC4 total-employment denominators; section labels follow [UN ISIC Rev.4](https://unstats.un.org/unsd/demographic-social/census/documents/isic_rev4.pdf). "
              "Total-sex annual observations are used; duplicate source keys fail rather than being silently selected. "
              "Education and health are a defensible comparison for direct industrial-robot exposure, but technological change, income, public budgets and labour reallocation "
              "can affect their employment. Thus the requested strictly non-automatable-sector placebo is not identified by the archive. "
              "The completed analysis is expressly an exposure-pattern diagnostic, including manufacturing as a comparison. "
              "The 1998+ window follows the 1993-97 share construction; all outcomes and trend variants use one common sample.", "",
              "| Employment outcome | Share x log trend | Coefficient (CR1 SE) | Wild p | N |", "|---|---|---:|---:|---:|"]
    for row in sectors.itertuples():
        lines.append(f"| {row.outcome} | {row.manufacturing_trend} | {row.coef:.3f} ({row.se:.3f}) | {row.p_wild:.4f} | {row.n} |")
    lines += ["", "Coefficients are employment percentage points per estimation-sample SD of the lagged robot instrument. "
              "The manufacturing-share-by-log-time control addresses one class of differential trends; it does not measure China exposure. "
              "Results are exploratory, not adjusted as a new confirmatory testing family. Neither significance nor insignificance validates exclusion.", "",
              "## Exact annual coup roster and remaining event limitation", "",
              f"The archive has {len(roster)} successful-coup country-years and {int(roster.e_pt_coup.sum())} reported coups in 1960-2024. "
              "Annual observations containing multiple coups are preserved through the original count. They cannot be expanded into dated actor-specific events from this file. "
              "The roster includes every successful-coup country-year, prior RoW regime, democracy and multiparty flags, lagged capital/tax values, tax source and availability, "
              "and complete-case eligibility for H4 columns 1-3. Unavailable prior-year data are left missing. Every actor classification is explicitly unclassified.", "",
              "| Sample | N | Countries | Coup country-years | Source coup count | Event countries |", "|---|---:|---:|---:|---:|---:|"]
    for row in coup_summary.itertuples():
        lines.append(f"| {row.sample} | {row.observations} | {row.countries} | {row.coup_country_years} | {row.reported_coup_count} | {row.countries_with_coups} |")
    lines += ["", "Multiparty (RoW>=1) includes electoral autocracies, and should not be relabelled elected democratic government. "
              "Tax revenue measures realized revenue, not a feasible ceiling. No prior-regime flag or military-support code establishes that capital owners initiated a coup. "
              "The event-country/window restrictions used in the revised event-profile analysis remain in force; this roster is not a reason to restore withheld estimates.", "",
              "## Manuscript-ready addition", ""]
    cp = ar.set_index("specification")
    sect = sectors[(sectors.outcome == "education_health_share") & ~sectors.manufacturing_trend].iloc[0]
    sect_trend = sectors[(sectors.outcome == "education_health_share") & sectors.manufacturing_trend].iloc[0]
    c1 = coup_summary[coup_summary["sample"] == "H4 column 1"].iloc[0]
    c2 = coup_summary[coup_summary["sample"] == "H4 column 2"].iloc[0]
    lines += [f"Weak-instrument-robust Anderson-Rubin tests, enumerating the restricted country wild-bootstrap Rademacher distribution, "
              f"give zero-effect p-values of {cp.loc['commodity_repression', 'ar_zero_p_wild']:.3f}, "
              f"{cp.loc['commodity_choice', 'ar_zero_p_wild']:.3f} and {cp.loc['commodity_labour', 'ar_zero_p_wild']:.3f} "
              "for the commodity-instrument repression, policy-choice and labour-share equations, respectively. "
              f"The corresponding robot-instrument p-values are {cp.loc['robot_repression', 'ar_zero_p_wild']:.3f} "
              f"and {cp.loc['robot_labour', 'ar_zero_p_wild']:.3f}; with the manufacturing-share trend, the repression p-value is "
              f"{cp.loc['robot_repression_trend', 'ar_zero_p_wild']:.3f}. Finite-grid inversions document weak precision and are not presented as full bounded confidence intervals. "
              "These tests protect against weak relevance under the maintained model but do not establish instrument exclusion. "
              f"In an additional common-sample sector diagnostic (N={int(sect['n'])}), the robot instrument's coefficient for education-plus-health employment "
              f"is {sect['coef']:.3f} percentage points per instrument SD (wild p={sect['p_wild']:.3f}), "
              f"and {sect_trend['coef']:.3f} with a manufacturing-share trend (p={sect_trend['p_wild']:.3f}). "
              "These sectors can respond to fiscal policy and economic reallocation, so this is not a zero-effect placebo. "
              f"The annual coup roster contains {len(roster)} coup country-years; the H4 multiparty sample retains "
              f"{int(c1['coup_country_years'])} in the capital-only specification and {int(c2['coup_country_years'])} after requiring tax data. "
              "Because the source lacks actor attribution, these events cannot be interpreted as a validated sample of capital-led coups.", ""]
    bounds = {key: json.loads(cp.loc[key, "accepted_wild_segments"])[0] for key in cp.index}
    lines += [f"At the nominal 95% level, the accepted commodity-instrument grid points extend from "
              f"{bounds['commodity_repression'][0]:.3f} to {bounds['commodity_repression'][1]:.3f} for repression "
              f"(grid step 0.005), {bounds['commodity_choice'][0]:.3f} to {bounds['commodity_choice'][1]:.3f} "
              f"for policy choice (step 0.025), and {bounds['commodity_labour'][0]:.4f} to {bounds['commodity_labour'][1]:.4f} "
              "for labour share (step 0.0025). The policy-choice set reaches its lower search boundary; its continuation below -20 is undetermined. "
              "All inspected robot-effect grid points are accepted: [-5,5] for repression with and without the manufacturing-share trend, "
              "and [-2,2] for labour share. These are finite search ranges in outcome units per endogenous-regressor SD; "
              "they do not establish the endpoints or topology of the full confidence sets.", ""]
    (ROOT / "05_red_team/identification_completion.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    p = load_panel((1990, 2024))
    p["rep_choice"] = (p.rep_phys - p.rep_phys.mean()) / p.rep_phys.std() - (p.abs_red - p.abs_red.mean()) / p.abs_red.std()
    p["china_x_manuf"] = p.w_manuf_use * np.log(p.year - 1989)
    ar = iv_diagnostics(p)
    sectors = sector_diagnostics(p)
    roster, coup_summary = coup_roster()
    report(ar, sectors, roster, coup_summary)
    print("done: identification diagnostics, numerical verification and annual event roster", flush=True)


if __name__ == "__main__":
    main()
