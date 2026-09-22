"""Archived-data measurement and common-sample sensitivity for H1/H2.

Run: python 03_results/scripts/08_measurement_robustness.py
No raw or clean data are overwritten. Outputs use tab_measurement_*.
SWIID uses its 100 released imputations, not simulated summary errors.
V-Dem endpoint substitutions are deterministic sensitivity analyses.
"""
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t as student_t

from common import CONTROLS, NAMES, ROOT, TABLES, twfe, write_table

RAW = ROOT / "02_data" / "raw"
DETAILS = []


def record(res, variable, block, specification):
    if variable not in res.params:
        return None
    b = float(res.params[variable])
    se = float(res.bse[variable])
    groups = res._d.iso3.nunique()
    # Explicit few-cluster t reference, rather than the default normal p-value.
    p = float(2 * student_t.sf(abs(b / se), groups - 1))
    item = dict(block=block, specification=specification, variable=variable,
                b=b, se=se, p=p, n=int(res.nobs), countries=int(groups),
                first_year=int(res._d.year.min()), last_year=int(res._d.year.max()))
    DETAILS.append(item)
    return item


def cell(item):
    return f"{item['b']:.4f} ({item['se']:.4f})"


def fit(df, y, xs):
    res = twfe(df, y, xs)
    assert res is not None
    return res


def common_samples(p):
    specifications = [
        ("PWT/ILO: capital", "labsh_any", ["k_l1"] + CONTROLS),
        ("PWT/ILO: capital + robots", "labsh_any", ["k_l1", "ln_robots_pw_l1"] + CONTROLS),
        ("PWT/ILO: capital + routine", "labsh_any", ["k_l1", "routine_share_l1"] + CONTROLS),
        ("ILO: capital + robots", "ilo_labsh", ["k_l1", "ln_robots_pw_l1"] + CONTROLS),
    ]
    needed = sorted({v for _, y, xs in specifications for v in [y] + xs})
    common = p.dropna(subset=needed)
    rows = []
    for name, y, xs in specifications:
        native = record(fit(p, y, xs), "k_l1", "h1_common", name + ": available")
        same = record(fit(common, y, xs), "k_l1", "h1_common", name + ": common")
        rows.append([name, cell(native), str(native['n']), cell(same), str(same['n'])])
    write_table(rows, ["Available b (SE)", "N", "Common b (SE)", "N common"],
                "tab_measurement_common_h1", "H1a. Capital coefficients with a fixed common sample",
                "tab:measurement_h1", "Outcome is labour share on the 0-1 scale; multiply coefficients by 100 for percentage points. "
                "Coefficients are per one within-estimation-sample SD of lagged log capital per worker; "
                "the common columns use identical country-years and regressor scales. Country and year fixed effects, "
                "four baseline controls and country-clustered CR1 standard errors. Changes in the available-sample "
                "columns can reflect both sample composition and specification. Common sample includes "
                f"{common.iso3.nunique()} countries, {common.year.min()}-{common.year.max()}.")

    threat = ["k_l1", "threat_mob_l1", "k_x_threat", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS
    antisys = ["k_l1", "antisys_l5", "k_x_antisys", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS
    specifications = [("Physical: mobilization threat", "rep_phys", threat),
                      ("Physical: anti-system threat", "rep_phys", antisys),
                      ("CSO: anti-system threat", "rep_cso", antisys),
                      ("PTS: anti-system threat", "pts_mean", antisys),
                      ("Composite: anti-system threat", "rep_index", antisys)]
    needed = sorted({v for _, y, xs in specifications for v in [y] + xs})
    common = p.dropna(subset=needed)
    rows = []
    for name, y, xs in specifications:
        native = record(fit(p, y, xs), "k_l1", "h2_common", name + ": available")
        same = record(fit(common, y, xs), "k_l1", "h2_common", name + ": common")
        rows.append([name, cell(native), str(native['n']), cell(same), str(same['n'])])
    write_table(rows, ["Available b (SE)", "N", "Common b (SE)", "N common"],
                "tab_measurement_common_h2", "H2. Capital coefficients with a fixed common sample",
                "tab:measurement_h2", "Raw interaction products and continuous regressors are standardized separately, "
                "as in the main tables. These are capital main coefficients at threat=0, not average marginal associations. "
                "Outcome units differ: physical repression 0-1, CSO reversed V-Dem scale, PTS 1-5, composite index units. "
                "Country/year fixed effects; four controls; country-clustered CR1 SE. Regressors absorbed by fixed effects "
                "are omitted. Restriction to the common sample fixes composition across columns, but cannot recover countries "
                f"excluded by the mobilization threat definition. Common sample: {len(common)} observations, "
                f"{common.iso3.nunique()} countries, {common.year.min()}-{common.year.max()}.")


def robot_sensitivity(p):
    ct = pd.read_csv(RAW / "comtrade_847950_imports.csv")
    ct = ct[ct.iso3.isin(NAMES)].copy()
    assert not ct.duplicated(["iso3", "year"]).any()
    wdi = pd.read_csv(RAW / "wdi.csv")
    us = wdi[(wdi.iso3 == "USA") & (wdi.variable == "inflation")].dropna(subset=["value"]).sort_values("year")
    us["cpi"] = (1 + us.value / 100).cumprod()
    us["cpi"] /= us.loc[us.year == 2015, "cpi"].iloc[0]
    ct = ct.merge(us[["year", "cpi"]], on="year", validate="many_to_one")
    ct["flow"] = ct.value_usd / ct.cpi / 1e6
    records = []
    coverage = []
    for iso, g in ct.groupby("iso3"):
        g = g.set_index("year").reindex(range(1996, 2025))
        flow = g.flow
        filled = flow.interpolate(limit_area="inside").fillna(0)
        coverage.append([NAMES[iso], str(flow.notna().sum()), str(flow.isna().sum()),
                         str(int(flow[flow.notna()].index.min()))])
        variants = {"base": (flow.fillna(0), 0.10, 0.0),
                    "interior_interpolation": (filled, 0.10, 0.0),
                    "depreciation05": (flow.fillna(0), 0.05, 0.0),
                    "depreciation15": (flow.fillna(0), 0.15, 0.0),
                    # Explicit initial-stock stress test, not an estimated pre-1996 stock.
                    "initial_stock": (flow.fillna(0), 0.10, float(flow.dropna().head(3).mean() / 0.10))}
        stocks = {}
        for name, (imports, depreciation, initial) in variants.items():
            stock = initial
            out = []
            for value in imports:
                stock = (1 - depreciation) * stock + value
                out.append(stock)
            stocks[name] = out
        complete5 = flow.notna().rolling(5, min_periods=5).sum().eq(5)
        for j, year in enumerate(g.index):
            row = dict(iso3=iso, year=year, recent5_reported=bool(complete5.loc[year]))
            row.update({"stock_" + name: values[j] for name, values in stocks.items()})
            records.append(row)
    stocks = pd.DataFrame(records)
    # The denominator exactly reproduces the robot measure in build_panel.
    p = p.merge(stocks, on=["iso3", "year"], how="left", validate="one_to_one")
    employment = p.pwt_emp.fillna(p.ilo_emp / 1000)
    variants = ["base", "interior_interpolation", "depreciation05", "depreciation15", "initial_stock"]
    for name in variants:
        p["robot_" + name] = np.log1p(p["stock_" + name] / employment)
        p["robot_" + name + "_l1"] = p.groupby("iso3")["robot_" + name].shift(1)
    p['recent5_l1'] = p.groupby('iso3').recent5_reported.shift(1).eq(True)
    available = p.ln_robots_pw.notna()
    np.testing.assert_allclose(p.loc[available, "robot_base"], p.loc[available, "ln_robots_pw"], rtol=1e-11, atol=1e-11)
    write_table(coverage, ["Reported years", "Missing years", "First report"],
                "tab_measurement_robot_coverage", "Industrial-robot import-value coverage, 1996-2024",
                "tab:measurement_robot_coverage", "Each country has 29 possible annual observations. A missing import value "
                "is not evidence of zero imports. Stocks in the baseline set missing flows to zero; this table audits that assumption "
                "against the archived raw Comtrade file.")
    rows = []
    labels = {"base": "Zero fill, initial zero, 10% depreciation",
              "interior_interpolation": "Interpolate internal gaps, edge gaps zero",
              "depreciation05": "Zero fill, initial zero, 5% depreciation",
              "depreciation15": "Zero fill, initial zero, 15% depreciation",
              "initial_stock": "Initial stock = first 3 reported flows / 0.10"}
    samples = {"baseline": p, "recent5": p[p.recent5_l1], "observed_window": p[p.year <= 2019]}
    for key, data in samples.items():
        common = data.dropna(subset=["labsh_any", "k_l1"] + CONTROLS + ["robot_" + v + "_l1" for v in variants])
        for name in variants if key == "baseline" else ["base"]:
            xs = ["k_l1", "robot_" + name + "_l1"] + CONTROLS
            res = fit(common, "labsh_any", xs)
            title = labels[name] if key == "baseline" else ("Five latest lagged flows reported" if key == "recent5" else "Outcome years through 2019")
            robot = record(res, xs[1], "robot", title)
            capital = record(res, "k_l1", "robot_capital", title)
            rows.append([title, cell(robot), f"{robot['p']:.3f}", cell(capital), str(robot['n']), str(robot['countries'])])
    write_table(rows, ["Robot b (SE)", "p", "Capital b (SE)", "N", "Countries"],
                "tab_measurement_robots", "H1a. Sensitivity to construction of the robot-import stock proxy",
                "tab:measurement_robots", "Outcome: PWT/ILO labour share, 0-1. Each slope is per one sample SD of its own "
                "lagged logged regressor. Country/year FE and four controls; CR1 SE, t p-values with countries minus one df. "
                "First five rows use identical observations. Internal-gap interpolation uses adjacent reported real-dollar flows; "
                "unbounded edge gaps remain zero. The initial-stock stress test sets the end-1995 stock to the mean of the first "
                "three reported annual real imports divided by 0.10; it uses future information and is not a historical estimate. "
                "Five latest flows reported means each of t-5 through t-1 is observed; older gaps can still affect the stock. "
                "These last two restrictions change sample composition. Robot stocks measure depreciated import value per total "
                "worker in 2015 US dollars, not physical robots or manufacturing robot density.")


def vdem_sensitivity(p):
    # Reverse the physical-integrity endpoints because higher repression is 1-integrity.
    p = p.copy()
    p["rep_lower"] = 1 - p.v2x_clphy_codehigh
    p["rep_upper"] = 1 - p.v2x_clphy_codelow
    assert (p.rep_lower <= p.rep_phys + 1e-12).all()
    assert (p.rep_phys <= p.rep_upper + 1e-12).all()
    specifications = [("Basic", ["k_l1"] + CONTROLS),
                      ("Anti-system threat", ["k_l1", "antisys_l5", "k_x_antisys", "milexp_gdp_l1", "mil_support_l1"] + CONTROLS)]
    rows = []
    for name, xs in specifications:
        sample = p.dropna(subset=xs + ["rep_phys", "rep_lower", "rep_upper"])
        for outcome, label in [("rep_lower", "Lower repression endpoint"), ("rep_phys", "Point estimate"), ("rep_upper", "Upper repression endpoint")]:
            res = fit(sample, outcome, xs)
            capital = record(res, "k_l1", "vdem", name + ": " + label)
            interaction = record(res, "k_x_antisys", "vdem_interaction", name + ": " + label)
            rows.append([name + ": " + label, cell(capital), f"{capital['p']:.3f}",
                         cell(interaction) if interaction else "", str(capital['n']), str(capital['countries'])])
    write_table(rows, ["Capital b (SE)", "p", "Interaction b (SE)", "N", "Countries"],
                "tab_measurement_vdem", "H2. Physical-repression endpoint sensitivity",
                "tab:measurement_vdem", "Outcome units: physical repression 0-1; regressors standardized. Lower repression "
                "= 1 minus V-Dem codehigh; upper repression = 1 minus codelow. Country/year FE, controls, CR1 SE; t p-values "
                "use countries minus one df. Within each specification the same observations and point-estimate covariates "
                "are held fixed. V-Dem endpoints summarize approximately 68% marginal measurement-model probability. "
                "Applying an endpoint simultaneously to every observation is a deterministic stress test: it is neither a "
                "joint posterior draw nor a confidence interval for the regression coefficient. Interaction-model capital "
                "coefficients refer to anti-system threat=0.")


def rubin_pool(q, u, df_complete):
    """Rubin total variance and Barnard-Rubin small-sample df.

    Cluster df G-1 is supplied as the complete-data approximation; there is no
    claim that this produces exact inference with 18 observational countries.
    """
    q, u = np.asarray(q), np.asarray(u)
    m = len(q)
    qbar = q.mean(axis=0)
    ubar = u.mean(axis=0)
    between = np.cov(q, rowvar=False, ddof=1)
    total = ubar + (1 + 1 / m) * between
    return qbar, ubar, between, total


def pooled_contrast(qbar, ubar, between, total, vector, m, df_complete):
    b = float(vector @ qbar)
    variance = float(vector @ total @ vector)
    between_v = float(vector @ between @ vector)
    within_v = float(vector @ ubar @ vector)
    lam = max(0.0, (1 + 1 / m) * between_v / variance)
    tmp = (1 - lam) * (1 + df_complete) * df_complete
    df = (m - 1) * tmp / ((df_complete + 3) * (m - 1) + lam * lam * tmp)
    se = np.sqrt(variance)
    p = float(2 * student_t.sf(abs(b / se), df))
    ci = student_t.ppf(.975, df) * se
    return dict(b=b, se=float(se), p=p, lower=float(b-ci), upper=float(b+ci),
                within_variance=within_v, between_variance=between_v, df=float(df), imputations=m)


def swiid_sensitivity(p):
    with zipfile.ZipFile(RAW / "swiid9_92.zip") as archive:
        member = "swiid9_92/swiid9_92.dta"
        cols = ["country", "year", "abs_red"] + [f"_{m}_{variable}" for variable in
                ['abs_red', 'gini_mkt', 'gini_disp'] for m in range(1, 101)]
        sw = pd.read_stata(io.BytesIO(archive.read(member)), columns=cols).copy()
    # Check which archived representation is actually being used before pooling.
    base_observed = int(sw.abs_red.notna().sum())
    draw_identity_error = 0.0
    for m in range(1, 101):
        released = sw[f'_{m}_abs_red'].to_numpy(float)
        derived = sw[f'_{m}_gini_mkt'].to_numpy(float) - sw[f'_{m}_gini_disp'].to_numpy(float)
        available = np.isfinite(released)
        np.testing.assert_allclose(released[available], derived[available], atol=1e-8, rtol=0)
        draw_identity_error = max(draw_identity_error, float(np.max(np.abs(released[available]-derived[available]))))
    mapping = {name: iso for iso, name in NAMES.items()}
    sw['iso3'] = sw.country.map(mapping)
    sw = sw[sw.iso3.notna()].copy()
    draws = [f"_{m}_abs_red" for m in range(1, 101)]
    # Wide MI files may keep nonimputed observations in the base variable.
    for column in draws:
        sw[column] = sw[column].fillna(sw.abs_red)
    sw = sw[['iso3', 'year'] + draws].copy()
    # The Stata imputations are float32 on disk. Promote before averaging so
    # that the fixed-design linearity check is not limited by float32 sums.
    sw[draws] = sw[draws].astype('float64')
    p = p.merge(sw, on=['iso3', 'year'], how='left', validate='one_to_one')
    summary = pd.read_csv(RAW / 'swiid9_92_summary.csv')
    summary['iso3'] = summary.country.map(mapping)
    summary = summary[summary.iso3.notna()][['iso3', 'year', 'abs_red']].rename(columns={'abs_red': 'summary_abs_red'})
    representation = p[['iso3', 'country', 'year', 'abs_red']].merge(
        summary, on=['iso3', 'year'], how='left', validate='one_to_one')
    np.testing.assert_allclose(representation.abs_red, representation.summary_abs_red,
                               atol=0, rtol=0, equal_nan=True)
    representation['draw_mean_abs_red'] = p[draws].mean(axis=1)
    representation['available_draws'] = p[draws].notna().sum(axis=1)
    representation = representation[representation.summary_abs_red.notna() &
                                    representation.available_draws.eq(100)].copy()
    representation['mean_minus_summary'] = representation.draw_mean_abs_red-representation.summary_abs_red
    representation['absolute_difference'] = representation.mean_minus_summary.abs()
    representation = representation.drop(columns='abs_red').sort_values(['iso3', 'year'])
    representation.to_csv(TABLES / 'tab_measurement_swiid_representation.csv', index=False)
    largest = representation.loc[representation.absolute_difference.idxmax()]
    changed = int((representation.absolute_difference > 1e-10).sum())
    print(f"SWIID representations: {len(representation)} paired rows, {changed} differ; "
          f"max absolute difference={largest.absolute_difference:.6f} ({largest.iso3} {int(largest.year)}); "
          f"base abs_red nonmissing={base_observed}; max draw identity error={draw_identity_error:.2g}", flush=True)
    specifications = [
        ('Redistribution: basic', 'abs_red', ['k_l1'] + CONTROLS),
        ('Redistribution: interaction', 'abs_red', ['k_l1', 'rep_phys_l1', 'k_x_rep'] + CONTROLS),
        ('Relative choice: anti-system', 'rep_choice', ['k_l1', 'antisys_l5', 'k_x_antisys', 'milexp_gdp_l1', 'mil_support_l1'] + CONTROLS)]
    red_mean, red_sd = p.abs_red.mean(), p.abs_red.std()
    rep_mean, rep_sd = p.rep_phys.mean(), p.rep_phys.std()
    rows = []
    for name, y, xs in specifications:
        d = p.dropna(subset=[y] + xs + draws).copy()
        res = fit(d, y, xs)
        X = res._X
        q, u = [], []
        for m, column in enumerate(draws, start=1):
            yy = d[column] if y == 'abs_red' else (d.rep_phys-rep_mean)/rep_sd - (d[column]-red_mean)/red_sd
            draw_fit = sm.OLS(yy.astype(float), X).fit(cov_type='cluster', cov_kwds={'groups': d.iso3})
            q.append(draw_fit.params.to_numpy())
            u.append(draw_fit.cov_params().to_numpy())
        q, u = np.asarray(q), np.asarray(u)
        qbar, ubar, between, total = rubin_pool(q, u, d.iso3.nunique()-1)
        # For a fixed design OLS is linear in the outcome: validate averaging.
        draw_mean = d[draws].mean(axis=1)
        yy_mean = draw_mean if y == 'abs_red' else (d.rep_phys-rep_mean)/rep_sd - (draw_mean-red_mean)/red_sd
        np.testing.assert_allclose(qbar, sm.OLS(yy_mean, X).fit().params, atol=1e-9, rtol=1e-9)
        assert np.linalg.eigvalsh(total).min() > -1e-8
        contrasts = []
        for variable in ['k_l1', 'rep_phys_l1', 'k_x_rep']:
            if variable in X:
                vector = np.zeros(X.shape[1]); vector[X.columns.get_loc(variable)] = 1
                contrasts.append((variable, vector))
        if 'rep_phys_l1' in X:
            # One SD of lagged repression, evaluated at median observed capital.
            vector = np.zeros(X.shape[1]); vector[X.columns.get_loc('rep_phys_l1')] = 1
            vector[X.columns.get_loc('k_x_rep')] = d.k_l1.median() * d.rep_phys_l1.std() / d.k_x_rep.std()
            contrasts.append(('Repression at median capital', vector))
        for label, vector in contrasts:
            out = pooled_contrast(qbar, ubar, between, total, vector, 100, d.iso3.nunique()-1)
            out.update(block='swiid', specification=name, variable=label, n=len(d), countries=int(d.iso3.nunique()),
                       first_year=int(d.year.min()), last_year=int(d.year.max()))
            DETAILS.append(out)
            summary_b = float(vector @ res.params)
            summary_se = float(np.sqrt(vector @ res.cov_params() @ vector))
            rows.append([name + ': ' + label, f'{summary_b:.4f} ({summary_se:.4f})', cell(out),
                         f"{out['p']:.3f}", f"[{out['lower']:.3f}, {out['upper']:.3f}]", str(len(d))])
    write_table(rows, ['Summary b (SE)', 'MI b (SE)', 'MI p', 'MI 95% CI', 'N'],
                'tab_measurement_swiid', 'H2. SWIID uncertainty propagated through 100 released imputations',
                'tab:measurement_swiid', 'All 100 official absolute-redistribution imputations are read directly from the archived '
                'SWIID 9.92 Stata wide file. No independent normal draws are simulated. Every fit uses identical complete-case '
                'country-years, covariates and standardization. Within-imputation CR1 country-cluster covariance and '
                'between-imputation covariance are combined by Rubin rules, T=Ubar+(1+1/100)B. Approximate t inference uses '
                'Barnard-Rubin df with complete-data df=countries-1. Redistribution outcomes are Gini points; relative choice '
                'uses z(repression)-z(redistribution) with fixed full-window summary means/SDs in every imputation. '
                'Raw-product interaction main coefficients are evaluated at the other variable=0. The last redistribution '
                'contrast instead reports one SD of repression at median observed capital, with full covariance. '
                'The released draw means differ from the archived summary CSV in '
                f'{changed} of {len(representation)} paired 1990-2024 country-years; the largest absolute difference is '
                f'{largest.absolute_difference:.3f} Gini points. Thus this comparison changes point estimates as well as '
                'propagating variance; it is not purely an uncertainty correction. The companion representation CSV '
                'documents those differences without inferring source error or version incompatibility. '
                'Other data-source uncertainty is not propagated.')


def observed_window(p):
    rows = []
    specifications = [
        ('H1a labour share', 'labsh_any', ['k_l1'] + CONTROLS),
        ('H2 physical repression', 'rep_phys', ['k_l1'] + CONTROLS),
        ('H2 redistribution', 'abs_red', ['k_l1', 'rep_phys_l1', 'k_x_rep'] + CONTROLS)]
    for name, y, xs in specifications:
        for window, d in [('1990-2024', p), ('1990-2019', p[p.year <= 2019])]:
            res = fit(d, y, xs)
            item = record(res, 'k_l1', 'observed_window', name + ': ' + window)
            rows.append([name + ': ' + window, cell(item), f"{item['p']:.3f}", str(item['n']), str(item['countries'])])
    write_table(rows, ['Capital b (SE)', 'p', 'N', 'Countries'], 'tab_measurement_observed',
                'Capital associations before the post-2019 capital extension', 'tab:measurement_observed',
                'Continuous regressors standardized within each estimation sample. Country/year FE and baseline controls; '
                'CR1 country-cluster SE and t p-values with countries minus one df. Labour-share outcome units are 0-1, '
                'physical repression 0-1 and redistribution Gini points. The restricted window excludes all post-2019 '
                'outcomes and their constructed capital inputs. Window restriction changes sample and regressor SDs. '
                'The redistribution capital coefficient is the raw-product-model main coefficient at repression=0.')


def write_brief():
    frame = pd.DataFrame(DETAILS)
    frame.to_csv(TABLES / 'tab_measurement_estimates.csv', index=False)
    (TABLES / 'tab_measurement_estimates.json').write_text(json.dumps(DETAILS, indent=2), encoding='utf-8')
    lines = ['# Measurement and sample completion', '',
             'Generated by `03_results/scripts/08_measurement_robustness.py` from archived raw inputs.', '',
             'Completed: H1a/H2 common-sample comparisons; raw Comtrade missingness audit and robot-stock construction '
             'sensitivities; capital-outcome windows ending in 2019; V-Dem physical-repression endpoint substitutions; '
             '100-imputation SWIID pooling for the main H2 redistribution and relative-choice regressions.', '',
             'Each estimate below is generated by code. Coefficients use outcome units per standardized continuous '
             'regressor (see table captions). Fixed effects and baseline controls are included. No result identifies a causal effect.', '']
    def pick(block, specification, variable='k_l1'):
        return next(item for item in DETAILS if item['block'] == block
                    and item['specification'] == specification and item['variable'] == variable)
    h1 = pick('h1_common', 'PWT/ILO: capital: common')
    h1r = pick('h1_common', 'PWT/ILO: capital + robots: common')
    h1o = pick('h1_common', 'PWT/ILO: capital + routine: common')
    h1i = pick('h1_common', 'ILO: capital + robots: common')
    h2m = pick('h2_common', 'Physical: mobilization threat: common')
    h2a = pick('h2_common', 'Physical: anti-system threat: common')
    robot = frame[frame.block == 'robot'].head(5)
    vdem = frame[(frame.block == 'vdem') & frame.specification.str.startswith('Basic:')]
    pooled = pick('swiid', 'Redistribution: interaction')
    rep = pick('swiid', 'Redistribution: interaction', 'Repression at median capital')
    choice = pick('swiid', 'Relative choice: anti-system')
    old = pick('observed_window', 'H1a labour share: 1990-2019')
    representation = pd.read_csv(TABLES / 'tab_measurement_swiid_representation.csv')
    largest = representation.loc[representation.absolute_difference.idxmax()]
    changed = int((representation.absolute_difference > 1e-10).sum())
    lines += ['## Manuscript-ready numerical findings', '',
              f"On the common H1a sample (N={h1['n']}, {h1['countries']} countries, {h1['first_year']}-{h1['last_year']}), "
              f"the capital coefficient is {cell(h1)} for the basic PWT/ILO outcome, {cell(h1r)} with the robot control, "
              f"{cell(h1o)} with routine occupations and {cell(h1i)} for the ILO outcome with robots. "
              "All are negative, but magnitudes and uncertainty vary. Multiply these labour-share coefficients by 100 to obtain percentage points.", '',
              f"On the common H2 sample (N={h2m['n']}, {h2m['countries']} countries), the physical-repression capital main "
              f"coefficient is {cell(h2m)} with the mobilization threat proxy and {cell(h2a)} with anti-system threat. "
              "These are coefficients at threat=0, and neither is precisely estimated.", '',
              f"Holding N={int(robot.n.iloc[0])} and {int(robot.countries.iloc[0])} countries fixed, robot-proxy slopes range from "
              f"{robot.b.min():.6f} to {robot.b.max():.6f}; the smallest t-reference p-value is {robot.p.min():.3f}. "
              f"Across V-Dem endpoint substitutions, the basic capital/repression coefficient ranges from {vdem.b.min():.6f} "
              f"to {vdem.b.max():.6f} (N={int(vdem.n.iloc[0])}); the smallest p-value is {vdem.p.min():.3f}.", '',
              f"For redistribution with the repression interaction (N={pooled['n']}, {pooled['countries']} countries), "
              f"100-imputation pooling gives a capital coefficient of {cell(pooled)} (p={pooled['p']:.3f}). "
              f"The one-SD repression contrast at median capital is {cell(rep)} Gini points (p={rep['p']:.3f}, "
              f"95% CI [{rep['lower']:.3f}, {rep['upper']:.3f}]). The relative-choice capital coefficient is "
              f"{cell(choice)} (p={choice['p']:.3f}, N={choice['n']}, {choice['countries']} countries). "
              "The change in point estimates and increase in standard errors relative to summary-only estimation require a more cautious interpretation of redistribution findings.", '',
              f"Ending the outcome window in 2019 leaves the basic labour-share/capital association negative: "
              f"{cell(old)}, N={old['n']}, {old['countries']} countries. The nominal 1990-2024 complete-case specifications "
              "in these checks end in 2021 because of control coverage.", '',
              'Full-precision estimates, sample dates and country counts: `03_results/tables/tab_measurement_estimates.csv` '
              '(also JSON). Seven display tables have stems `tab_measurement_common_h1`, `tab_measurement_common_h2`, '
              '`tab_measurement_robots`, `tab_measurement_robot_coverage`, `tab_measurement_vdem`, `tab_measurement_swiid` '
              'and `tab_measurement_observed`.', '']
    lines += ['## Archived SWIID representations', '',
              f"The source check pairs {len(representation)} country-years in 1990-2024 with nonmissing summary values and all 100 "
              f"released imputations. Their means differ from the archived CSV summary in {changed} cases (absolute tolerance 1e-10). "
              f"The largest absolute difference is {largest.absolute_difference:.6f} Gini points in {largest.country} "
              f"({int(largest.year)}): summary={largest.summary_abs_red:.6f}, released-draw mean={largest.draw_mean_abs_red:.6f}. "
              "The clean-panel summary values match the archived summary CSV exactly. All 100 released absolute-redistribution "
              "draws also match the corresponding market-minus-disposable Gini draws, and the source file's base abs_red column "
              "has no nonmissing observations in this archive.", '',
              "The comparison therefore replaces summary point values with means of released draws as well as adding within- and "
              "between-imputation uncertainty. Differences in the fitted coefficients must not be attributed solely to variance propagation. "
              "This check documents two archived representations and does not establish source error, a version mismatch, or the cause "
              "of their differences. Paired country/year values, signed differences and draw counts are in "
              "`03_results/tables/tab_measurement_swiid_representation.csv`.", '',
              '## Interpretation and remaining limits', '',
              'Common-sample comparisons isolate some changes due to composition but do not cure selection from the threat definition. '
              'Robot variants address three explicit construction assumptions; import values, CPI deflation and total-employment '
              'denominators remain imperfect proxies for robot adoption. The five-reported-flow restriction does not erase older gaps.', '',
              'The 100 released SWIID imputations preserve their paired country-year draws. Rubin pooling includes within-draw clustered '
              'uncertainty and between-draw covariance; the fixed reference scale makes the relative-choice coefficient comparable across draws. '
              'The Barnard-Rubin calculation with G-1 complete-data degrees of freedom is an approximation for a small country-cluster sample. '
              'This completed propagation covers the displayed main H2 specifications; exploratory thresholds and forest outputs retain their '
              'summary-data status. No artificial imputation from independent summary standard errors was used.', '',
              'V-Dem endpoint estimates are deterministic substitutions of available marginal bounds. They are not regression confidence '
              'bounds, full posterior integration or joint credible trajectories. Full propagation would require matched joint posterior draws '
              'across country-years and all V-Dem variables in each specification; the archived marginal bounds do not supply those objects. '
              'Identification limits, actor-based coup classification and external instrument validity remain substantive limits, not software tasks.', '',
              'Methods checked against primary documentation: '
              '[SWIID guidance](https://fsolt.org/swiid/swiid_faq/posts/how-to/mean-and-std-err.html), '
              '[Rubin pooling](https://amices.org/mice/reference/pool.scalar.html), '
              '[Barnard-Rubin implementation](https://github.com/amices/mice/blob/master/R/barnard.rubin.R), '
              '[V-Dem methodology](https://www.v-dem.net/about/v-dem-project/methodology/).', '',
              'Verification: reconstructed baseline robot logs match archived panel at 1e-11 tolerance; V-Dem endpoint ordering checked; '
              'pooled fixed-design OLS coefficients equal coefficients of the draw-mean outcome at 1e-9 tolerance; pooled covariance checked '
              'for numerical positive semidefiniteness.']
    (ROOT / '05_red_team' / 'measurement_completion.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    from common import load_panel
    p = load_panel((1990, 2024))
    p['k_x_threat'] = p.k_l1 * p.threat_mob_l1
    p['k_x_antisys'] = p.k_l1 * p.antisys_l5
    p['k_x_rep'] = p.k_l1 * p.rep_phys_l1
    p['rep_choice'] = (p.rep_phys-p.rep_phys.mean())/p.rep_phys.std() - (p.abs_red-p.abs_red.mean())/p.abs_red.std()
    for action in [common_samples, robot_sensitivity, vdem_sensitivity, swiid_sensitivity, observed_window]:
        print('Running ' + action.__name__, flush=True)
        action(p)
    write_brief()
    print(f'Done: {len(DETAILS)} measurement/sampling estimates and completion note.', flush=True)


if __name__ == '__main__':
    main()
