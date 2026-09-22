"""Shared estimation helpers: panel loading, two-way FE OLS/IV with cluster-robust
and wild-cluster-bootstrap inference, Driscoll-Kraay, LaTeX table writers."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "02_data" / "scripts"))
from config import CLEAN, FIGURES, ISO3, NAMES, SEED, TABLES  # noqa: E402

RNG = np.random.default_rng(SEED)
CONTROLS = ["ln_gdppc", "urban_share", "trade_open", "resource_rents"]


def load_panel(window=(1990, 2024)):
    p = pd.read_parquet(CLEAN / "panel.parquet")
    p = p.sort_values(["iso3", "year"]).reset_index(drop=True)
    g = p.groupby("iso3")
    # lags used across scripts
    for v in ["k", "ln_robots_pw", "routine_share", "labsh_any", "ln_sk_sl", "rep_phys", "rep_cso", "rep_expr",
              "rep_index", "pts_mean", "abs_red", "socexp_gdp", "tax_gdp", "tax_rev_wdi", "dem", "v2x_regime",
              "v2x_polyarchy", "milexp_gdp", "armed_pc", "mil_support", "mob", "high_rep", "z_commodity", "z_robot",
              "ln_gdppc", "urban_share", "trade_open", "resource_rents", "fh_cl", "mmp_protests", "ln_mmp_protests",
              "threat_mob", "antisys_l1", "coup", "coup_att", "aut_onset"]:
        if v in p:
            p[v + "_l1"] = g[v].shift(1)
            p[v + "_l2"] = g[v].shift(2)
    p["dk"] = g.k.diff()
    # alternative threat proxy available for all countries: lagged 5-year mean of anti-system movement strength
    if "v2csantimv" in p:
        p["antisys_l5"] = g.v2csantimv.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    # smoother commodity instrument: lagged 5-year mean (capital responds to persistent price shifts)
    if "z_commodity" in p:
        p["z_commodity_ma5_l1"] = g.z_commodity.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
        p["z_commodity_d5_l1"] = g.z_commodity.transform(lambda s: s.shift(1) - s.shift(6))
    p["multiparty_l1"] = (p.v2x_regime_l1 >= 1).astype(float).where(p.v2x_regime_l1.notna())
    if window is not None:
        p = p[(p.year >= window[0]) & (p.year <= window[1])]
    return p.copy()


def zscore(s):
    return (s - s.mean()) / s.std()


def fe_residualize(values, df, fe=("iso3", "year")):
    """Exact projection; works on unbalanced panels and threshold interactions."""
    F = np.column_stack([np.ones(len(df))] + [
        pd.get_dummies(df[f], drop_first=True).to_numpy(dtype=float) for f in fe])
    a = np.asarray(values, dtype=float)
    return a - F @ np.linalg.lstsq(F, a, rcond=None)[0]


def remove_absorbed(X, xs, d, fe):
    residuals = fe_residualize(X[xs], d, fe)
    absorbed = [v for j, v in enumerate(xs)
                if np.linalg.norm(residuals[:, j]) < 1e-10 * max(1, np.linalg.norm(X[v]))]
    if absorbed:
        print('Absorbed by fixed effects; omitted: ' + ', '.join(absorbed), flush=True)
        X = X.drop(columns=absorbed)
    if np.linalg.matrix_rank(X.to_numpy()) < X.shape[1]:
        raise ValueError('Remaining design is rank deficient; inspect jointly collinear regressors')
    return X, absorbed


def twfe(df, y, xs, cluster="iso3", fe=("iso3", "year"), weights=None, std=True):
    """Two-way FE OLS by dummy-variable regression; cluster-robust SEs (CR1)."""
    cols = [y] + xs + list(fe)
    d = df.dropna(subset=cols).copy()
    if len(d) == 0:
        return None
    X = d[xs].astype(float).copy()
    if std:
        for c in xs:
            if d[c].nunique() > 2:
                X[c] = zscore(X[c])
    for f in fe:
        dums = pd.get_dummies(d[f], prefix=f, drop_first=True).astype(float)
        X = pd.concat([X, dums], axis=1)
    X = sm.add_constant(X)
    X, absorbed = remove_absorbed(X, xs, d, fe)
    mod = sm.OLS(d[y].astype(float), X)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    res._xs = xs; res._d = d; res._X = X; res._y = y
    res._absorbed = absorbed
    return res


def wild_cluster_p(res, var, B=999, cluster="iso3", seed=SEED):
    """Wild cluster bootstrap p-value (Rademacher, restricted, Cameron-Gelbach-Miller 2008)."""
    d, X, y = res._d, res._X, res._y
    yy = d[y].astype(float).values
    Xm = X.values
    j = list(X.columns).index(var)
    # restricted fit: impose beta_var = 0
    Xr = np.delete(Xm, j, axis=1)
    br = np.linalg.lstsq(Xr, yy, rcond=None)[0]
    ur = yy - Xr @ br
    groups = d[cluster].values
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    t_obs = res.tvalues[var]
    rng = np.random.default_rng(seed)
    XtX_inv = np.linalg.pinv(Xm.T @ Xm)
    ts = np.empty(B)
    for b in range(B):
        w = rng.choice([-1.0, 1.0], size=len(ug))
        ub = ur.copy()
        for gi, g in enumerate(ug):
            ub[idx[g]] *= w[gi]
        yb = Xr @ br + ub
        bb = XtX_inv @ (Xm.T @ yb)
        eb = yb - Xm @ bb
        # CR1 cluster variance for coefficient j
        meat = np.zeros((Xm.shape[1], Xm.shape[1]))
        for g in ug:
            xg = Xm[idx[g]]; eg = eb[idx[g]]
            s = xg.T @ eg
            meat += np.outer(s, s)
        G = len(ug); N, K = Xm.shape
        V = XtX_inv @ meat @ XtX_inv * (G / (G - 1)) * ((N - 1) / (N - K))
        ts[b] = bb[j] / np.sqrt(V[j, j])
    return float(np.mean(np.abs(ts) >= abs(t_obs)))


def iv2sls(df, y, endog, instr, exog, fe=("iso3", "year"), cluster="iso3", std=True):
    """2SLS with two-way FE dummies; returns (second stage results, first-stage F, exog names)."""
    from linearmodels.iv import IV2SLS
    cols = [y, endog] + instr + exog + list(fe)
    d = df.dropna(subset=cols).copy()
    if len(d) < 30:
        return None
    Z = d[instr].astype(float).copy(); W = d[exog].astype(float).copy(); E = d[[endog]].astype(float).copy()
    if std:
        for c in instr: Z[c] = zscore(Z[c])
        for c in exog:
            if d[c].nunique() > 2: W[c] = zscore(W[c])
        E[endog] = zscore(E[endog])
    for f in fe:
        W = pd.concat([W, pd.get_dummies(d[f], prefix=f, drop_first=True).astype(float)], axis=1)
    W = sm.add_constant(W)
    res = IV2SLS(d[y].astype(float), W, E, Z).fit(cov_type="clustered", clusters=d[cluster])
    # first stage F on excluded instruments (cluster-robust Wald)
    fs = sm.OLS(E[endog], pd.concat([W, Z], axis=1)).fit(cov_type="cluster", cov_kwds={"groups": d[cluster]})
    R = np.zeros((len(instr), len(fs.params)))
    for i, c in enumerate(instr):
        R[i, list(fs.params.index).index(c)] = 1
    F = float(fs.f_test(R).fvalue)
    res._fsF = F; res._n = len(d); res._d = d
    return res


def driscoll_kraay(df, y, xs, fe=("iso3", "year"), maxlags=3, std=True):
    cols = [y] + xs + list(fe)
    d = df.dropna(subset=cols).copy().sort_values(["year", "iso3"])
    X = d[xs].astype(float).copy()
    if std:
        for c in xs:
            if d[c].nunique() > 2: X[c] = zscore(X[c])
    for f in fe:
        X = pd.concat([X, pd.get_dummies(d[f], prefix=f, drop_first=True).astype(float)], axis=1)
    X = sm.add_constant(X)
    groups = d.year.astype(int).values
    X, _ = remove_absorbed(X, xs, d, fe)
    res = sm.OLS(d[y].astype(float), X).fit(cov_type="hac-groupsum", cov_kwds={"time": groups, "maxlags": maxlags})
    return res


def stars(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def fmt(b, se, p):
    return f"{b:.3f}{stars(p)} ({se:.3f})"


def write_table(rows, colnames, path_stem, caption, label, notes, rownames=None):
    """rows: list of lists of strings; writes .tex (booktabs) and .csv"""
    import re
    def safe(value):
        parts = str(value).split('$')
        for i in range(0, len(parts), 2):
            parts[i] = re.sub(r'(?<!\\)([%&_#])', r'\\\1', parts[i])
        return '$'.join(parts)
    tex = ["\\begin{table}[htbp]\\centering", f"\\caption{{{safe(caption)}}}", f"\\label{{{label}}}", "\\small",
           "\\resizebox{\\textwidth}{!}{%",
           "\\begin{tabular}{l" + "c" * len(colnames) + "}", "\\toprule",
           " & " + " & ".join(map(safe, colnames)) + " \\\\", "\\midrule"]
    for r in rows:
        tex.append(" & ".join(safe(x) for x in r) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}}", f"\\begin{{tablenotes}}\\footnotesize {safe(notes)}\\end{{tablenotes}}", "\\end{table}"]
    (TABLES / f"{path_stem}.tex").write_text("\n".join(tex), encoding="utf-8")
    pd.DataFrame(rows, columns=["row"] + colnames).to_csv(TABLES / f"{path_stem}.csv", index=False)


def coef_rows(results, xs_labels, bootstrap_vars=(), B=499):
    """Build table rows from a list of twfe results (columns) for given variables (rows)."""
    rows = []
    for var, lab in xs_labels:
        r = [lab]
        for res in results:
            if res is None or var not in res.params:
                r.append("")
            else:
                r.append(fmt(res.params[var], res.bse[var], res.pvalues[var]))
        rows.append(r)
        if var in bootstrap_vars:
            rb = ["\\quad wild-cluster bootstrap p"]
            for res in results:
                if res is None or var not in res.params:
                    rb.append("")
                else:
                    rb.append(f"[{wild_cluster_p(res, var, B=B):.3f}]")
            rows.append(rb)
    rows.append(["Observations"] + [str(int(res.nobs)) if res is not None else "" for res in results])
    rows.append(["Countries"] + [str(res._d.iso3.nunique()) if res is not None else "" for res in results])
    return rows
