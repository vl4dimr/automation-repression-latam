"""Build the country-year panel from the raw files.

Outputs (02_data/clean/):
  panel.csv, panel.parquet   one row per country-year, 1960-2025, 18 countries
  dictionary.csv             variable, description, unit, source, construction
  build_log.txt              coverage report

Run: python 02_data/scripts/02_build_panel.py
"""
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CLEAN, DONORS, ISO3, NAMES, RAW, REFERENCE, YEAR_MAX, YEAR_MIN

DICT = []          # (variable, description, unit, source, construction)
LOGLINES = []


def log(s):
    print(s, flush=True)
    LOGLINES.append(str(s))


def d(var, desc, unit, src, how=""):
    DICT.append(dict(variable=var, description=desc, unit=unit, source=src, construction=how))


SWIID_NAMES = {"Argentina": "ARG", "Bolivia": "BOL", "Brazil": "BRA", "Chile": "CHL", "Colombia": "COL",
               "Costa Rica": "CRI", "Dominican Republic": "DOM", "Ecuador": "ECU", "El Salvador": "SLV",
               "Guatemala": "GTM", "Honduras": "HND", "Mexico": "MEX", "Nicaragua": "NIC", "Panama": "PAN",
               "Paraguay": "PRY", "Peru": "PER", "Uruguay": "URY", "Venezuela": "VEN"}

# ----------------------------------------------------------------------------- skeleton
years = list(range(YEAR_MIN, YEAR_MAX + 1))
panel = pd.MultiIndex.from_product([ISO3, years], names=["iso3", "year"]).to_frame(index=False)
panel["country"] = panel.iso3.map(NAMES)
d("iso3", "ISO3 country code", "", "", "")
d("year", "Calendar year", "", "", "")
d("country", "Country name", "", "", "")


def merge(df, keys=("iso3", "year")):
    global panel
    panel = panel.merge(df, on=list(keys), how="left")


# ----------------------------------------------------------------------------- V-Dem
vd = pd.read_parquet(RAW / "vdem_v16_subset.parquet")
vd = vd[vd.country_text_id.isin(ISO3)].rename(columns={"country_text_id": "iso3"})
vd["year"] = vd.year.astype(int)
keep = {
    "v2x_clphy": "V-Dem physical violence index (higher = more respect)",
    "v2x_clphy_codelow": "v2x_clphy lower bound", "v2x_clphy_codehigh": "v2x_clphy upper bound",
    "v2csreprss": "V-Dem CSO repression (higher = less repression; measurement-model scale)",
    "v2x_freexp_altinf": "V-Dem freedom of expression and alternative information index",
    "v2cagenmob": "V-Dem mass mobilization (higher = more)",
    "v2cademmob": "V-Dem mobilization for democracy", "v2caautmob": "V-Dem mobilization for autocracy",
    "v2caassemb": "V-Dem freedom of peaceful assembly (higher = more respected)",
    "v2clkill": "V-Dem freedom from political killings", "v2cltort": "V-Dem freedom from torture",
    "v2x_regime": "V-Dem Regimes of the World (0 closed aut., 1 electoral aut., 2 electoral dem., 3 liberal dem.)",
    "v2x_polyarchy": "V-Dem electoral democracy index", "v2x_libdem": "V-Dem liberal democracy index",
    "v2x_civlib": "V-Dem civil liberties index", "v2x_clpol": "V-Dem political civil liberties",
    "v2x_clpriv": "V-Dem private civil liberties",
    "e_pt_coup": "Number of successful coups (Powell & Thyne, via V-Dem)",
    "e_pt_coup_attempts": "Number of coup attempts (Powell & Thyne, via V-Dem)",
    "v2regimpgroup": "Regime most important support group (V-Dem code)",
    "v2csantimv": "V-Dem anti-system movements (higher = stronger)",
    "e_civil_war": "Civil war (V-Dem historical)", "v2x_ex_military": "Military dimension of executive",
    "v2x_corr": "V-Dem political corruption index",
}
cols = [c for c in keep if c in vd.columns]
missing = [c for c in keep if c not in vd.columns]
log(f"V-Dem: using {len(cols)} vars; not in file: {missing}")
merge(vd[["iso3", "year"] + cols])
for c in cols:
    d(c, keep[c], "index", "V-Dem v16", "as published")

panel["rep_phys"] = 1 - panel.v2x_clphy
d("rep_phys", "Physical repression = 1 - v2x_clphy", "0-1", "V-Dem v16", "reversed index")
panel["rep_cso"] = -panel.v2csreprss
d("rep_cso", "CSO repression = -v2csreprss", "std. scale", "V-Dem v16", "sign reversed")
panel["rep_expr"] = 1 - panel.v2x_freexp_altinf
d("rep_expr", "Expression repression = 1 - v2x_freexp_altinf", "0-1", "V-Dem v16", "reversed index")
panel["mob"] = panel.v2cagenmob
d("mob", "Mass mobilization (V-Dem v2cagenmob)", "std. scale", "V-Dem v16", "as published")
panel["dem"] = (panel.v2x_regime >= 2).astype(float).where(panel.v2x_regime.notna())
d("dem", "Democracy = 1 if v2x_regime in {2,3}", "0/1", "V-Dem v16", "")
panel["coup"] = (panel.e_pt_coup > 0).astype(float).where(panel.e_pt_coup.notna())
d("coup", "Successful coup in year", "0/1", "Powell & Thyne via V-Dem e_pt_coup", "")
panel["coup_att"] = (panel.e_pt_coup_attempts > 0).astype(float).where(panel.e_pt_coup_attempts.notna())
d("coup_att", "Coup attempt in year (successful or not)", "0/1", "Powell & Thyne via V-Dem", "")
if "v2regimpgroup" in panel:
    panel["mil_support"] = (panel.v2regimpgroup == 5).astype(float).where(panel.v2regimpgroup.notna())
    panel["labor_support"] = panel.v2regimpgroup.isin([9, 11]).astype(float).where(panel.v2regimpgroup.notna())
    d("mil_support", "Regime's main support group is the military", "0/1", "V-Dem v2regimpgroup", "code 5")
    d("labor_support", "Regime's main support group is urban or rural working class", "0/1", "V-Dem v2regimpgroup", "codes 9, 11")

# ----------------------------------------------------------------------------- ERT
ert = pd.read_csv(RAW / "ert_v16.csv")
ert = ert[ert.country_text_id.isin(ISO3)].rename(columns={"country_text_id": "iso3"})
ert["aut_onset"] = (ert.year == ert.aut_ep_start_year).astype(float)
ert["dem_onset"] = (ert.year == ert.dem_ep_start_year).astype(float)
merge(ert[["iso3", "year", "aut_ep", "aut_onset", "aut_ep_start_year", "aut_ep_outcome", "dem_ep", "dem_onset", "reg_type"]])
d("aut_ep", "Inside an autocratization episode (ERT)", "0/1", "ERT v16", "")
d("aut_onset", "First year of an autocratization episode", "0/1", "ERT v16", "year == aut_ep_start_year")
d("aut_ep_outcome", "ERT autocratization outcome code", "code", "ERT v16", "")
d("dem_ep", "Inside a democratization episode (ERT)", "0/1", "ERT v16", "")
d("dem_onset", "First year of a democratization episode", "0/1", "ERT v16", "")
d("reg_type", "ERT regime type (0 autocracy, 1 democracy)", "0/1", "ERT v16", "")

# ----------------------------------------------------------------------------- PWT
import pyreadstat
pwt, _ = pyreadstat.read_dta(str(RAW / "pwt1001.dta"))
pwt = pwt[pwt.countrycode.isin(ISO3 + list(DONORS) + list(REFERENCE))].rename(columns={"countrycode": "iso3"})
pwt["year"] = pwt.year.astype(int)
pcols = ["rnna", "cn", "emp", "labsh", "rgdpna", "pop", "hc", "delta", "csh_i", "rtfpna"]
pcols = [c for c in pcols if c in pwt.columns]
pw = pwt[pwt.iso3.isin(ISO3)][["iso3", "year"] + pcols].rename(columns={c: f"pwt_{c}" for c in pcols})
merge(pw)
for c in pcols:
    d(f"pwt_{c}", f"PWT 10.01 {c}", "PWT units (mil. 2017 US$, millions of persons, shares)", "PWT 10.01", "as published")

# ----------------------------------------------------------------------------- WDI
wdi = pd.read_csv(RAW / "wdi.csv")
wdi_w = wdi.pivot_table(index=["iso3", "year"], columns="variable", values="value").reset_index()
merge(wdi_w[wdi_w.iso3.isin(ISO3)])
wdi_desc = {
    "gdppc_const2015": "GDP per capita, constant 2015 US$", "gdp_const2015": "GDP, constant 2015 US$",
    "pop_wdi": "Population", "urban_share": "Urban population, % of total", "trade_open": "Trade, % of GDP",
    "resource_rents": "Total natural resource rents, % of GDP", "tax_rev_wdi": "Tax revenue, % of GDP (central govt)",
    "gfcf_const2015": "Gross fixed capital formation, constant 2015 US$", "manuf_va_share": "Manufacturing value added, % of GDP",
    "milexp_gdp": "Military expenditure, % of GDP", "armed_forces": "Armed forces personnel, total",
    "labor_force": "Labor force, total", "fdi_in_gdp": "FDI net inflows, % of GDP",
    "exp_fuel_share": "Fuel exports, % of merchandise exports", "exp_metal_share": "Ores and metals exports, % of merchandise exports",
    "exp_agri_share": "Agricultural raw materials exports, % of merchandise exports", "inflation": "CPI inflation, annual %",
    "unemp": "Unemployment, % of labor force (ILO modelled)",
}
for k, v in wdi_desc.items():
    if k in panel:
        d(k, v, "", "WDI (World Bank API, pull 2026-09)", "as published")

# ----------------------------------------------------------------------------- ILO
def ilo_annual(path):
    x = pd.read_csv(path, low_memory=False)
    x = x[x.TIME_PERIOD.astype(str).str.fullmatch(r"\d{4}")]
    x["year"] = x.TIME_PERIOD.astype(int)
    return x.rename(columns={"REF_AREA": "iso3", "OBS_VALUE": "value"})


ilo_ls = ilo_annual(RAW / "ilo_labshare.csv")
ilo_ls = ilo_ls[ilo_ls.iso3.isin(ISO3)][["iso3", "year", "value"]].rename(columns={"value": "ilo_labsh"})
ilo_ls["ilo_labsh"] = ilo_ls.ilo_labsh / 100
merge(ilo_ls.drop_duplicates(["iso3", "year"]))
d("ilo_labsh", "Labour income share of GDP (ILO modelled estimates)", "share", "ILOSTAT LAP_2GDP_NOC_RT", "/100")

ilo_emp = ilo_annual(RAW / "ilo_emp_modelled.csv")
ilo_emp = ilo_emp[(ilo_emp.SEX == "SEX_T") & (ilo_emp.AGE == "AGE_YTHADULT_YGE15")]
ilo_emp_all = ilo_emp[["iso3", "year", "value"]].rename(columns={"value": "ilo_emp"}).drop_duplicates(["iso3", "year"])
merge(ilo_emp_all[ilo_emp_all.iso3.isin(ISO3)])
d("ilo_emp", "Employment, thousands (ILO modelled, 15+)", "thousands", "ILOSTAT EMP_2EMP_SEX_AGE_NB", "")

# Routine occupation share: ISCO major groups 4 (clerical), 7 (craft), 8 (plant and machine operators)
occ = ilo_annual(RAW / "ilo_emp_occ.csv")
occ = occ[(occ.SEX == "SEX_T") & occ.iso3.isin(ISO3)]
rows = []
for (iso, yr), g in occ.groupby(["iso3", "year"]):
    for cls in ("ISCO08", "ISCO88"):
        gg = g[g.OCU.str.startswith(f"OCU_{cls}_")]
        tot = gg[gg.OCU == f"OCU_{cls}_TOTAL"].value
        if len(tot) and tot.iloc[0] > 0:
            v = {k: gg[gg.OCU == f"OCU_{cls}_{k}"].value.sum() for k in ["4", "7", "8", "9", "1", "2", "3"]}
            rows.append(dict(iso3=iso, year=yr, occ_cls=cls, tot=tot.iloc[0],
                             routine_share=(v["4"] + v["7"] + v["8"]) / tot.iloc[0],
                             elementary_share=v["9"] / tot.iloc[0],
                             highskill_share=(v["1"] + v["2"] + v["3"]) / tot.iloc[0]))
            break
occ_df = pd.DataFrame(rows)
merge(occ_df[["iso3", "year", "routine_share", "elementary_share", "highskill_share", "occ_cls"]])
d("routine_share", "Employment share of routine-intensive occupations (ISCO 4+7+8)", "share", "ILOSTAT EMP_TEMP_SEX_OCU_NB",
  "ISCO-08 where available, ISCO-88 otherwise; annual national LFS")
d("elementary_share", "Employment share of elementary occupations (ISCO 9)", "share", "ILOSTAT", "")
d("highskill_share", "Employment share of ISCO 1-3", "share", "ILOSTAT", "")
d("occ_cls", "ISCO classification used in routine_share", "", "ILOSTAT", "")

# Manufacturing employment share for shares of the shift-share instrument
manuf = None
p = RAW / "ilo_emp_eco_modelled.csv"
if p.exists():
    eco = ilo_annual(p)
    eco = eco[eco.SEX == "SEX_T"]
    eco_col = [c for c in eco.columns if c.startswith("ECO")][0]
    m = eco[eco[eco_col] == "ECO_ISIC4_C"].groupby(["iso3", "year"]).value.first().rename("man")
    t = eco[eco[eco_col] == "ECO_ISIC4_TOTAL"].groupby(["iso3", "year"]).value.first().rename("tot")
    mm = pd.concat([m, t], axis=1).dropna().reset_index()
    mm["manuf_emp_share"] = mm.man / mm.tot
    manuf = mm[["iso3", "year", "manuf_emp_share"]].drop_duplicates(["iso3", "year"])
    log(f"ILO manufacturing share (modelled, ISIC4 C): {len(mm)} rows")
if manuf is not None:
    merge(manuf[manuf.iso3.isin(ISO3)])
    d("manuf_emp_share", "Manufacturing share of employment (ILO modelled, ISIC rev.4 section C)", "share",
      "ILOSTAT EMP_2EMP_SEX_ECO_NB", "C / TOTAL")

# ----------------------------------------------------------------------------- CEPALSTAT
soc = pd.read_csv(RAW / "cepal_socexp_functions_b_3127.csv")
soc = soc[soc.iso3.isin(ISO3)].rename(columns={"Years__ESTANDAR": "year"})
soc["year"] = soc.year.astype(int)
socx = soc[soc["Classification of the functions of government"] == "Social expenditure"]
sp = soc[soc["Classification of the functions of government"] == "Social protection"]


def pick_cov(x, name):
    cg = x[x["Institutional coverage"] == "Central government"][["iso3", "year", "value"]].rename(columns={"value": name})
    gg = x[x["Institutional coverage"] == "General government"][["iso3", "year", "value"]].rename(columns={"value": name + "_gg"})
    return cg.drop_duplicates(["iso3", "year"]), gg.drop_duplicates(["iso3", "year"])


a, b = pick_cov(socx, "socexp_gdp"); merge(a); merge(b)
a, b = pick_cov(sp, "socprot_gdp"); merge(a); merge(b)
d("socexp_gdp", "Public social expenditure, % of GDP, central government", "% GDP", "CEPALSTAT ind. 3127", "COFOG social functions")
d("socexp_gdp_gg", "Public social expenditure, % of GDP, general government", "% GDP", "CEPALSTAT ind. 3127", "")
d("socprot_gdp", "Social protection expenditure, % of GDP, central government", "% GDP", "CEPALSTAT ind. 3127", "COFOG 10")
d("socprot_gdp_gg", "Social protection expenditure, % of GDP, general government", "% GDP", "CEPALSTAT ind. 3127", "")

tax = pd.read_csv(RAW / "cepal_tax_rev_821.csv")
tax = tax[tax.iso3.isin(ISO3)].rename(columns={"Years__ESTANDAR": "year"})
tax["year"] = tax.year.astype(int)
for cls, nm in (("Tax revenue", "tax_gdp"), ("Tax revenue (including social contributions)", "tax_sc_gdp")):
    a, b = pick_cov(tax[tax.Tax_classification == cls], nm); merge(a); merge(b)
d("tax_gdp", "Tax revenue, % of GDP, central government", "% GDP", "CEPALSTAT ind. 821", "")
d("tax_gdp_gg", "Tax revenue, % of GDP, general government", "% GDP", "CEPALSTAT ind. 821", "")
d("tax_sc_gdp", "Tax revenue incl. social contributions, % of GDP, central government", "% GDP", "CEPALSTAT ind. 821", "")
d("tax_sc_gdp_gg", "Tax revenue incl. social contributions, % of GDP, general government", "% GDP", "CEPALSTAT ind. 821", "")

# ----------------------------------------------------------------------------- Andersson & Brambor tax revenue
gr_path = RAW / "govrev_v3.csv"
if gr_path.exists():
    gr = pd.read_csv(gr_path, low_memory=False)
    gr = gr[gr.ccodewb.isin(ISO3)].rename(columns={"ccodewb": "iso3"})
    gr["tax_ab_gdp"] = pd.to_numeric(gr.centaxgdp, errors="coerce")   # already in percent of GDP
    merge(gr[["iso3", "year", "tax_ab_gdp"]].drop_duplicates(["iso3", "year"]))
    d("tax_ab_gdp", "Central government tax revenue, % of GDP (Andersson & Brambor v3, 1800-2024)", "% GDP",
      "Financing the State v3", "centaxgdp (percent); 11 of 18 countries")

# ----------------------------------------------------------------------------- SWIID
sw = pd.read_csv(RAW / "swiid9_92_summary.csv")
sw["iso3"] = sw.country.map(SWIID_NAMES)
sw = sw[sw.iso3.notna()][["iso3", "year", "gini_disp", "gini_mkt", "abs_red", "rel_red", "gini_disp_se", "gini_mkt_se"]]
merge(sw)
d("gini_disp", "Gini, disposable income (SWIID)", "0-100", "SWIID 9.92", "")
d("gini_mkt", "Gini, market income (SWIID)", "0-100", "SWIID 9.92", "")
d("abs_red", "Absolute redistribution = gini_mkt - gini_disp", "Gini points", "SWIID 9.92", "")
d("rel_red", "Relative redistribution = abs_red / gini_mkt", "%", "SWIID 9.92", "")

# ----------------------------------------------------------------------------- PTS
pts = pd.read_excel(RAW / "PTS-2025.xlsx")
pts = pts[pts.WordBank_Code_A.isin(ISO3)].rename(columns={"WordBank_Code_A": "iso3", "Year": "year"})
pts["pts_mean"] = pts[["PTS_A", "PTS_H", "PTS_S"]].mean(axis=1)
merge(pts[["iso3", "year", "PTS_A", "PTS_H", "PTS_S", "pts_mean"]].drop_duplicates(["iso3", "year"]))
d("PTS_A", "Political Terror Scale, Amnesty", "1-5", "PTS-2025", "")
d("PTS_H", "Political Terror Scale, Human Rights Watch", "1-5", "PTS-2025", "")
d("PTS_S", "Political Terror Scale, US State Dept", "1-5", "PTS-2025", "")
d("pts_mean", "Mean of available PTS scores", "1-5", "PTS-2025", "")

# ----------------------------------------------------------------------------- Freedom House
fh = pd.read_excel(RAW / "FIW_1973-2024.xlsx", sheet_name="Country Ratings, Statuses ", header=None)
yrs = fh.iloc[1, 1:].tolist(); hdr = [str(h).strip() for h in fh.iloc[2, 1:].tolist()]   # early columns read 'CL ' with a space
recs = []
FHN = {"Argentina": "ARG", "Bolivia": "BOL", "Brazil": "BRA", "Chile": "CHL", "Colombia": "COL", "Costa Rica": "CRI",
       "Dominican Republic": "DOM", "Ecuador": "ECU", "El Salvador": "SLV", "Guatemala": "GTM", "Honduras": "HND",
       "Mexico": "MEX", "Nicaragua": "NIC", "Panama": "PAN", "Paraguay": "PRY", "Peru": "PER", "Uruguay": "URY",
       "Venezuela": "VEN"}
for r in range(3, len(fh)):
    name = str(fh.iloc[r, 0]).strip()
    if name not in FHN:
        continue
    cur_year = None
    for j, h in enumerate(hdr):
        y = yrs[j]
        if pd.notna(y):
            import re
            found = re.findall(r"(19\d\d|20\d\d)", str(y))
            cur_year = int(found[-1]) if found else None   # spans like 'Jan.1981-Aug. 1982' -> 1982
        if cur_year is None or h not in ("PR", "CL"):
            continue
        v = fh.iloc[r, j + 1]
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = np.nan
        recs.append(dict(iso3=FHN[name], year=cur_year, var="fh_" + h.lower(), value=v))
fhw = pd.DataFrame(recs).pivot_table(index=["iso3", "year"], columns="var", values="value").reset_index()
merge(fhw)
d("fh_cl", "Freedom House civil liberties rating (1 free - 7 not free)", "1-7", "FIW 1973-2024", "year under review")
d("fh_pr", "Freedom House political rights rating", "1-7", "FIW 1973-2024", "")

# ----------------------------------------------------------------------------- MMP
mmp = pd.read_csv(RAW / "mmp_073120.csv", low_memory=False)
mmp = mmp[mmp.country.isin(SWIID_NAMES)].copy()
mmp["iso3"] = mmp.country.map(SWIID_NAMES)
resp_cols = [c for c in mmp.columns if c.startswith("stateresponse")]
REPRESSIVE = {"crowd dispersal", "arrests", "shootings", "killings", "beatings"}
mmp["repressive_resp"] = mmp[resp_cols].apply(lambda r: any(str(v).strip().lower() in REPRESSIVE for v in r), axis=1)
mmp["viol"] = mmp.protesterviolence.fillna(0)


def parse_part(x):
    s = str(x).replace(",", "").lower()
    import re
    m = re.findall(r"\d+", s)
    if m:
        return float(m[0])
    return {"hundreds": 300, "thousands": 3000, "tens of thousands": 30000, "dozens": 36}.get(s.strip(), np.nan)


mmp["part_num"] = mmp.participants.apply(parse_part)
ag = mmp.groupby(["iso3", "year"]).agg(mmp_protests=("protest", "sum"),
                                       mmp_repressed=("repressive_resp", "sum"),
                                       mmp_participants=("part_num", "sum"),
                                       mmp_violent=("viol", "sum")).reset_index()
ag["mmp_repress_share"] = ag.mmp_repressed / ag.mmp_protests.replace(0, np.nan)
merge(ag)
d("mmp_protests", "Number of protest events (>=50 participants) in year", "count", "Mass Mobilization Project v5", "sum of protest flag")
d("mmp_repressed", "Protests met with a repressive state response", "count", "MMP v5", "any of dispersal/arrests/shootings/killings/beatings")
d("mmp_repress_share", "Share of protests met with repression", "share", "MMP v5", "")
d("mmp_participants", "Sum of reported participants (categories imputed)", "persons", "MMP v5", "hundreds=300, thousands=3000, tens of thousands=30000")
d("mmp_violent", "Protests with protester violence", "count", "MMP v5", "")

# ----------------------------------------------------------------------------- Comtrade robots
ct_path = RAW / "comtrade_847950_imports.csv"
if ct_path.exists():
    ct = pd.read_csv(ct_path)
    ct["year"] = ct.year.astype(int)
    # US CPI index from WDI inflation (USA) to deflate nominal USD to 2015 prices
    us = wdi_w[wdi_w.iso3 == "USA"][["year", "inflation"]].dropna().sort_values("year")
    us["cpi"] = (1 + us.inflation / 100).cumprod()
    us["cpi"] = us.cpi / us.loc[us.year == 2015, "cpi"].iloc[0]
    ct = ct.merge(us[["year", "cpi"]], on="year", how="left")
    ct["rob_imp_real"] = ct.value_usd / ct.cpi / 1e6   # millions of 2015 US$
    ct = ct.sort_values(["iso3", "year"])
    DEP = 0.10
    stocks = []
    for iso, g in ct.groupby("iso3"):
        g = g.set_index("year").reindex(range(1996, 2025))
        m = g.rob_imp_real.fillna(0.0)
        s = 0.0; out = {}
        for y in range(1996, 2025):
            s = (1 - DEP) * s + m.loc[y]
            out[y] = s
        stocks.append(pd.DataFrame(dict(iso3=iso, year=list(out), rob_stock_real=list(out.values()),
                                        rob_imp_real=m.values, rob_imp_missing=g.rob_imp_real.isna().values)))
    st = pd.concat(stocks)
    # Employment for per-worker normalisation: PWT emp (millions) extended with ILO modelled employment growth
    emp_all = pwt[["iso3", "year", "emp"]].copy()
    ilo_all = ilo_emp_all.copy()
    st = st.merge(emp_all, on=["iso3", "year"], how="left").merge(ilo_all, on=["iso3", "year"], how="left")
    st["emp_use"] = st.emp
    st.loc[st.emp_use.isna() & st.ilo_emp.notna(), "emp_use"] = st.ilo_emp / 1000
    st["robots_pw"] = st.rob_stock_real / st.emp_use          # million 2015 US$ per million workers = US$ per worker
    st["rob_imp_pw"] = st.rob_imp_real / st.emp_use
    lat = st[st.iso3.isin(ISO3)]
    merge(lat[["iso3", "year", "rob_imp_real", "rob_imp_missing", "rob_stock_real", "robots_pw", "rob_imp_pw"]])
    d("rob_imp_real", "Imports of industrial robots (HS 847950), million 2015 US$", "mil. 2015 US$", "UN Comtrade", "deflated with US CPI")
    d("rob_imp_missing", "Robot import value not reported to Comtrade that year (set to 0 in stock)", "0/1", "UN Comtrade", "")
    d("rob_stock_real", "Robot capital stock, PIM with 10% depreciation from 1996", "mil. 2015 US$", "UN Comtrade", "S_t = 0.9 S_{t-1} + M_t")
    d("robots_pw", "Robot stock per worker", "2015 US$ per worker", "Comtrade + PWT/ILO employment", "")
    d("rob_imp_pw", "Robot imports per worker", "2015 US$ per worker", "Comtrade + PWT/ILO employment", "")
    # Donor shift: pooled donor imports per worker
    dn = st[st.iso3.isin(DONORS)].groupby("year").agg(m=("rob_imp_real", "sum"), s=("rob_stock_real", "sum"), e=("emp_use", "sum")).reset_index()
    dn["donor_rob_pw"] = dn.s / dn.e
    dn["donor_imp_pw"] = dn.m / dn.e
    merge(dn[["year", "donor_rob_pw", "donor_imp_pw"]], keys=("year",))
    d("donor_rob_pw", "Robot stock per worker, 14 non-Latin American donor economies pooled", "2015 US$ per worker", "Comtrade", "shift of the instrument")
    d("donor_imp_pw", "Robot imports per worker, donors pooled", "2015 US$ per worker", "Comtrade", "")
    log(f"Comtrade: {ct.iso3.nunique()} reporters, {len(ct)} country-years; LatAm reported share = {1 - lat.rob_imp_missing.mean():.2f}")
else:
    log("Comtrade file not present; robot variables skipped")

# ----------------------------------------------------------------------------- Pink Sheet
ps = pd.read_excel(RAW / "CMO-Historical-Data-Annual.xlsx", sheet_name="Annual Indices (Real)", header=None)
hdr_rows = ps.iloc[5:9].astype(str).agg(" ".join).str.replace("nan", "").str.replace("\n", " ").str.strip()
data = ps.iloc[9:].copy()
data.columns = ["year"] + hdr_rows.tolist()[1:]
data = data[data.year.astype(str).str.fullmatch(r"\d{4}(\.0)?")]
data["year"] = data.year.astype(float).astype(int)
def find(col_kw):
    for c in data.columns[1:]:
        if all(k.lower() in c.lower() for k in col_kw):
            return c
    return None
c_en, c_me, c_ag = find(["Energy"]), find(["Metals"]), find(["Agriculture"])
pk = data[["year", c_en, c_me, c_ag]].rename(columns={c_en: "p_energy", c_me: "p_metals", c_ag: "p_agri"})
for c in ["p_energy", "p_metals", "p_agri"]:
    pk[c] = pd.to_numeric(pk[c], errors="coerce")
merge(pk, keys=("year",))
d("p_energy", "World Bank real energy price index, 2010=100", "index", "Pink Sheet Sept 2026", "")
d("p_metals", "World Bank real metals and minerals price index", "index", "Pink Sheet Sept 2026", "")
d("p_agri", "World Bank real agriculture price index", "index", "Pink Sheet Sept 2026", "")

# ----------------------------------------------------------------------------- constructed variables
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
g = panel.groupby("iso3")

# capital per worker, extended 2020-2025 by PIM with WDI real GFCF growth and ILO employment growth
panel["K"] = panel.pwt_rnna
panel["L"] = panel.pwt_emp
panel["labsh"] = panel.pwt_labsh
panel["k_extended"] = 0.0
for iso, gi in panel.groupby("iso3"):
    idx = gi.index
    last = gi[gi.pwt_rnna.notna()].year.max()
    if pd.isna(last):
        continue
    delta = gi.pwt_delta.dropna().iloc[-1] if gi.pwt_delta.notna().any() else 0.04
    K_prev = gi.loc[gi.year == last, "pwt_rnna"].iloc[0]
    K_prev2 = gi.loc[gi.year == last - 1, "pwt_rnna"].iloc[0]
    I_last = K_prev - (1 - delta) * K_prev2
    gf_last = gi.loc[gi.year == last, "gfcf_const2015"].iloc[0]
    L_prev = gi.loc[gi.year == last, "pwt_emp"].iloc[0]
    ilo_last = gi.loc[gi.year == last, "ilo_emp"].iloc[0]
    ls_last = gi.loc[gi.year == last, "pwt_labsh"].iloc[0]
    ilo_ls_last = gi.loc[gi.year == last, "ilo_labsh"].iloc[0]
    for y in range(last + 1, YEAR_MAX + 1):
        row = gi[gi.year == y]
        if row.empty:
            continue
        i = row.index[0]
        gf = row.gfcf_const2015.iloc[0]
        if pd.notna(gf) and pd.notna(gf_last) and gf_last > 0 and pd.notna(I_last):
            I_y = I_last * gf / gf_last
            K_y = (1 - delta) * K_prev + I_y
            panel.loc[i, "K"] = K_y; panel.loc[i, "k_extended"] = 1.0
            K_prev = K_y
        ie = row.ilo_emp.iloc[0]
        if pd.notna(ie) and pd.notna(ilo_last) and pd.notna(L_prev):
            panel.loc[i, "L"] = L_prev * ie / ilo_last
        il = row.ilo_labsh.iloc[0]
        if pd.notna(il) and pd.notna(ilo_ls_last) and pd.notna(ls_last):
            panel.loc[i, "labsh"] = ls_last * il / ilo_ls_last
panel["k"] = np.log(panel.K / panel.L)
panel["kl_level"] = panel.K / panel.L
d("K", "Capital stock (PWT rnna; 2020+ perpetual inventory with WDI real GFCF growth and PWT delta)", "mil. 2017 US$", "PWT + WDI", "")
d("L", "Employment (PWT emp; 2020+ ILO modelled employment growth)", "millions", "PWT + ILO", "")
d("labsh", "Labour share (PWT labsh; 2020+ spliced with ILO labour income share growth)", "share", "PWT + ILO", "")
d("k", "log capital per worker = log(K/L)", "log(mil. 2017 US$ per mil. workers)", "constructed", "")
d("kl_level", "Capital per worker, level", "2017 US$ per worker", "constructed", "")
d("k_extended", "k uses the 2020+ extension", "0/1", "constructed", "")
# fiscal capacity: CEPALSTAT central government (1990+), then Andersson-Brambor, then WDI
panel["tax_any"] = panel.tax_gdp
if "tax_ab_gdp" in panel:
    panel["tax_any"] = panel.tax_any.fillna(panel.tax_ab_gdp)
panel["tax_any"] = panel.tax_any.fillna(panel.tax_rev_wdi)
d("tax_any", "Tax revenue % GDP: CEPALSTAT, else Andersson-Brambor, else WDI", "% GDP", "constructed", "splice for 1960-2024")
panel["labsh_any"] = panel.labsh.fillna(panel.ilo_labsh)
d("labsh_any", "Labour share: PWT where available, else ILO", "share", "constructed", "used for El Salvador")
panel["ln_sk_sl"] = np.log((1 - panel.labsh_any) / panel.labsh_any)
d("ln_sk_sl", "log(capital share / labour share)", "log ratio", "constructed", "the model's inequality term")

# repression composite: mean of standardized rep_phys, rep_cso, rep_expr
for c in ["rep_phys", "rep_cso", "rep_expr"]:
    panel[c + "_z"] = (panel[c] - panel[c].mean()) / panel[c].std()
panel["rep_index"] = panel[["rep_phys_z", "rep_cso_z", "rep_expr_z"]].mean(axis=1)
d("rep_index", "Repression composite: mean of standardized rep_phys, rep_cso, rep_expr", "z", "constructed", "")

# regime state classification: high repression = above sample median of rep_phys (1990-2024)
med = panel.loc[(panel.year >= 1990) & (panel.year <= 2024), "rep_phys"].median()
panel["high_rep"] = (panel.rep_phys > med).astype(float).where(panel.rep_phys.notna())
d("high_rep", "rep_phys above the 1990-2024 sample median", "0/1", "constructed", f"median = {med:.3f}")
panel["rep_tercile"] = pd.qcut(panel.loc[panel.year >= 1990, "rep_phys"], 3, labels=[1, 2, 3]).astype(float)
d("rep_tercile", "Tercile of rep_phys (1990+)", "1-3", "constructed", "")

# threat proxies
panel["mob_l5"] = g.mob.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
d("mob_l5", "Lagged 5-year mean of mass mobilization", "std. scale", "constructed", "years t-5..t-1")
low = panel.mob.where(panel.high_rep == 0)
panel["threat_mob"] = low.groupby(panel.iso3).transform(lambda s: s.shift(1).rolling(10, min_periods=3).mean())
d("threat_mob", "Lagged 10-year mean of mobilization using only low-repression years", "std. scale", "constructed",
  "proxy for beta that is not censored by contemporaneous repression")
panel["antisys_l1"] = g.v2csantimv.shift(1) if "v2csantimv" in panel else np.nan
d("antisys_l1", "Lagged anti-system movement strength", "std. scale", "V-Dem", "")

# cost / capacity of repression
panel["armed_pc"] = panel.armed_forces / panel.pop_wdi * 1000
d("armed_pc", "Armed forces per 1,000 population", "per 1000", "WDI", "")
panel["coldwar"] = (panel.year <= 1990).astype(float)
d("coldwar", "Cold War years (<=1990)", "0/1", "constructed", "")
panel["decade"] = (panel.year // 10 * 10).astype(int)
d("decade", "Decade", "", "constructed", "")

# instruments
def first5(gdf):
    x = gdf[gdf.year >= 1990].dropna(subset=["exp_fuel_share", "exp_metal_share", "exp_agri_share"]).head(5)
    return x[["exp_fuel_share", "exp_metal_share", "exp_agri_share"]].mean() / 100
base = panel.groupby("iso3").apply(first5, include_groups=False)
base = base.rename(columns={"exp_fuel_share": "w_fuel", "exp_metal_share": "w_metal", "exp_agri_share": "w_agri"})
panel = panel.merge(base, left_on="iso3", right_index=True, how="left")
panel["z_commodity"] = (panel.w_fuel * np.log(panel.p_energy) + panel.w_metal * np.log(panel.p_metals)
                        + panel.w_agri * np.log(panel.p_agri))
d("w_fuel", "Fuel share of merchandise exports, mean of first 5 available years from 1990", "share", "WDI", "share of the commodity instrument")
d("w_metal", "Ores and metals share of merchandise exports, same window", "share", "WDI", "")
d("w_agri", "Agricultural raw materials share of merchandise exports, same window", "share", "WDI", "")
d("z_commodity", "Commodity price instrument = sum_c w_c * log(real price index_c)", "log points", "constructed",
  "energy, metals, agriculture; Pink Sheet real indices")
if "manuf_emp_share" in panel:
    w_m = panel[(panel.year >= 1993) & (panel.year <= 1997)].groupby("iso3").manuf_emp_share.mean().rename("w_manuf")
else:
    w_m = pd.Series(dtype=float, name="w_manuf")
w_mva = (panel[(panel.year >= 1993) & (panel.year <= 1997)].groupby("iso3").manuf_va_share.mean() / 100).rename("w_manuf_va")
panel = panel.merge(w_m, left_on="iso3", right_index=True, how="left").merge(w_mva, left_on="iso3", right_index=True, how="left")
panel["w_manuf_use"] = panel.w_manuf.fillna(panel.w_manuf_va)
if "donor_rob_pw" in panel:
    panel["z_robot"] = panel.w_manuf_use * np.log1p(panel.donor_rob_pw)
    d("z_robot", "Shift-share instrument = 1993-97 manufacturing share x log(1 + donor robot stock per worker)", "", "constructed", "")
d("w_manuf", "Manufacturing employment share, 1993-97 mean (ILO)", "share", "ILOSTAT", "")
d("w_manuf_va", "Manufacturing value added share of GDP, 1993-97 mean (WDI)", "share", "WDI", "fallback share")
d("w_manuf_use", "Share used in z_robot (ILO employment share, else WDI value-added share)", "share", "constructed", "")

# logs and transforms used in estimation
panel["ln_gdppc"] = np.log(panel.gdppc_const2015)
panel["ln_robots_pw"] = np.log1p(panel.robots_pw) if "robots_pw" in panel else np.nan
panel["ln_mmp_protests"] = np.log1p(panel.mmp_protests)
d("ln_gdppc", "log GDP per capita (constant 2015 US$)", "log", "WDI", "")
d("ln_robots_pw", "log(1 + robot stock per worker)", "log", "constructed", "")
d("ln_mmp_protests", "log(1 + MMP protest count)", "log", "constructed", "")
panel["in_panel"] = ((panel.year >= 1990) & (panel.year <= 2024)).astype(int)
d("in_panel", "Inside the 1990-2024 baseline window", "0/1", "constructed", "")

# ----------------------------------------------------------------------------- save
panel = panel.sort_values(["iso3", "year"]).reset_index(drop=True)
panel.to_csv(CLEAN / "panel.csv", index=False)
panel.to_parquet(CLEAN / "panel.parquet", index=False)
pd.DataFrame(DICT).drop_duplicates("variable").to_csv(CLEAN / "dictionary.csv", index=False)

# coverage report
core = ["rep_phys", "rep_cso", "mob", "k", "labsh_any", "ln_robots_pw", "routine_share", "abs_red", "socexp_gdp",
        "tax_gdp", "pts_mean", "fh_cl", "mmp_protests", "coup", "dem", "z_commodity", "z_robot"]
core = [c for c in core if c in panel]
cov = panel[panel.in_panel == 1].groupby("iso3")[core].apply(lambda x: x.notna().sum())
log("\nNon-missing country-years per variable, 1990-2024:")
log(cov.to_string())
log(f"\nPanel shape: {panel.shape}; countries {panel.iso3.nunique()}; years {panel.year.min()}-{panel.year.max()}")
(CLEAN / "build_log.txt").write_text("\n".join(LOGLINES), encoding="utf-8")
print("saved panel.csv / panel.parquet / dictionary.csv")
