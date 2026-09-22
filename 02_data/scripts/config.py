"""Shared configuration: paths, country lists, seed, source registry.

Every script in 02_data/ and 03_results/ imports this module so that the
sample, the seed and the file locations are defined once.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "02_data" / "raw"
CLEAN = ROOT / "02_data" / "clean"
RESULTS = ROOT / "03_results"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / os.environ.get("FIG_DIR", "figures")
for p in (RAW, CLEAN, TABLES, FIGURES):
    p.mkdir(parents=True, exist_ok=True)

SEED = 20260907

# Baseline sample: 18 Latin American countries. Cuba is absent from PWT 10.01;
# Haiti is dropped for data quality. Both decisions are documented in 01_design.md.
LATAM = {
    "ARG": ("Argentina", 32), "BOL": ("Bolivia", 68), "BRA": ("Brazil", 76),
    "CHL": ("Chile", 152), "COL": ("Colombia", 170), "CRI": ("Costa Rica", 188),
    "DOM": ("Dominican Republic", 214), "ECU": ("Ecuador", 218),
    "SLV": ("El Salvador", 222), "GTM": ("Guatemala", 320), "HND": ("Honduras", 340),
    "MEX": ("Mexico", 484), "NIC": ("Nicaragua", 558), "PAN": ("Panama", 591),
    "PRY": ("Paraguay", 600), "PER": ("Peru", 604), "URY": ("Uruguay", 858),
    "VEN": ("Venezuela", 862),
}
ISO3 = list(LATAM)
M49 = {k: v[1] for k, v in LATAM.items()}
NAMES = {k: v[0] for k, v in LATAM.items()}

# Donor countries for the shift-share instrument: non-Latin American
# middle-income economies with robot-import series in Comtrade. The shift is
# their robot imports per worker; Latin America is excluded by construction.
DONORS = {
    "TUR": 792, "ZAF": 710, "THA": 764, "MYS": 458, "IDN": 360, "POL": 616,
    "ROU": 642, "PHL": 608, "MAR": 504, "EGY": 818, "IND": 356, "VNM": 704,
    "CZE": 203, "HUN": 348,
}
# High-income reference economies (descriptive comparison only)
REFERENCE = {"USA": 842, "DEU": 276, "JPN": 392, "KOR": 410}

YEAR_MIN, YEAR_MAX = 1960, 2025      # raw window (H4 uses 1960+)
PANEL_MIN, PANEL_MAX = 1990, 2024    # baseline panel window

COMTRADE_YEARS = list(range(1996, 2025))  # HS 847950 exists from HS1996

# World Bank WDI indicators
WDI = {
    "NY.GDP.PCAP.KD": "gdppc_const2015",
    "NY.GDP.MKTP.KD": "gdp_const2015",
    "SP.POP.TOTL": "pop_wdi",
    "SP.URB.TOTL.IN.ZS": "urban_share",
    "NE.TRD.GNFS.ZS": "trade_open",
    "NY.GDP.TOTL.RT.ZS": "resource_rents",
    "GC.TAX.TOTL.GD.ZS": "tax_rev_wdi",
    "NE.GDI.FTOT.KD": "gfcf_const2015",
    "NV.IND.MANF.ZS": "manuf_va_share",
    "MS.MIL.XPND.GD.ZS": "milexp_gdp",
    "MS.MIL.TOTL.P1": "armed_forces",
    "SL.TLF.TOTL.IN": "labor_force",
    "BX.KLT.DINV.WD.GD.ZS": "fdi_in_gdp",
    "TX.VAL.FUEL.ZS.UN": "exp_fuel_share",
    "TX.VAL.MMTL.ZS.UN": "exp_metal_share",
    "TX.VAL.AGRI.ZS.UN": "exp_agri_share",
    "FP.CPI.TOTL.ZG": "inflation",
    "SL.UEM.TOTL.ZS": "unemp",
}

# CEPALSTAT indicators (id -> short name); members are resolved at download
CEPAL = {
    3126: "cepal_socexp_functions",   # public social expenditure by COFOG function
    3127: "cepal_socexp_functions_b",
    3128: "cepal_socexp_functions_c",
    821: "cepal_tax_rev",             # tax revenue by type, % GDP
    4405: "cepal_socx",               # SOCX public social spending
}

# ILOSTAT SDMX dataflows
ILO = {
    "DF_LAP_2GDP_NOC_RT": "ilo_labshare",          # labour income share, % GDP (modelled)
    "DF_EMP_2EMP_SEX_AGE_NB": "ilo_emp_modelled",  # employment, thousands (modelled)
    "DF_EMP_TEMP_SEX_OCU_NB": "ilo_emp_occ",       # employment by ISCO major group
    "DF_EMP_TEMP_SEX_ECO_NB": "ilo_emp_eco",       # employment by ISIC section
    "DF_EMP_2EMP_SEX_ECO_NB": "ilo_emp_eco_modelled",
}

SOURCES = {
    "vdem": dict(name="V-Dem Country-Year v16 (vdemdata R package)",
                 url="https://github.com/vdeminstitute/vdemdata", version="16.0",
                 license="CC BY-SA 4.0 (V-Dem data policy)"),
    "ert": dict(name="Episodes of Regime Transformation (ERT) v16",
                url="https://github.com/vdeminstitute/ERT", version="16", license="CC BY-SA 4.0"),
    "pwt": dict(name="Penn World Table 10.01", url="https://www.rug.nl/ggdc/productivity/pwt/",
                version="10.01 (DataverseNL file 354098)", license="CC BY 4.0"),
    "comtrade": dict(name="UN Comtrade, HS 847950 industrial robots, imports, public preview API",
                     url="https://comtradeapi.un.org/public/v1/preview", version="pull 2026-09",
                     license="UN Comtrade terms of use (free public data)"),
    "wdi": dict(name="World Bank World Development Indicators API v2",
                url="https://api.worldbank.org/v2", version="pull 2026-09", license="CC BY 4.0"),
    "pts": dict(name="Political Terror Scale 1976-2024", url="http://www.politicalterrorscale.org",
                version="PTS-2025", license="free for academic use"),
    "fh": dict(name="Freedom House, Freedom in the World country ratings 1973-2024",
               url="https://freedomhouse.org/report/freedom-world", version="FIW 2024 file",
               license="free for non-commercial use"),
    "mmp": dict(name="Mass Mobilization Project 1990-2020", url="https://doi.org/10.7910/DVN/HTTWYL",
                version="v5 (mmALL_073120)", license="CC0"),
    "swiid": dict(name="Standardized World Income Inequality Database 9.92",
                  url="https://doi.org/10.7910/DVN/LM4OWF", version="9.92 (2026-04)", license="CC0"),
    "cepalstat": dict(name="CEPALSTAT API v1", url="https://api-cepalstat.cepal.org", version="pull 2026-09",
                      license="ECLAC open data"),
    "ilostat": dict(name="ILOSTAT SDMX REST API", url="https://sdmx.ilo.org/rest", version="pull 2026-09",
                    license="ILO open data"),
}
