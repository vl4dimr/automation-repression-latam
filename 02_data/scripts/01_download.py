"""Download every raw source into 02_data/raw.

Idempotent: existing files are kept; the Comtrade pull resumes from its CSV.
Each source is wrapped so that one failure does not stop the others.
Run:  python 02_data/scripts/01_download.py
Log:  02_data/raw/download_log.txt
"""
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from config import (CEPAL, COMTRADE_YEARS, DONORS, ILO, ISO3, LATAM, M49, RAW,
                    REFERENCE, WDI)

LOG = RAW / "download_log.txt"
UA = {"User-Agent": "Mozilla/5.0 (research; automation-repression-latam)"}


def log(msg):
    line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch(url, dest, timeout=600):
    if dest.exists() and dest.stat().st_size > 0:
        log(f"keep    {dest.name}")
        return dest
    r = requests.get(url, headers=UA, timeout=timeout)
    r.raise_for_status()
    dest.write_bytes(r.content)
    log(f"saved   {dest.name} ({len(r.content):,} B) <- {url}")
    return dest


# ----------------------------------------------------------------------------
def get_vdem():
    dest = RAW / "vdem_v16.RData"
    fetch("https://github.com/vdeminstitute/vdemdata/raw/master/data/vdem.RData", dest)
    out = RAW / "vdem_v16_subset.parquet"
    if out.exists():
        log("keep    vdem_v16_subset.parquet")
        return
    import pyreadr
    df = pyreadr.read_r(str(dest))["vdem"]
    keep = [c for c in df.columns if c.startswith(("country_", "year", "e_regionpol", "e_pt_",
                                                     "v2x_", "v2xcl_", "v2xnp_", "v2cl", "v2cs",
                                                     "v2ca", "v2me", "v2reg", "v2ju", "v2xlg",
                                                     "e_pop", "e_gdppc", "e_miurbani",
                                                     "e_civil_war", "e_miinteco", "e_miinterc",
                                                     "e_pt_coup", "v2exrescon", "v2x_ex_military",
                                                     "v2xnp_pres", "v2exl_legitideol", "v2exl_legitperf"))]
    sub = df[keep]
    sub.to_parquet(out, index=False)
    log(f"saved   vdem_v16_subset.parquet {sub.shape}")


def get_ert():
    fetch("https://raw.githubusercontent.com/vdeminstitute/ERT/master/inst/ert.csv", RAW / "ert_v16.csv")


def get_pwt():
    fetch("https://dataverse.nl/api/access/datafile/354098", RAW / "pwt1001.dta")


def get_pts():
    fetch("http://www.politicalterrorscale.org/Data/Files/PTS-2025.xlsx", RAW / "PTS-2025.xlsx")


def get_fh():
    fetch("https://freedomhouse.org/sites/default/files/2024-02/"
          "Country_and_Territory_Ratings_and_Statuses_FIW_1973-2024.xlsx",
          RAW / "FIW_1973-2024.xlsx")


def get_pinksheet():
    # The World Bank rotates the document hash monthly; the September 2026 link is
    # recorded here and the landing page (worldbank.org/en/research/commodity-markets)
    # is the place to refresh it.
    fetch("https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/"
          "CMO-Historical-Data-Annual.xlsx", RAW / "CMO-Historical-Data-Annual.xlsx")


def get_govrev():
    # Andersson & Brambor, Financing the State: government tax revenue 1800-2024 (v3),
    # distributed by the authors (perfandersson.com/data); 1800-2012 version at doi:10.7910/DVN/P1VAGP
    fetch("https://www.perfandersson.com/uploads/1/2/7/5/127505228/govrev_v3.csv", RAW / "govrev_v3.csv")


def get_mmp():
    fetch("https://dataverse.harvard.edu/api/access/datafile/4291456?format=original",
          RAW / "mmp_073120.csv")


def get_swiid():
    z = fetch("https://dataverse.harvard.edu/api/access/datafile/13657070", RAW / "swiid9_92.zip")
    out = RAW / "swiid9_92_summary.csv"
    if not out.exists():
        with zipfile.ZipFile(z) as zf:
            name = [n for n in zf.namelist() if n.endswith("summary.csv")][0]
            out.write_bytes(zf.read(name))
        log("saved   swiid9_92_summary.csv")


# ----------------------------------------------------------------------------
def get_wdi():
    out = RAW / "wdi.csv"
    if out.exists():
        log("keep    wdi.csv")
        return
    codes = ";".join(list(LATAM) + list(DONORS) + list(REFERENCE) + ["CUB", "HTI"])
    frames = []
    for ind, name in WDI.items():
        url = (f"https://api.worldbank.org/v2/country/{codes}/indicator/{ind}"
               f"?format=json&per_page=20000&date=1960:2025")
        for attempt in range(3):
            try:
                js = requests.get(url, headers=UA, timeout=120).json()
                rows = js[1] if len(js) > 1 and js[1] else []
                break
            except Exception as e:  # noqa
                log(f"wdi retry {ind}: {e}")
                time.sleep(3)
                rows = []
        for r in rows:
            frames.append(dict(iso3=r["countryiso3code"], year=int(r["date"]),
                               indicator=ind, variable=name, value=r["value"]))
        log(f"wdi     {ind} -> {len(rows)} rows")
        time.sleep(0.5)
    pd.DataFrame(frames).to_csv(out, index=False)
    log(f"saved   wdi.csv")


# ----------------------------------------------------------------------------
def cepal_dims(ind):
    js = requests.get(f"https://api-cepalstat.cepal.org/cepalstat/api/v1/indicator/{ind}/dimensions"
                      f"?lang=en&format=json", headers=UA, timeout=120).json()
    body = js.get("body", js)
    dims = body.get("dimensions", body)
    return {d["id"]: (d["name"], {m["id"]: m["name"] for m in d.get("members", [])}) for d in dims}


def get_cepal():
    for ind, name in CEPAL.items():
        out = RAW / f"{name}_{ind}.csv"
        if out.exists():
            log(f"keep    {out.name}")
            continue
        try:
            meta_js = requests.get(f"https://api-cepalstat.cepal.org/cepalstat/api/v1/indicator/{ind}/data"
                                   f"?lang=en&format=json", headers=UA, timeout=300).json()
            body = meta_js.get("body", meta_js)
            data = body["data"]
            dims = cepal_dims(ind)
            rows = []
            for r in data:
                row = dict(iso3=r.get("iso3"), value=r.get("value"), indicator_id=ind)
                for k, v in r.items():
                    if k.startswith("dim_"):
                        did = int(k[4:])
                        dname, members = dims.get(did, (k, {}))
                        row[dname] = members.get(v, v)
                rows.append(row)
            df = pd.DataFrame(rows)
            meta = body.get("metadata", {})
            (RAW / f"{name}_{ind}_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
            df.to_csv(out, index=False)
            log(f"saved   {out.name} {df.shape}")
        except Exception as e:  # noqa
            log(f"cepal   FAILED {ind}: {e}")


# ----------------------------------------------------------------------------
def ilo_pull(df_id, iso, start=1990):
    base = f"https://sdmx.ilo.org/rest/data/ILO,{df_id},1.0/"
    for dots in (".....", "....", "...", "..", "."):
        url = f"{base}{iso}{dots}?format=csv&startPeriod={start}"
        r = requests.get(url, headers={**UA, "Accept": "text/csv"}, timeout=180)
        if r.status_code == 200 and r.text.strip():
            return pd.read_csv(io.StringIO(r.text))
        if r.status_code == 404:
            return pd.DataFrame()
    return pd.DataFrame()


def get_ilo():
    countries = list(LATAM) + list(DONORS) + list(REFERENCE)
    for df_id, name in ILO.items():
        out = RAW / f"{name}.csv"
        if out.exists():
            log(f"keep    {out.name}")
            continue
        frames = []
        for iso in countries:
            try:
                d = ilo_pull(df_id, iso, start=1990)
                if len(d):
                    frames.append(d)
                time.sleep(0.3)
            except Exception as e:  # noqa
                log(f"ilo     {df_id} {iso} failed: {e}")
        if frames:
            pd.concat(frames).to_csv(out, index=False)
            log(f"saved   {out.name} ({sum(len(f) for f in frames)} rows)")
        else:
            log(f"ilo     {df_id}: no data")


# ----------------------------------------------------------------------------
def comtrade_year(codes, year):
    """One call per year for a whole reporter group (the preview API allows one
    period per call but many reporters; the aggregate filters keep rows <= 500)."""
    url = ("https://comtradeapi.un.org/public/v1/preview/C/A/HS"
           f"?reporterCode={','.join(str(c) for c in codes.values())}&period={year}&cmdCode=847950&flowCode=M"
           "&partnerCode=0&partner2Code=0&motCode=0&customsCode=C00")
    for attempt in range(8):
        try:
            js = requests.get(url, headers=UA, timeout=90).json()
        except Exception:  # noqa
            time.sleep(3)
            continue
        err = js.get("error") or js.get("message")
        if err:
            if "Rate limit" in str(err):
                time.sleep(2 + attempt)
                continue
            return None, str(err)
        data = js.get("data", [])
        agg = [r for r in data if r.get("partnerCode") == 0 and r.get("partner2Code") == 0
               and r.get("motCode") == 0 and r.get("customsCode") == "C00"]
        by = {}
        for r in agg:
            m = r["reporterCode"]
            b = by.setdefault(m, dict(value_usd=0.0, net_kg=0.0, qty=0.0, n_rows=0))
            b["value_usd"] += r.get("primaryValue") or 0
            b["net_kg"] += r.get("netWgt") or 0
            b["qty"] += r.get("qty") or 0
            b["n_rows"] += 1
        return by, None
    return None, "gave up"


def get_comtrade():
    out = RAW / "comtrade_847950_imports.csv"
    if out.exists():
        log("keep    comtrade_847950_imports.csv")
        return
    groups = [("latam", M49), ("donor", DONORS), ("reference", REFERENCE)]
    rows = []
    for gname, codes in groups:
        inv = {v: k for k, v in codes.items()}
        for y in COMTRADE_YEARS:
            by, err = comtrade_year(codes, y)
            if err:
                log(f"comtrade {gname} {y}: {err}")
                continue
            for iso, m in codes.items():
                rec = by.get(m, dict(value_usd=None, net_kg=None, qty=None, n_rows=0))
                rows.append(dict(iso3=iso, m49=m, year=y, group=gname, **rec))
            time.sleep(1.2)
        log(f"comtrade group {gname} done ({len(codes)} reporters x {len(COMTRADE_YEARS)} years)")
    pd.DataFrame(rows).to_csv(out, index=False)
    log(f"saved   comtrade_847950_imports.csv ({len(rows)} rows)")


if __name__ == "__main__":
    log("=== download run start ===")
    for step in (get_vdem, get_ert, get_pwt, get_pts, get_fh, get_pinksheet, get_govrev, get_mmp, get_swiid,
                 get_wdi, get_cepal, get_ilo, get_comtrade):
        try:
            step()
        except Exception as e:  # noqa
            log(f"FAILED {step.__name__}: {e!r}")
    log("=== download run end ===")
