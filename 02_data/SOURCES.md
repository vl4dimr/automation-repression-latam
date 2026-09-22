# Data sources

All files were downloaded by `02_data/scripts/01_download.py` on 2026-09-07 (UTC timestamps, byte sizes and URLs in `raw/download_log.txt`). Licenses are those stated by each provider on the access date.

| Source | What we use | Version | URL | License | Notes |
|---|---|---|---|---|---|
| V-Dem Country-Year, Full+Others | repression, mobilization, regime, coups (Powell & Thyne series) | v16 (March 2026), as bundled in the `vdemdata` R package | https://github.com/vdeminstitute/vdemdata (data/vdem.RData) | CC BY-SA 4.0 | Powell & Thyne's own site returned HTTP 403; V-Dem's `e_pt_coup` / `e_pt_coup_attempts` carry the same series to 2024 |
| Episodes of Regime Transformation | autocratization / democratization episodes | v16 | https://github.com/vdeminstitute/ERT (inst/ert.csv) | CC BY-SA 4.0 | |
| Penn World Table | capital stock (rnna, cn), employment, labour share, GDP | 10.01 | https://dataverse.nl/api/access/datafile/354098 | CC BY 4.0 | 1950–2019; no Cuba; no labour share for El Salvador; labour share unchanged in approximately 63% of observed current/prior pairs, 1960–2019 (repetition alone does not establish imputation) |
| UN Comtrade | imports of industrial robots, HS 847950, value USD, net weight | public preview API, pull 2026-09-07 | https://comtradeapi.un.org/public/v1/preview/C/A/HS | UN Comtrade terms of use | one period per call; aggregate rows (partner world, mode all, customs C00) |
| ILOSTAT | labour income share (modelled), employment (modelled), employment by ISCO major group, employment by ISIC section (modelled) | SDMX REST, pull 2026-09-07 | https://sdmx.ilo.org/rest/data/ILO,DF_*,1.0/ | ILO open data | dataflows LAP_2GDP_NOC_RT, EMP_2EMP_SEX_AGE_NB, EMP_TEMP_SEX_OCU_NB, EMP_2EMP_SEX_ECO_NB |
| SWIID | Gini market and disposable, absolute and relative redistribution | 9.92 (April 2026) | https://doi.org/10.7910/DVN/LM4OWF (file 13657070) | CC0 | summary CSV and 100 paired official imputations from the archived release ZIP; their means are not identical to the summary representation |
| CEPALSTAT | social expenditure by COFOG function (% GDP), tax revenue by type (% GDP), SOCX | API v1, pull 2026-09-07 | https://api-cepalstat.cepal.org/cepalstat/api/v1/indicator/{3126,3127,3128,821,4405}/data | ECLAC open data | |
| Financing the State (Andersson & Brambor) | central government tax revenue % GDP 1800–2024 | v3 (2026) | https://www.perfandersson.com/uploads/1/2/7/5/127505228/govrev_v3.csv ; 1800–2012 version doi:10.7910/DVN/P1VAGP | authors' terms (academic use) | 11 of 18 countries |
| World Development Indicators | GDP, population, urbanization, trade, resource rents, tax revenue, GFCF, manufacturing VA, military, exports by product, inflation, unemployment | API v2, pull 2026-09-07 | https://api.worldbank.org/v2/country/{codes}/indicator/{code} | CC BY 4.0 | |
| World Bank Commodity Price Data (Pink Sheet) | real price indices: energy, metals and minerals, agriculture | September 2026 | https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Annual.xlsx | CC BY 4.0 | link hash rotates monthly |
| Political Terror Scale | PTS-A, PTS-H, PTS-S | PTS-2025 (covers 1976–2024) | http://www.politicalterrorscale.org/Data/Files/PTS-2025.xlsx | free for academic use | |
| Freedom House, Freedom in the World | civil liberties and political rights ratings | 2024 edition file (1972–2023) | https://freedomhouse.org/sites/default/files/2024-02/Country_and_Territory_Ratings_and_Statuses_FIW_1973-2024.xlsx | free for non-commercial use | 2025/2026 files not at a stable URL on access date |
| Mass Mobilization Project | protest events, participants, state responses | v5, file mmALL_073120 (1990–2020) | https://doi.org/10.7910/DVN/HTTWYL (file 4291456) | CC0 | |

## Not available or not used

- **IFR World Robotics** (robot stock by industry): paid; https://ifr.org/free-downloads/ links to a commercial shop. Replaced by Comtrade HS 847950 import values.
- **ACLED**: requires registration and an API key; V-Dem mobilization and MMP are used instead.
- **Latinobarómetro**: registration required; not used.
- **WID**: bulk file of 880 MB; SWIID covers pre/post-fiscal inequality.
- **Powell & Thyne direct file**: HTTP 403 from uky.edu on access date; same data via V-Dem.
- **ENAHO (Peru)**: deferred to a separate paper.

## Sample

18 countries: ARG, BOL, BRA, CHL, COL, CRI, DOM, ECU, SLV, GTM, HND, MEX, NIC, PAN, PRY, PER, URY, VEN. Cuba is absent from PWT; Haiti was excluded for coverage. Baseline window 1990–2024; coups analysed 1960–2024.
