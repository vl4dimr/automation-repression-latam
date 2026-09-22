# Restoring excluded provider inputs

The archive includes the constructed clean panel and numerical outputs, but not raw provider data. `02_data/SOURCES.md` and the download script document provider URLs and versions. `ARCHIVED_INPUTS.json` records SHA-256 digests of the local raw files available when the release was packaged; exact reproduction requires matching snapshots. A failed provider download must not be replaced with generated observations.

## Extra inputs for complete estimation from the supplied panel

Place these files under `02_data/raw/`:

| File | Used by | Source and required content |
|---|---|---|
| `comtrade_847950_imports.csv` | `08_measurement_robustness.py` | Archived UN Comtrade industrial-robot import observations, HS 847950; restore the original CSV downloaded through the public preview endpoint. |
| `wdi.csv` | `08_measurement_robustness.py` | Archived WDI API pull, including the deflator inputs used to reconstruct the robot proxy. |
| `swiid9_92.zip` | `08_measurement_robustness.py` | Official SWIID **9.92** full distribution with `swiid9_92/swiid9_92.dta`, including all 100 released imputations. |
| `swiid9_92_summary.csv` | `08_measurement_robustness.py` | The archived summary CSV extracted from that official distribution. |
| `ilo_emp_eco_modelled.csv` | `09_identification_diagnostics.py` | Archived ILO modelled employment by economic activity data, with industry categories used by the sector diagnostic. |

SWIID source: [dataset DOI](https://doi.org/10.7910/DVN/LM4OWF), [archived datafile 13657070](https://dataverse.harvard.edu/api/access/datafile/13657070). The downloader's `get_swiid()` saves that ZIP and extracts its summary CSV. The 100 imputation columns are **not** recoverable from the summary CSV or clean panel. Use the intact official ZIP; do not simulate replacements from reported summary standard errors. The release documents differences between released draw means and summary point values without assuming they are interchangeable.

Run `python run_analysis.py --check-inputs` to check file presence. Compare hashes to `ARCHIVED_INPUTS.json` when exact archived files are obtained; presence alone does not certify version identity. Then `python run_analysis.py` reruns every estimation step from the existing clean panel. The analytical reports under `05_red_team/` will be generated anew by the diagnostic scripts.

## Full raw-to-panel rebuild

`python run_all.py --analysis-only` runs provider downloads, DAG, panel construction, descriptives and all estimation scripts. `python run_all.py --no-download --analysis-only` requires the archived raw files listed in the source notes and in the builder, including V-Dem, ERT, PWT, WDI, ILO labour share/employment/occupations/sectors, CEPAL fiscal data, SWIID summaries, PTS, Freedom House, Mass Mobilization Project, Comtrade and the World Bank Pink Sheet. The current downloader stores additional files it may not use in the final panel.

V-Dem repository branches and public APIs can change. Some sources require academic/noncommercial use, have provider-specific terms or change download URLs. Restore the stated vintage and retain each provider's notices. Upstream access restrictions and the absence of a stable historical API snapshot can limit an exact fresh download; the included clean panel fixes the analysis input used for this release.
