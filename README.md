# Automation and Repression: Panel Evidence from Latin America

Data and code for the article of the same title, under review at World Development.

## Authors

- Vladimiro Ibáñez Quispe, FINESI, Universidad Nacional del Altiplano, Puno, Peru (ORCID [0000-0002-0277-4945](https://orcid.org/0000-0002-0277-4945))
- Milton Vladimir Mamani Calisaya, FINESI, Universidad Nacional del Altiplano, Puno, Peru (ORCID [0000-0002-0676-0989](https://orcid.org/0000-0002-0676-0989))
- Yasbel Mercedes Quispe Alata, Ministerio de Educación, Puno, Peru (ORCID [0009-0007-3381-0551](https://orcid.org/0009-0007-3381-0551))
- Fred Torres-Cruz, Universidad Nacional del Altiplano, Puno, Peru (ORCID [0000-0003-0834-6834](https://orcid.org/0000-0003-0834-6834))

Corresponding author: Milton Vladimir Mamani Calisaya, Facultad de Ingeniería Estadística e Informática (FINESI), Universidad Nacional del Altiplano, Avenida Floral 1153, Puno, Peru.
This archive supports the empirical analysis of capital accumulation, automation proxies and repression in an 18-country Latin American panel. It contains country-level analysis data, a variable dictionary, source-access instructions, scripts and the final numerical outputs. Manuscript files, correspondence, raw provider archives, local tool installers and internal audit files are excluded.

## Environment

The archived calculation environment used Python 3.14.0. Exact package versions are in `requirements.txt`; use an isolated environment with compatible wheels or build tools.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On Linux/macOS, activate with `source .venv/bin/activate`. Run commands from the extracted archive root. The random seed is fixed in `02_data/scripts/config.py` (`20260907`). Estimation writes tables/figures over the supplied output copies; retain the original ZIP for comparison.

Re-running the pipeline reproduces every table byte for byte with one exception. `03_results/tables/cate_country_year.csv` holds the per-observation forest predictions, which are summed in parallel, so their last digits vary between runs at the scale of machine precision (observed maximum absolute difference 1.6e-15). The reported forest statistics are unaffected at the precision printed in the manuscript. Compare that file numerically rather than by hash.

## Replay from the clean panel

```powershell
python run_analysis.py --core-only --check-inputs
python run_analysis.py --core-only
```

This reruns descriptives and estimation scripts 01–07 from `02_data/clean/panel.parquet`: fixed effects/IV, threshold models, exploratory forests, coup/event profiles, baseline robustness, power and interaction marginal effects. CSV and Parquet versions of the clean panel and `dictionary.csv` are supplied. The CSV is a transparent exchange format; scripts read Parquet.

The default complete analysis replay also reruns measurement and identification diagnostics. Obtain the five additional archived source inputs listed in `RAW_INPUTS.md`, then run:

```powershell
python run_analysis.py --check-inputs
python run_analysis.py
```

The runner stops before estimating if those inputs are missing. It does not silently replace released SWIID imputations with simulated data. The ZIP contains the final tables for all analyses, including diagnostics requiring these additional files; `--core-only` leaves those diagnostic tables as supplied rather than reproducing them.

## Rebuild the panel and every analysis

```powershell
python run_all.py --analysis-only
```

This invokes the download/build pipeline as well as estimation. If every required raw input has already been restored from the correct archived versions, use `python run_all.py --no-download --analysis-only`. The default downloader can depend on changing provider URLs, API coverage and access conditions; `ARCHIVED_INPUTS.json` records hashes for the local source files used in the release. Fetching a newer response is not necessarily an exact reproduction of an older API snapshot.

## Interpretations and limits

Estimates are conditional associations. Capital is not a direct measure of automation; robot import values depend on coverage, prices and stock assumptions. Corrected threshold nonrejection does not disprove the model. Sparse political events, imperfect actor classification, limited country clusters and weak or contestable instruments constrain identification. V-Dem endpoint substitutions are not full posterior integration. SWIID analysis pools 100 released imputations and separately documents their differences from archived summary values. The methods and table notes identify these distinctions.

Claude Code and OpenAI Codex assisted with drafting and research programming, debugging and numerical checks. Estimates are computed by the included statistical scripts from documented source inputs; human responsibility for interpretation and source verification remains unchanged.

## Files and rights

- `02_data/SOURCES.md`: source versions, URLs and attribution recorded by the project.
- `RAW_INPUTS.md`: restoring source files, especially SWIID 9.92's complete imputation archive.
- `DATA_RIGHTS.md`: limitations on interpreting source-license notes and redistributing combined data.
- `MANIFEST.json`: size and SHA-256 digest of each included file, excluding the manifest itself.
- `ARCHIVED_INPUTS.json`: hashes and sizes of excluded local raw inputs; no raw bytes are included.

Code is released under the MIT licence (see LICENSE). Derived analysis data remain subject to the upstream provider terms recorded in DATA_RIGHTS.md and 02_data/SOURCES.md; no relicensing of provider data is implied.

## Cite

The data and code are archived on Zenodo: [10.5281/zenodo.22915358](https://doi.org/10.5281/zenodo.22915358).
Machine-readable metadata is in `CITATION.cff`.
