# Academic screening and multidimensional development

Code and aggregate results for **Narrow academic screening misses most students with high multidimensional development**.

## Version

Artifact v1.0.0, prepared 2026-10-03. The archived tables contain the estimates reported in the manuscript. The analysis evaluates a top-decile Pareto-depth target across curiosity, creativity, sociability, and well-being; it does not identify individual students for admissions or other decisions.

## Requirements and reproduction

Python 3.12 or later. Install the packages in requirements.txt into a separate environment.

```text
python -m pip install -r requirements.txt
python tests/test_analysis.py
python visualization/plot_figures.py
python run_analysis.py --data-2019 /path/to/INT_01_ST_2021.04.14_Public.dta --data-2023 "/path/to/SSES R2 ST PUF INTERNATIONAL.dta" --workers 4
```

The first two reproduction commands test ranking and survey variance calculations and reproduce the five manuscript figures from the included aggregate tables. Full analysis requires the original OECD files, checks their SHA-256 values against analysis_config.json, computes site-wide and cohort estimates using official weights, and writes new results under reproduced_results. It preserves the archived result tables. Large Pareto-depth calculations and replicate estimation can take substantial time.

## Data access

Obtain the 2019 and 2023 SSES student public-use files from the OECD Survey on Social and Emotional Skills database and comply with the terms supplied with the releases. Raw files and student-level derivatives are not redistributed. The included CSVs contain site-level, cohort-level, survey-replicate, or cross-site aggregate statistics. The filename prefixes identify the survey wave and analytical scope.

## Statistical design

Each wave is analysed separately. Achievement is the mean of the mathematics and reading/language grades. Developmental ranks use noncompensatory Pareto dominance and survey-weighted fractional ranks. Symmetric fractional inclusion handles cutoff ties. The 2019 variance multiplier is 0.5 with 76 JRR replicates; the 2023 multiplier is 0.05 with 80 Fay-BRR replicates. Cross-wave differences combine independent variances. Equal-site summaries describe the included sites rather than a global student-weighted population. Correlations use Fisher-z REML synthesis.

## Artifact layout

- sses2023_alignment/src: ranking, data preparation, site estimation, and cohort calculations used for both waves.
- run4_crosswave_replication/src: aggregation routines for the reported cross-wave statistics.
- The exp directories: frozen aggregate inputs and reported estimates.
- visualization: reproducible manuscript plots.
- run_analysis.py: portable public-data entry point; original local paths are supplied through command-line options.

MANIFEST.sha256 records the exact release files. The archive excludes manuscript sources, PDFs, submission documents, review notes, credentials, caches, and student-level records.

## Authors

Rongwen Hong; Yu Ke; Yueqian Ke; Zhifeng Yang; Zhigao Huang. Rongwen Hong and Yu Ke share first authorship. Correspondence: zghuang@qztc.edu.cn.

## Licenses

Python code is licensed under MIT. Aggregate CSV results and documentation are licensed under CC BY 4.0. OECD source files remain subject to OECD terms and are not redistributed.
