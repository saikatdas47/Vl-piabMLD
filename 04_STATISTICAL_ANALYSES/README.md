# Statistical analysis outputs

The folders follow the reporting sequence.

1. `01_PRIMARY_ANALYSIS/` contains dataset-level estimates, paired contrasts, multiplicity-adjusted tests, sample-size summaries, and model-family results.
2. `02_SECONDARY_AND_SENSITIVITY/` contains H3-H6 moderator analyses, mixed-effects outputs, diagnostics, matched fallback analyses, robustness checks, and the code audit.
3. `03_INTERPRETATION/` contains the integrated audit and final hypothesis decisions. Diagnostic limitations are retained rather than hidden.
4. `04_REVIEWER_WORKBOOKS/` contains reviewer-readable Excel versions of the master results and complete hypothesis analysis.

CSV files are the canonical machine-readable outputs. Excel files are convenience copies for inspection.

`01_PRIMARY_ANALYSIS/sample_size_summary.csv` reports independent dataset counts and medians after model-specific effects are averaged within each dataset. The denominators are 7, 13, and 14 at 50%, 75%, and 100%, respectively. This public derived table corrects a previous model-cell denominator; frozen atomic results remain unchanged. Its four-condition medians agree with the separately generated H3 table in `02_SECONDARY_AND_SENSITIVITY/h3_corrected_sample_size_summary.csv` to floating-point precision. The H3 bootstrap intervals in that separate table use a different order of random-number draws from the primary analysis, so same-fraction interval endpoints need not be identical.

`02_SECONDARY_AND_SENSITIVITY/fold_class_counts.csv` provides the minority-class counts for each of the 84 outer assessment folds and their reproduced inner validation/training folds. The outer assignments were reconstructed exactly from the frozen registry before inner counts were derived; the table is descriptive support for the three-fold design, not a new performance analysis.
