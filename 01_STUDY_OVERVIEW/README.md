# Public Analysis Scope

This release reports a validated analysis set of 14 public biomedical binary-classification datasets and three prespecified model families.

The reference condition is fold-contained repeated nested cross-validation. The four primary comparisons evaluate non-nested tuning, globally fitted preprocessing, globally fitted supervised feature selection, and global oversampling. These contrasts describe implemented pipelines. They do not isolate the causal effect of validation-error placement. Predictor-signature grouping does not establish patient identity or guarantee the absence of clinical target leakage.

The primary estimand is the paired within-dataset difference in AUROC relative to the reference. Secondary outcomes and analyses include AUPRC, balanced accuracy, F1 score, Brier score, calibration, sample-size sensitivity, feature-to-sample ratio, minority prevalence, model family, robustness checks, and mixed-effects diagnostics.

The mixed-effects model converged but failed residual-normality and equal-variance diagnostics. Its coefficient and likelihood-ratio results are therefore retained as sensitivity evidence. Final H3-H6 decisions use the prespecified matched or stratified fallback analyses.

The archived protocol and locked analysis plan are available at https://doi.org/10.5281/zenodo.22735962.

`dataset_source_versions.csv` lists the 14 included datasets using public display labels, their source identifiers and DOIs, acquisition dates, source URLs, and SHA-256 hashes of the acquired archives and official-page snapshots. It identifies the source versions used for this benchmark; it does not redistribute the original datasets.
