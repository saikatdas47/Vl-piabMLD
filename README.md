# Validation Leakage and Performance Inflation in Biomedical Machine Learning

This repository contains the public reproducibility record for a prespecified multi-dataset benchmark of validation design and performance inflation in biomedical binary classification.

The repository is organised in the same order as the study workflow: study definition, executable research code, validated results, statistical analyses, final figures, and integrity records. Submission documents, local configuration, private working files, document builders, and figure-rendering scripts are not included.

## Study summary

- 14 included public biomedical datasets
- 3 model families: logistic regression, random forest, and gradient-boosted trees
- 42 complete dataset-model packages
- 2,946 validated atomic result units
- Fold-contained repeated nested cross-validation as the reference
- Four primary comparisons: non-nested tuning, global preprocessing, global supervised feature selection, and global oversampling

The locked protocol and analysis plan are archived at [Zenodo](https://doi.org/10.5281/zenodo.22735962).

## Repository structure

### `01_STUDY_OVERVIEW/`

Public analysis scope, sequential dataset display map, dataset-level characteristics, and `dataset_source_versions.csv` with source identifiers, access dates, and source-file hashes. This folder defines what was analysed before a reader moves to the implementation.

### `02_REPRODUCIBILITY_CODE/`

Core research code arranged chronologically:

1. data and split preparation;
2. model and validation-condition implementation;
3. result collection and integrity checking;
4. primary and secondary statistical analysis.

The folder contains research and analysis code only. It does not contain manuscript-generation code, Word-document builders, workbook-formatting utilities, or figure-rendering scripts.

### `03_VALIDATED_RESULTS/`

The frozen atomic result archive for the included analysis set, its validated manifest, the completion matrix, and result-quality audit outputs. These files are read-only evidence and should not be edited directly.

### `04_STATISTICAL_ANALYSES/`

Primary contrasts, model-family analyses, sample-size analyses, moderator analyses, mixed-effects diagnostics, sensitivity analyses, hypothesis decisions, and reviewer-readable Excel workbooks. `01_PRIMARY_ANALYSIS/sample_size_summary.csv` counts independent datasets after averaging model-specific effects within each dataset; the 50%, 75%, and 100% denominators are 7, 13, and 14. The full-fold minority counts are provided in `02_SECONDARY_AND_SENSITIVITY/fold_class_counts.csv`.

### `05_FINAL_FIGURES/`

Six main-text TIFF figures and three supplementary TIFF figures match the current submission package. The editable draw.io source for Fig 1 is included. The [figure map](05_FINAL_FIGURES/README.md) records the role of each image and the supplementary renumbering. Superseded figures remain available in Git history. Figure-rendering scripts are not included.

### `06_REPRODUCIBILITY_RECORDS/`

Release manifest, file inventory, SHA-256 hashes, software requirements, code-audit record, and public integrity summary.

## Chronological reproduction workflow

The exact commands depend on where a reader stores the independently acquired public datasets. The public workflow is:

1. Acquire the source datasets under their original terms and verify their versions.
2. Prepare the analysis tables and frozen splits using `02_REPRODUCIBILITY_CODE/01_DATA_AND_SPLITS/`.
3. Run the registered model families and validation conditions using `02_REPRODUCIBILITY_CODE/02_MODEL_AND_VALIDATION/`.
4. Collect and audit atomic outputs using `02_REPRODUCIBILITY_CODE/03_RESULT_COLLECTION/`.
5. Recreate the primary tables using `02_REPRODUCIBILITY_CODE/04_STATISTICAL_ANALYSIS/build_final_analysis.py`.
6. Prepare H3-H6 analysis data, fit the registered mixed-effects sensitivity model, and apply the prespecified fallback analyses using the remaining scripts in `04_STATISTICAL_ANALYSIS/`.
7. Compare regenerated outputs with `03_VALIDATED_RESULTS/`, `04_STATISTICAL_ANALYSES/`, and the SHA-256 records in `06_REPRODUCIBILITY_RECORDS/`. The model-cell denominator in an earlier derived sample-size summary was corrected at the public reporting layer; the frozen atomic results were not changed.

Detailed script order and input-output relationships are provided in `02_REPRODUCIBILITY_CODE/README.md`.

## Main findings represented in this release

Global oversampling showed the clearest evidence of higher internal AUROC estimates relative to the fold-contained reference. The other primary conditions showed smaller or inconsistent differences. Sample size and feature-to-sample ratio did not show consistent moderation. Minority prevalence and model family showed descriptive patterns. The mixed-effects model failed its diagnostic gate.

The contrasts describe implemented pipeline differences, not isolated causal effects of validation-error placement. C4 also changed representation, feature count, and tuning space. Condition-specific search seeds did not enforce identical candidates. C2 used pooled out-of-fold AUROC for reporting rather than the reference mean of outer-fold AUROCs. Predictor-signature grouping does not establish patient identity or guarantee the absence of clinical target leakage.

See `04_STATISTICAL_ANALYSES/03_INTERPRETATION/FINAL_HYPOTHESIS_REPORT.md` for the complete hypothesis decisions and diagnostic qualifications.

## Data and reporting boundaries

- Public source datasets are not redistributed when their source terms do not permit redistribution.
- Only the included validated analysis set is represented in public results.
- Public tables and figures use sequential display labels.
- Raw result files are frozen and hash-validated.
- The source-version registry and fold-count table are descriptive provenance/support records, not additional analysis datasets or confirmatory tests.
- Submission documents and author-only working records are maintained separately from this repository.

## Integrity checks

The release inventory records the relative path, size, and SHA-256 hash of every public file. The validated result archive contains no missing dataset-model package in the reported analysis set. Statistical conclusions distinguish confirmatory evidence from secondary or diagnostic-dependent sensitivity evidence.

## Citation

Please cite the final article when available. Until then, cite the archived protocol using its Zenodo DOI.
