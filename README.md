# Validation Leakage and Performance Inflation in Biomedical Machine Learning

This repository contains the public reproducibility record for a prespecified multi-dataset benchmark of validation design and performance inflation in biomedical binary classification.

The repository is organised in the same order as the study workflow: study definition, executable research code, validated results, statistical analyses, final figures, and integrity records. Submission documents, local configuration, private working files, document builders, and figure-rendering scripts are not included.

## Study summary

- 14 included public biomedical datasets
- 3 model families: logistic regression, random forest, and gradient-boosted trees
- 42 complete dataset-model packages
- 2,946 validated atomic result units
- Leakage-free repeated nested cross-validation as the reference
- Four primary comparisons: non-nested tuning, global preprocessing, global supervised feature selection, and global oversampling

The locked protocol and analysis plan are archived at [Zenodo](https://doi.org/10.5281/zenodo.22735962).

## Repository structure

### `01_STUDY_OVERVIEW/`

Public analysis scope, sequential dataset display map, and dataset-level characteristics used in the reported analyses. This folder defines what was analysed before a reader moves to the implementation.

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

Primary contrasts, model-family analyses, sample-size analyses, moderator analyses, mixed-effects diagnostics, sensitivity analyses, hypothesis decisions, and reviewer-readable Excel workbooks.

### `05_FINAL_FIGURES/`

Five main-text figures and three supplementary figures at 400 DPI. These are the final reported visual outputs. Figure-generation code is intentionally excluded from the public release.

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
7. Compare regenerated outputs with `03_VALIDATED_RESULTS/`, `04_STATISTICAL_ANALYSES/`, and the SHA-256 records in `06_REPRODUCIBILITY_RECORDS/`.

Detailed script order and input-output relationships are provided in `02_REPRODUCIBILITY_CODE/README.md`.

## Main findings represented in this release

Global oversampling produced the clearest positive performance inflation relative to the leakage-free reference. The other primary mechanisms showed smaller or inconsistent effects. Sample size and feature-to-sample ratio did not show consistent moderation. Minority prevalence and model family showed descriptive patterns, but the corresponding mixed-effects evidence is treated as sensitivity evidence because the diagnostic gate failed.

See `04_STATISTICAL_ANALYSES/03_INTERPRETATION/FINAL_HYPOTHESIS_REPORT.md` for the complete hypothesis decisions and diagnostic qualifications.

## Data and reporting boundaries

- Public source datasets are not redistributed when their source terms do not permit redistribution.
- Only the included validated analysis set is represented in public results.
- Public tables and figures use sequential display labels.
- Raw result files are frozen and hash-validated.
- Submission documents and author-only working records are maintained separately from this repository.

## Integrity checks

The release inventory records the relative path, size, and SHA-256 hash of every public file. The validated result archive contains no missing dataset-model package in the reported analysis set. Statistical conclusions distinguish confirmatory evidence from secondary or diagnostic-dependent sensitivity evidence.

## Citation

Please cite the final article when available. Until then, cite the archived protocol using its Zenodo DOI.
