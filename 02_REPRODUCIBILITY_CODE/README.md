# Reproducibility Code Workflow

The folders below follow the chronological study workflow. Paths are relative to this directory.

## 1. Data and split preparation

- `01_DATA_AND_SPLITS/freeze_splits.py`: creates frozen analysis splits after dataset eligibility and grouping checks.
- `01_DATA_AND_SPLITS/prepare_preexecution_registries.py`: creates the registered execution identities used by the benchmark.

Public datasets must be acquired independently under their original source terms. Private acquisition records and local paths are not included here.

## 2. Model and validation implementation

- `02_MODEL_AND_VALIDATION/benchmark_pipeline.py`: implements preprocessing, feature selection, resampling, tuning, prediction, and metrics.
- `02_MODEL_AND_VALIDATION/experiment_system.py`: defines the registered experimental conditions and result contract.
- `02_MODEL_AND_VALIDATION/shared_dataset_runner.py`: executes the common dataset-model workflow.

## 3. Result collection and integrity

- `03_RESULT_COLLECTION/update_master_tracker.py`: updates completion status from atomic outputs.
- `03_RESULT_COLLECTION/audit_completed_package.py`: validates a completed dataset-model package.
- `03_RESULT_COLLECTION/aggregate_scientific_results.py`: combines validated atomic outputs for downstream analysis.

## 4. Statistical analysis

- `04_STATISTICAL_ANALYSIS/build_final_analysis.py`: freezes the included result set and creates the primary paired analyses.
- `04_STATISTICAL_ANALYSIS/prepare_secondary_analysis.py`: constructs the H3-H6 analysis cells and matched fallback summaries.
- `04_STATISTICAL_ANALYSIS/run_secondary_mixed_effects.R`: fits the registered mixed-effects sensitivity model and records diagnostics.
- `04_STATISTICAL_ANALYSIS/finalize_secondary_analysis.py`: creates the final H3-H6 summaries, paired fallbacks, and code-audit outputs.

## Recommended execution order

```text
freeze_splits.py
  -> prepare_preexecution_registries.py
  -> shared_dataset_runner.py
  -> update_master_tracker.py
  -> audit_completed_package.py
  -> aggregate_scientific_results.py
  -> build_final_analysis.py
  -> prepare_secondary_analysis.py
  -> run_secondary_mixed_effects.R
  -> finalize_secondary_analysis.py
```

The frozen public outputs in `../03_VALIDATED_RESULTS/` and `../04_STATISTICAL_ANALYSES/` are the reference outputs for verification.

## Public analysis inputs

The final-analysis scripts use only paths included in this release:

- dataset map: `../01_STUDY_OVERVIEW/dataset_display_map.csv`
- primary execution registry: `../01_STUDY_OVERVIEW/registered_execution_plan/primary_job_registry.csv`
- sample-size execution registry: `../01_STUDY_OVERVIEW/registered_execution_plan/sample_size_job_registry.csv`
- dataset characteristics: `../01_STUDY_OVERVIEW/dataset_characteristics.csv`
- validated atomic results: `../03_VALIDATED_RESULTS/01_ATOMIC_RESULTS/`

Use `python <script> --help` for the exact arguments. Run the statistical scripts in the sequence listed above. The R step requires the packages recorded in `requirements_analysis.txt` and preserves the diagnostic outputs whether or not the diagnostic gate passes.
