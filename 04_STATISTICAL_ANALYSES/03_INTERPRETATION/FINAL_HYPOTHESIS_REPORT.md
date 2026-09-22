# Final hypothesis analysis report

Analysis date: 2026-09-22

## Verification outcome

- Canonical result archive: complete and hash-validated.
- Secondary analysis cells: 408 across 14 datasets, three model families, four primary comparison conditions, and all estimable registered fractions.
- Mixed-effects model: converged, but residual normality and equal-variance diagnostics failed.
- Reporting decision: use the prespecified stratified and matched fallback estimates for scientific conclusions; retain mixed-model results as sensitivity evidence.

## Final hypothesis decisions

- **H1 — PARTIALLY_SUPPORTED**: Global oversampling showed positive multiplicity-adjusted inflation; global preprocessing and feature selection did not.
- **H2 — DIRECTIONALLY_CONSISTENT_NOT_CONFIRMED**: The median contrast was positive, but the Holm-adjusted test was not significant.
- **H3 — NOT_SUPPORTED**: The moderator likelihood-ratio test was not significant and matched fractions did not show a consistent increase as sample size decreased.
- **H4 — NOT_SUPPORTED**: The feature-to-sample moderator likelihood-ratio test was not significant and stratified associations were inconsistent.
- **H5 — DESCRIPTIVE_SUPPORT_NOT_CONFIRMATORY**: Inflation varied with minority prevalence most clearly for global oversampling, but mixed-model residual and variance diagnostics failed.
- **H6 — DESCRIPTIVE_SUPPORT_NOT_CONFIRMATORY**: Flexible model families showed larger oversampling inflation, but mixed-model residual and variance diagnostics failed.

## Secondary model diagnostics

- Condition number: 19.34.
- Residual normality p-value: 1.7e-21.
- Equal-variance diagnostic p-value: 8.01e-72.
- Leave-one-dataset-out failures: 0.

## Moderator tests retained as sensitivity evidence

- H3 interaction LRT p=0.2248.
- H4 interaction LRT p=0.2201.
- H5 interaction LRT p=5.506e-34.
- H6 interaction LRT p=9.191e-09.

These p-values are not used alone to claim confirmation because the model diagnostics failed. The manuscript must distinguish confirmatory H1-H2 conclusions from secondary/fallback H3-H6 evidence.
