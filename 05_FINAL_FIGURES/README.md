# Figure map

This set follows the current manuscript and supplementary numbering. It contains six main figures and five supplementary figures. These are PNG copies of the current author-maintained figures. Pixels and resolution metadata have not changed. Nominal resolution is 300 or 400 DPI, as recorded per image in the release manifest. Repository PNGs do not replace the separate journal submission files.

## Main text

1. `01_MAIN_TEXT/Figure_1_Methodology_Flowchart.png` — study workflow and deliberate validation errors.
2. `01_MAIN_TEXT/Figure_2_Primary_Contrasts.png` — the four dataset-level primary contrasts relative to C0.
3. `01_MAIN_TEXT/Figure_3_Dataset_Heterogeneity.png` — dataset-level differences across the four conditions.
4. `01_MAIN_TEXT/Figure_4_Matched_Sample_Size.png` — trajectories for the same seven datasets at all three fractions. Black points show their median. There are no uncertainty bars.
5. `01_MAIN_TEXT/Figure_5_Moderators.png` — dataset-level associations with feature-to-sample ratio and minority prevalence.
6. `01_MAIN_TEXT/Figure_6_Paired_Model_Family.png` — descriptive paired model-family differences across datasets. These are fallback summaries, not adjusted mixed-model estimates.

## Supplementary material

1. `02_SUPPLEMENTARY/Figure_S1_Model_Family_Sensitivity.png` — model-family sensitivity analysis.
2. `02_SUPPLEMENTARY/Figure_S2_Mixed_Model_Diagnostics.png` — mixed-effects diagnostic evidence.
3. `02_SUPPLEMENTARY/Figure_S3_Robustness.png` — fold-aggregation sensitivity and sample-fraction medians for the same seven datasets.
4. `02_SUPPLEMENTARY/Figure_S4_Adjusted_Model_Sensitivity.png` — adjusted mixed-model estimates retained only as sensitivity evidence. The diagnostic gate failed. These estimates do not support confirmatory adjusted inference.
5. `02_SUPPLEMENTARY/Figure_S5_Dataset_Differences.png` — the 14 primary differences in each condition. Dataset ordering varies between panels. This display does not establish symmetry.

## Interpretation and version boundary

C0 is the fold-contained reference described in the manuscript. The supplied flowchart uses the earlier label "Leakage-free reference". This label does not guarantee clinical availability of every predictor or rule out unidentified repeated patients. The documented grouping rule uses predictor signatures.

C4 retained a fixed 50 percent of globally ranked raw predictors. C0 tuned the retained percentage of transformed features. C4 also changed representation and the tuning search space. The comparison does not isolate feature-selection placement alone.

The former adjusted model-family main figure is now Fig S4. The matched seven-dataset sample-size plot replaces the changing-panel main plot. Superseded paths are removed from the current set. Earlier versions remain recoverable in Git history.

This update changes figures, their mapping, and integrity records. It does not change frozen experiment outputs, primary analysis tables, code, or the archived protocol. It does not certify end-to-end reproduction.
