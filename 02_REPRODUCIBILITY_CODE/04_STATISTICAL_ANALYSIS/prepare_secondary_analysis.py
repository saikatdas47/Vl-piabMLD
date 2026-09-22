#!/usr/bin/env python3
"""Prepare protocol-aligned H3-H6 analysis data from the frozen result package."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CONDITIONS = ["C2", "C3", "C4", "C5"]
MODELS = ["logistic_regression", "random_forest", "gradient_boosted_trees"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--characteristics", required=True, type=Path)
    args = ap.parse_args()
    package = args.package.resolve()
    out = package / "secondary_analysis"
    out.mkdir(parents=True, exist_ok=True)

    full = pd.read_csv(package / "tables/paired_model_effects.csv")
    full = full[["dataset_id", "dataset_name", "model", "condition", "delta_AUROC"]].copy()
    full["sample_fraction"] = 1.0
    lower = pd.read_csv(package / "tables/sample_size_model_effects.csv")
    lower = lower[["dataset_id", "dataset_name", "model", "condition", "sample_fraction", "delta_AUROC"]].copy()
    cells = pd.concat([full, lower], ignore_index=True)
    cells = cells[cells["condition"].isin(CONDITIONS) & cells["model"].isin(MODELS)].copy()

    characteristics = pd.read_csv(args.characteristics, encoding="utf-8-sig")
    keep = [
        "dataset_id", "analysis_rows", "analysis_predictors", "minority_count",
        "missing_predictor_cells",
    ]
    characteristics = characteristics[keep]
    cells = cells.merge(characteristics, on="dataset_id", how="left", validate="many_to_one")

    exact_rows = []
    for summary_path in (package / "frozen_results").rglob("summary.json"):
        data = json.loads(summary_path.read_text())
        if data.get("condition") in CONDITIONS:
            exact_rows.append({
                "dataset_id": data["dataset_id"], "model": data["model"],
                "condition": data["condition"], "sample_fraction": float(data["sample_fraction"]),
                "rows_in_analysis_frame": int(data["rows_in_analysis_frame"]),
            })
    exact = pd.DataFrame(exact_rows).groupby(
        ["dataset_id", "model", "condition", "sample_fraction"], as_index=False
    )["rows_in_analysis_frame"].median()
    cells = cells.merge(exact, on=["dataset_id", "model", "condition", "sample_fraction"], how="left", validate="one_to_one")
    if cells["rows_in_analysis_frame"].isna().any():
        raise RuntimeError("Exact analysis-row count missing for one or more cells")

    display_map = pd.read_csv(package / "tables/dataset_display_map.csv")
    cells = cells.merge(display_map, on="dataset_name", how="left", validate="many_to_one")
    cells["minority_prevalence"] = cells["minority_count"] / cells["analysis_rows"]
    cells["missingness_rate"] = cells["missing_predictor_cells"] / (
        cells["analysis_rows"] * cells["analysis_predictors"]
    )
    cells["feature_to_sample_ratio"] = cells["analysis_predictors"] / cells["rows_in_analysis_frame"]
    cells["log_sample_size"] = np.log(cells["rows_in_analysis_frame"])
    cells["log_feature_to_sample_ratio"] = np.log(cells["feature_to_sample_ratio"])
    for column in ["log_sample_size", "log_feature_to_sample_ratio", "minority_prevalence", "missingness_rate"]:
        sd = cells[column].std(ddof=1)
        cells[f"z_{column}"] = (cells[column] - cells[column].mean()) / sd

    cells = cells.sort_values(["display_index", "model", "condition", "sample_fraction"])
    cells.to_csv(out / "h3_h6_cell_level_analysis_data.csv", index=False)

    # Correct sample-size summary: dataset is the independent unit.
    dataset_fraction = cells.groupby(
        ["dataset_id", "display_index", "dataset_name", "condition", "sample_fraction"], as_index=False
    )["delta_AUROC"].mean()
    rng = np.random.default_rng(20260911)
    rows = []
    for keys, block in dataset_fraction.groupby(["condition", "sample_fraction"]):
        values = block["delta_AUROC"].to_numpy(float)
        boot = np.median(rng.choice(values, size=(5000, len(values)), replace=True), axis=1)
        rows.append({
            "condition": keys[0], "sample_fraction": keys[1],
            "independent_datasets": len(values),
            "median_delta_AUROC": float(np.median(values)),
            "ci95_low": float(np.quantile(boot, .025)),
            "ci95_high": float(np.quantile(boot, .975)),
        })
    corrected = pd.DataFrame(rows).sort_values(["condition", "sample_fraction"])
    corrected.to_csv(out / "h3_corrected_sample_size_summary.csv", index=False)
    dataset_fraction.to_csv(out / "h3_dataset_level_fraction_effects.csv", index=False)

    complete_ids = set.intersection(*[
        set(dataset_fraction.loc[dataset_fraction["sample_fraction"].eq(f), "dataset_id"])
        for f in [0.5, 0.75, 1.0]
    ])
    matched = dataset_fraction[dataset_fraction["dataset_id"].isin(complete_ids)].copy()
    matched.to_csv(out / "h3_matched_all_fraction_dataset_effects.csv", index=False)

    audit = {
        "cell_rows": len(cells),
        "datasets": int(cells["dataset_id"].nunique()),
        "models": int(cells["model"].nunique()),
        "conditions": int(cells["condition"].nunique()),
        "datasets_estimable_at_50_percent": int(dataset_fraction.loc[dataset_fraction.sample_fraction.eq(.5), "dataset_id"].nunique()),
        "datasets_estimable_at_75_percent": int(dataset_fraction.loc[dataset_fraction.sample_fraction.eq(.75), "dataset_id"].nunique()),
        "datasets_estimable_at_100_percent": int(dataset_fraction.loc[dataset_fraction.sample_fraction.eq(1), "dataset_id"].nunique()),
        "datasets_with_all_three_fractions": len(complete_ids),
        "missing_analysis_fields": int(cells.isna().sum().sum()),
    }
    (out / "secondary_data_preparation_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
