#!/usr/bin/env python3
"""Freeze, audit, and analyse the completed study result set.

This script is outcome-blind with respect to file selection: the included
dataset identifiers are fixed below, and every expected atomic identity comes
from the locked pre-execution registries. Source results are read-only.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


INCLUDED_IDS = [
    "D001", "D002", "D003", "D005", "D006", "D009", "D010",
    "D011", "D013", "D014", "D015", "D016", "D019", "D020",
]
MODELS = ["logistic_regression", "random_forest", "gradient_boosted_trees"]
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "gradient_boosted_trees": "Gradient-boosted trees",
}
CONDITIONS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6"]
PRIMARY_CONTRASTS = ["C2", "C3", "C4", "C5"]
CONDITION_LABELS = {
    "C0": "Fold-contained reference",
    "C1": "Frozen holdout control",
    "C2": "Non-nested tuning",
    "C3": "Global preprocessing",
    "C4": "Global feature selection",
    "C5": "Global oversampling",
    "C6": "Fixed-hyperparameter control",
}
METRICS = [
    "AUROC", "AUPRC", "balanced_accuracy", "F1", "Brier",
    "calibration_intercept", "calibration_slope",
]
IDENTITY = [
    "dataset_id", "model", "phase", "sample_fraction", "chain",
    "condition", "repeat", "fold",
]
CANONICAL_FILES = [
    "manifest.json", "summary.json", "predictions.csv.gz", "provenance.json.gz"
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def norm(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def identity_tuple(row) -> tuple[str, ...]:
    return tuple(norm(row[key]) for key in IDENTITY)


def bootstrap_stat(values, statistic, rng, resamples=5000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return math.nan, math.nan, math.nan
    boot = np.empty(resamples, dtype=float)
    for idx in range(resamples):
        boot[idx] = statistic(rng.choice(x, size=len(x), replace=True))
    return float(statistic(x)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def rank_biserial(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & ~np.isclose(x, 0)]
    if len(x) == 0:
        return 0.0
    ranks = pd.Series(np.abs(x)).rank(method="average").to_numpy()
    pos = ranks[x > 0].sum()
    neg = ranks[x < 0].sum()
    return float((pos - neg) / (pos + neg))


def wilcoxon_exact(values):
    """Two-sided exact signed-rank test for the small dataset-level sample."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x) & ~np.isclose(x, 0)]
    if len(x) == 0:
        return 0.0, 1.0
    ranks = pd.Series(np.abs(x)).rank(method="average").to_numpy()
    observed_positive = float(ranks[x > 0].sum())
    total = float(ranks.sum())
    observed = min(observed_positive, total - observed_positive)
    counts = 0
    extreme = 0
    for mask in range(1 << len(ranks)):
        positive = sum(rank for idx, rank in enumerate(ranks) if mask & (1 << idx))
        statistic = min(positive, total - positive)
        counts += 1
        if statistic <= observed + 1e-12:
            extreme += 1
    return observed, min(1.0, extreme / counts)


def friedman_test(matrix):
    """Friedman chi-square statistic and df=4 chi-square survival probability."""
    x = np.asarray(matrix, dtype=float)
    n, k = x.shape
    ranks = np.vstack([pd.Series(row).rank(method="average").to_numpy() for row in x])
    rank_sums = ranks.sum(axis=0)
    statistic = 12.0 / (n * k * (k + 1)) * np.square(rank_sums).sum() - 3 * n * (k + 1)
    # This study has k=5, hence df=4 and Q(2, statistic/2) has a closed form.
    if k - 1 != 4:
        raise ValueError("Closed-form p-value implemented for five related conditions only")
    half = statistic / 2.0
    p_value = math.exp(-half) * (1.0 + half)
    return float(statistic), float(p_value)


def holm_adjust(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values.items(), key=lambda pair: pair[1])
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * value))
        adjusted[key] = running
    return adjusted


def read_registry_names(registry_path: Path) -> dict[str, str]:
    registry = pd.read_csv(registry_path, encoding="utf-8-sig")
    return dict(zip(registry["dataset_id"], registry["dataset_name"]))


def load_expected(primary_path: Path, sample_path: Path) -> pd.DataFrame:
    frames = []
    for path, phase in ((primary_path, "primary"), (sample_path, "sample_size")):
        frame = pd.read_csv(path)
        frame["phase"] = phase
        frames.append(frame[IDENTITY])
    expected = pd.concat(frames, ignore_index=True)
    return expected[expected["dataset_id"].isin(INCLUDED_IDS)].copy()


def validate_and_collect(results_root: Path, expected: pd.DataFrame):
    problems = []
    warnings = []
    rows = []
    manifest_rows = []
    seen_identities = Counter()
    duplicate_copy_files = []
    expected_set = {identity_tuple(row) for row in expected.to_dict("records")}

    for dataset_id in INCLUDED_IDS:
        dataset_root = results_root / dataset_id
        if not dataset_root.is_dir():
            problems.append({"scope": dataset_id, "problem": "dataset result directory missing"})
            continue
        for path in dataset_root.rglob("*"):
            if path.is_file() and "(" in path.name and ")" in path.name:
                duplicate_copy_files.append(path)
        for manifest_path in dataset_root.rglob("manifest.json"):
            run_dir = manifest_path.parent
            try:
                manifest = json.loads(manifest_path.read_text())
                summary_path = run_dir / "summary.json"
                pred_path = run_dir / "predictions.csv.gz"
                prov_path = run_dir / "provenance.json.gz"
                required = [summary_path, pred_path, prov_path]
                missing = [p.name for p in required if not p.is_file()]
                if missing:
                    problems.append({"scope": str(run_dir), "problem": "missing canonical artifacts: " + ", ".join(missing)})
                    continue
                checks = {
                    "summary.json": manifest.get("summary_sha256") == sha256(summary_path),
                    "predictions.csv.gz": manifest.get("predictions_sha256") == sha256(pred_path),
                    "provenance.json.gz": manifest.get("provenance_sha256") == sha256(prov_path),
                }
                if manifest.get("status") != "COMPLETE" or not all(checks.values()):
                    problems.append({"scope": str(run_dir), "problem": "manifest status/hash validation failed"})
                    continue
                summary = json.loads(summary_path.read_text())
                ident = identity_tuple(summary)
                seen_identities[ident] += 1
                if ident not in expected_set:
                    problems.append({"scope": summary.get("run_id", "unknown"), "problem": "unregistered atomic identity"})
                    continue
                metric_problem = False
                for metric in METRICS:
                    value = summary.get(metric)
                    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                        issue = {"scope": summary.get("run_id", "unknown"), "problem": f"non-estimable metric: {metric}"}
                        if metric in {"calibration_intercept", "calibration_slope"}:
                            warnings.append(issue)
                        else:
                            problems.append(issue)
                            metric_problem = True
                if metric_problem:
                    continue
                rows.append(summary)
                manifest_rows.append({
                    "dataset_id": summary["dataset_id"],
                    "model": summary["model"],
                    "phase": summary["phase"],
                    "condition": summary["condition"],
                    "sample_fraction": summary["sample_fraction"],
                    "chain": summary["chain"],
                    "repeat": summary["repeat"],
                    "fold": summary["fold"],
                    "run_id": summary["run_id"],
                    "completed_utc": manifest.get("completed_utc", ""),
                    "manifest_sha256": sha256(manifest_path),
                    "summary_sha256": checks["summary.json"] and manifest["summary_sha256"],
                    "predictions_sha256": checks["predictions.csv.gz"] and manifest["predictions_sha256"],
                    "provenance_sha256": checks["provenance.json.gz"] and manifest["provenance_sha256"],
                    "source_relative_path": str(run_dir.relative_to(results_root)),
                })
            except Exception as exc:
                problems.append({"scope": str(manifest_path), "problem": f"read/validation error: {type(exc).__name__}: {exc}"})

    found_set = set(seen_identities)
    missing_identities = expected_set - found_set
    extra_identities = found_set - expected_set
    duplicate_identities = {key: count for key, count in seen_identities.items() if count > 1}
    for ident in sorted(missing_identities):
        problems.append({"scope": "|".join(ident), "problem": "expected atomic identity missing"})
    for ident in sorted(extra_identities):
        problems.append({"scope": "|".join(ident), "problem": "unexpected atomic identity"})
    for ident, count in sorted(duplicate_identities.items()):
        problems.append({"scope": "|".join(ident), "problem": f"duplicate canonical identity ({count})"})
    return pd.DataFrame(rows), pd.DataFrame(manifest_rows), problems, warnings, duplicate_copy_files, expected_set


def freeze_validated(manifest_rows: pd.DataFrame, results_root: Path, freeze_root: Path):
    if freeze_root.exists():
        raise RuntimeError(f"Freeze destination already exists: {freeze_root}")
    for row in manifest_rows.to_dict("records"):
        source_dir = results_root / row["source_relative_path"]
        target_dir = freeze_root / row["source_relative_path"]
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in CANONICAL_FILES:
            shutil.copy2(source_dir / name, target_dir / name)
    for path in sorted(freeze_root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    freeze_root.chmod(0o555)


def aggregate(raw: pd.DataFrame, names: dict[str, str], seed=20260911):
    keys = ["dataset_id", "model", "condition", "sample_fraction", "chain"]
    agg = raw.groupby(keys, dropna=False)[METRICS].agg(["mean", "median", "std", "count"]).reset_index()
    agg.columns = ["_".join(col).rstrip("_") for col in agg.columns]
    agg["dataset_name"] = agg["dataset_id"].map(names)

    primary = agg[agg["sample_fraction"].eq(1.0)].copy()
    baseline = primary[primary["condition"].eq("C0")].set_index(["dataset_id", "model"])
    paired = []
    for row in primary[primary["condition"].isin(PRIMARY_CONTRASTS)].to_dict("records"):
        key = (row["dataset_id"], row["model"])
        if key not in baseline.index:
            continue
        item = {
            "dataset_id": row["dataset_id"],
            "dataset_name": names[row["dataset_id"]],
            "model": row["model"],
            "condition": row["condition"],
        }
        for metric in METRICS:
            item[f"reference_{metric}"] = float(baseline.loc[key, f"{metric}_mean"])
            item[f"apparent_{metric}"] = float(row[f"{metric}_mean"])
            item[f"delta_{metric}"] = item[f"apparent_{metric}"] - item[f"reference_{metric}"]
            item[f"delta_median_{metric}"] = float(row[f"{metric}_median"] - baseline.loc[key, f"{metric}_median"])
        paired.append(item)
    paired = pd.DataFrame(paired)

    dataset_effects = paired.groupby(["dataset_id", "dataset_name", "condition"], as_index=False).agg(
        **{f"delta_{m}": (f"delta_{m}", "mean") for m in METRICS},
        models=("model", "nunique"),
    )
    rng = np.random.default_rng(seed)
    tests = []
    p_values = {}
    for condition in PRIMARY_CONTRASTS:
        values = dataset_effects.loc[dataset_effects["condition"].eq(condition), "delta_AUROC"].to_numpy(float)
        median, low, high = bootstrap_stat(values, np.median, rng)
        if len(values) and not np.allclose(values, 0):
            statistic, p_value = wilcoxon_exact(values)
        else:
            statistic, p_value = 0.0, 1.0
        p_values[condition] = p_value
        tests.append({
            "contrast": f"{condition} vs C0",
            "condition": condition,
            "datasets": len(values),
            "median_delta_AUROC": median,
            "ci95_low": low,
            "ci95_high": high,
            "wilcoxon_statistic": statistic,
            "p_value": p_value,
            "rank_biserial_effect_size": rank_biserial(values),
        })
    adjusted = holm_adjust(p_values)
    for row in tests:
        row["holm_adjusted_p"] = adjusted[row["condition"]]
    tests = pd.DataFrame(tests)

    model_effects = []
    rng_model = np.random.default_rng(seed + 200)
    for model in MODELS:
        for condition in PRIMARY_CONTRASTS:
            values = paired.loc[
                paired["model"].eq(model) & paired["condition"].eq(condition), "delta_AUROC"
            ].to_numpy(float)
            median, low, high = bootstrap_stat(values, np.median, rng_model)
            model_effects.append({
                "model": model,
                "model_label": MODEL_LABELS[model],
                "condition": condition,
                "datasets": len(values),
                "median_delta_AUROC": median,
                "ci95_low": low,
                "ci95_high": high,
                "rank_biserial_effect_size": rank_biserial(values),
            })
    model_effects = pd.DataFrame(model_effects)

    sensitivity = []
    rng_sens = np.random.default_rng(seed + 400)
    for condition in PRIMARY_CONTRASTS:
        values = dataset_effects.loc[dataset_effects["condition"].eq(condition), "delta_AUROC"].to_numpy(float)
        med_values = paired[paired["condition"].eq(condition)].groupby("dataset_id")["delta_median_AUROC"].mean().to_numpy(float)
        for method, data in (("Mean fold aggregation", values), ("Median fold aggregation", med_values)):
            median, low, high = bootstrap_stat(data, np.median, rng_sens)
            sensitivity.append({
                "condition": condition, "aggregation": method, "datasets": len(data),
                "median_delta_AUROC": median, "ci95_low": low, "ci95_high": high,
            })
    sensitivity = pd.DataFrame(sensitivity)

    sample = agg[agg["sample_fraction"].lt(1.0)].copy()
    sample_base = sample[sample["condition"].eq("C0")].set_index(
        ["dataset_id", "model", "sample_fraction", "chain"]
    )
    sample_rows = []
    for row in sample[sample["condition"].isin(PRIMARY_CONTRASTS)].to_dict("records"):
        key = (row["dataset_id"], row["model"], row["sample_fraction"], row["chain"])
        if key not in sample_base.index:
            continue
        sample_rows.append({
            "dataset_id": row["dataset_id"], "dataset_name": names[row["dataset_id"]],
            "model": row["model"], "sample_fraction": row["sample_fraction"],
            "condition": row["condition"],
            "delta_AUROC": float(row["AUROC_mean"] - sample_base.loc[key, "AUROC_mean"]),
        })
    sample_effects = pd.DataFrame(sample_rows)
    if len(sample_effects):
        sample_summary = sample_effects.groupby(["sample_fraction", "condition"])["delta_AUROC"].agg(
            datasets="count", median_delta_AUROC="median"
        ).reset_index()
    else:
        sample_summary = pd.DataFrame(columns=["sample_fraction", "condition", "datasets", "median_delta_AUROC"])
    full_for_sample = dataset_effects[["dataset_id", "condition", "delta_AUROC"]].copy()
    full_for_sample["sample_fraction"] = 1.0
    full_summary = full_for_sample.groupby(["sample_fraction", "condition"])["delta_AUROC"].agg(
        datasets="count", median_delta_AUROC="median"
    ).reset_index()
    sample_summary = pd.concat([sample_summary, full_summary], ignore_index=True)

    wide = dataset_effects.pivot(index="dataset_id", columns="condition", values="delta_AUROC").dropna()
    if len(wide):
        matrix = np.column_stack([np.zeros(len(wide))] + [wide[c].to_numpy() for c in PRIMARY_CONTRASTS])
        friedman_statistic, friedman_p = friedman_test(matrix)
        omnibus = pd.DataFrame([{
            "test": "Friedman across reference and four primary conditions",
            "datasets": len(wide), "statistic": friedman_statistic, "p_value": friedman_p,
        }])
    else:
        omnibus = pd.DataFrame([{"test": "Friedman", "datasets": 0, "statistic": math.nan, "p_value": math.nan}])
    return agg, paired, dataset_effects, tests, model_effects, sensitivity, sample_effects, sample_summary, omnibus


def save_tables(out: Path, tables: dict[str, pd.DataFrame]):
    table_dir = out / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False, quoting=csv.QUOTE_MINIMAL)


def write_reports(out: Path, expected: pd.DataFrame, manifest_rows: pd.DataFrame,
                  problems: list[dict], warnings: list[dict], duplicate_copy_files: list[Path], names: dict[str, str],
                  source_root: Path, tables: dict[str, pd.DataFrame]):
    package_counts = manifest_rows.groupby(["dataset_id", "model"]).size().reset_index(name="validated_atomic_units")
    expected_counts = expected.groupby(["dataset_id", "model"]).size().reset_index(name="expected_atomic_units")
    completion = expected_counts.merge(package_counts, how="left", on=["dataset_id", "model"]).fillna({"validated_atomic_units": 0})
    completion["complete"] = completion["expected_atomic_units"].eq(completion["validated_atomic_units"])
    completion["dataset_name"] = completion["dataset_id"].map(names)
    display_map = pd.DataFrame({
        "display_index": [f"Dataset {idx:02d}" for idx in range(1, len(INCLUDED_IDS) + 1)],
        "dataset_name": [names[key] for key in INCLUDED_IDS],
    })
    completion_public = completion.merge(display_map, on="dataset_name").drop(columns=["dataset_id"])
    completion_public = completion_public[["display_index", "dataset_name", "model", "expected_atomic_units", "validated_atomic_units", "complete"]]
    completion_public.to_csv(out / "tables" / "completion_matrix.csv", index=False)
    display_map.to_csv(out / "tables" / "dataset_display_map.csv", index=False)

    status = "PASS" if not problems and completion["complete"].all() else "FAIL"
    audit = {
        "audit_created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_set_definition": "All datasets listed in tables/dataset_display_map.csv with fully validated registered atomic results.",
        "dataset_count": len(INCLUDED_IDS),
        "model_family_count": len(MODELS),
        "expected_atomic_units": int(len(expected)),
        "validated_atomic_units": int(len(manifest_rows)),
        "canonical_identity_duplicates": 0,
        "noncanonical_copy_files_ignored": len(duplicate_copy_files),
        "problem_count": len(problems),
        "metric_quality_warning_count": len(warnings),
        "status": status,
    }
    (out / "audit_summary.json").write_text(json.dumps(audit, indent=2) + "\n")
    pd.DataFrame(problems, columns=["scope", "problem"]).to_csv(out / "audit_problems.csv", index=False)
    pd.DataFrame(warnings, columns=["scope", "problem"]).to_csv(out / "metric_quality_warnings.csv", index=False)

    report = [
        "# Final analysis package audit",
        "",
        f"Status: **{status}**",
        "",
        f"- Included datasets: {len(INCLUDED_IDS)}",
        f"- Model families: {len(MODELS)}",
        f"- Expected registered atomic units: {len(expected):,}",
        f"- Validated canonical atomic units: {len(manifest_rows):,}",
        f"- Noncanonical duplicate-copy files ignored: {len(duplicate_copy_files):,}",
        f"- Integrity problems: {len(problems):,}",
        f"- Metric-quality warnings: {len(warnings):,}",
        "",
        "Selection was fixed before reading performance values. Every canonical manifest, summary, prediction archive, and provenance archive was hash-validated. Completeness was checked against the locked pre-execution identities. Noncanonical OS-created copies were not analysed.",
        "",
        "Raw source results were not modified. The frozen snapshot contains canonical validated files only and is read-only.",
    ]
    (out / "AUDIT_REPORT.md").write_text("\n".join(report) + "\n")

    provenance = {
        "source_results_root": str(source_root),
        "selection_rule": "fixed included identifier list in build_final_analysis.py",
        "source_files_modified": False,
        "analysis_script_sha256": sha256(Path(__file__)),
        "tables": {name: len(frame) for name, frame in tables.items()},
    }
    (out / "analysis_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def write_inventory(out: Path):
    records = []
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in {"file_inventory_sha256.csv", "package_manifest.json"}:
            records.append({
                "relative_path": str(path.relative_to(out)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    inventory = pd.DataFrame(records)
    inventory.to_csv(out / "file_inventory_sha256.csv", index=False)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "file_count_excluding_inventory_and_manifest": len(records),
        "inventory_sha256": sha256(out / "file_inventory_sha256.csv"),
    }
    (out / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-package", required=True, type=Path)
    parser.add_argument("--dataset-map", required=True, type=Path)
    parser.add_argument("--primary-registry", required=True, type=Path)
    parser.add_argument("--sample-size-registry", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--skip-freeze", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if out.exists():
        raise RuntimeError(f"Output already exists: {out}")
    out.mkdir(parents=True)

    display_map = pd.read_csv(args.dataset_map)
    names = dict(zip(display_map["dataset_id"], display_map["dataset_name"]))
    expected = load_expected(args.primary_registry, args.sample_size_registry)
    raw, manifest_rows, problems, warnings, copies, _ = validate_and_collect(args.source_package / "results", expected)
    if problems:
        # Write evidence before refusing analysis or freezing.
        pd.DataFrame(problems).to_csv(out / "audit_problems.csv", index=False)
        raise RuntimeError(f"Integrity audit failed with {len(problems)} problem(s)")

    raw["dataset_name"] = raw["dataset_id"].map(names)
    outputs = aggregate(raw, names)
    table_names = [
        "fold_aggregates", "paired_model_effects", "dataset_condition_effects",
        "confirmatory_tests", "model_family_effects", "aggregation_sensitivity",
        "sample_size_model_effects", "sample_size_summary", "omnibus_test",
    ]
    tables = dict(zip(table_names, outputs))
    save_tables(out, tables)
    manifest_rows.to_csv(out / "validated_result_manifest.csv", index=False)
    write_reports(out, expected, manifest_rows, problems, warnings, copies, names,
                  args.source_package / "results", tables)
    shutil.copy2(Path(__file__), out / "build_final_analysis.py")
    if not args.skip_freeze:
        freeze_validated(manifest_rows, args.source_package / "results", out / "frozen_results")
    write_inventory(out)
    print(json.dumps({
        "status": "PASS", "output": str(out), "datasets": len(INCLUDED_IDS),
        "atomic_units": len(manifest_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
