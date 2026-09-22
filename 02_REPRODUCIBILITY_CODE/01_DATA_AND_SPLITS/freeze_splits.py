#!/usr/bin/env python3
"""Create the frozen repeated outer split registry for COHORT-FREEZE-01."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "data" / "00_tools" / "ml_vendor"))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


MASTER_SEED = 20260910
OUTER_FOLDS = 3
OUTER_REPEATS = 2
INNER_FOLDS = 3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    text = "|".join(map(str, parts)).encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:4], "big")


def main() -> int:
    registry = ROOT / "data" / "00_registry"
    split_root = ROOT / "data" / "04_splits"
    split_root.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((registry / "cohort_freeze_01.json").read_text())
    outer_seeds = [stable_seed(MASTER_SEED, "outer", repeat) for repeat in range(OUTER_REPEATS)]
    validation_rows = []

    for entry in freeze["datasets"]:
        dataset_id = entry["dataset_id"]
        data = pd.read_csv(ROOT / entry["analysis_reference"], low_memory=False)
        y = data["__target__"].astype(int).to_numpy()
        groups = data["__group_id__"].astype(str).to_numpy()
        out_path = split_root / f"{dataset_id}_outer_assignments.csv.gz"
        with gzip.open(out_path, "wt", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["dataset_id", "row_index", "source_row", "group_id", "target", "repeat", "outer_fold"])
            for repeat, seed in enumerate(outer_seeds):
                splitter = StratifiedGroupKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
                assignment = np.full(len(data), -1, dtype=int)
                for fold, (_, test_index) in enumerate(splitter.split(np.zeros(len(data)), y, groups)):
                    assignment[test_index] = fold
                    test_groups = set(groups[test_index])
                    train_groups = set(groups[np.setdiff1d(np.arange(len(data)), test_index)])
                    if test_groups & train_groups:
                        raise RuntimeError(f"Group leakage in {dataset_id}, repeat {repeat}, fold {fold}")
                    counts = np.bincount(y[test_index], minlength=2)
                    if (counts == 0).any():
                        raise RuntimeError(f"Missing class in {dataset_id}, repeat {repeat}, fold {fold}")
                    train_index = np.setdiff1d(np.arange(len(data)), test_index)
                    inner = StratifiedGroupKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=stable_seed(MASTER_SEED, "inner", dataset_id, repeat, fold))
                    inner_count = 0
                    for inner_train, inner_valid in inner.split(np.zeros(len(train_index)), y[train_index], groups[train_index]):
                        if set(groups[train_index][inner_train]) & set(groups[train_index][inner_valid]):
                            raise RuntimeError(f"Inner group leakage in {dataset_id}, repeat {repeat}, fold {fold}")
                        if len(np.unique(y[train_index][inner_train])) < 2 or len(np.unique(y[train_index][inner_valid])) < 2:
                            raise RuntimeError(f"Inner fold class infeasibility in {dataset_id}, repeat {repeat}, fold {fold}")
                        inner_count += 1
                    validation_rows.append({"dataset_id": dataset_id, "repeat": repeat, "fold": fold, "test_rows": len(test_index), "test_negative": int(counts[0]), "test_positive": int(counts[1]), "test_groups": len(test_groups), "group_overlap": 0, "inner_folds_validated": inner_count, "status": "PASS"})
                if (assignment < 0).any():
                    raise RuntimeError(f"Unassigned rows in {dataset_id}, repeat {repeat}")
                for row_index, fold in enumerate(assignment):
                    writer.writerow([dataset_id, row_index, int(data.iloc[row_index]["__source_row__"]), groups[row_index], int(y[row_index]), repeat, int(fold)])
        entry["outer_split_registry"] = str(out_path.relative_to(ROOT))
        entry["outer_split_registry_sha256"] = sha256(out_path)
        print(f"{dataset_id}: 2x3 outer assignments frozen", flush=True)

    with (split_root / "split_validation.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validation_rows[0]))
        writer.writeheader()
        writer.writerows(validation_rows)

    seed_registry = {
        "master_seed": MASTER_SEED,
        "outer_folds": OUTER_FOLDS,
        "outer_repeats": OUTER_REPEATS,
        "outer_repeat_seeds": outer_seeds,
        "inner_folds": INNER_FOLDS,
        "inner_seed_rule": "first 32 bits of SHA-256(master_seed|inner|dataset_id|repeat|outer_fold)",
        "model_seed_rule": "first 32 bits of SHA-256(master_seed|model|dataset_id|model_name|condition|repeat|outer_fold)",
        "hyperparameter_seed_rule": "first 32 bits of SHA-256(master_seed|search|dataset_id|model_name|condition|repeat|outer_fold)",
        "subsample_seed_rule": "first 32 bits of SHA-256(master_seed|subsample|dataset_id|chain|fraction)",
        "bootstrap_seed": stable_seed(MASTER_SEED, "bootstrap"),
    }
    (registry / "seed_registry.json").write_text(json.dumps(seed_registry, indent=2), encoding="utf-8")
    freeze["split_freeze_utc"] = datetime.now(timezone.utc).isoformat()
    freeze["outer_folds"] = OUTER_FOLDS
    freeze["outer_repeats"] = OUTER_REPEATS
    freeze["inner_folds"] = INNER_FOLDS
    freeze["datasets"] = freeze["datasets"]
    freeze_path = registry / "cohort_freeze_01.json"
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    (registry / "cohort_freeze_01.sha256").write_text(f"{sha256(freeze_path)}  cohort_freeze_01.json\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
