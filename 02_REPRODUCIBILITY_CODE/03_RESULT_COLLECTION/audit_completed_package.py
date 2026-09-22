#!/usr/bin/env python3
"""Audit one completed dataset-model execution index without reading model scores."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    index_path = Path(args.index).resolve()
    index = json.loads(index_path.read_text())
    config_hash = sha256(ROOT / "config" / "experiment_config.json")
    problems: list[dict[str, str]] = []
    identities: list[tuple] = []
    conditions: Counter[str] = Counter()
    phases: Counter[str] = Counter()
    models: set[str] = set()

    for job in index["jobs"]:
        identities.append((job["dataset_id"], job["model"], job["phase"], job["sample_fraction"], job["chain"], job["condition"], job["repeat"], str(job["fold"])))
        conditions[job["condition"]] += 1
        phases[job["phase"]] += 1
        models.add(job["model"])
        run_dir = Path(job["output_directory"])
        manifest_path = run_dir / "manifest.json"
        if job.get("status") not in {"COMPLETE", "SKIPPED_COMPLETE"}:
            problems.append({"run_id": job.get("run_id", ""), "problem": "index status is neither COMPLETE nor SKIPPED_COMPLETE"})
            continue
        if not manifest_path.is_file():
            problems.append({"run_id": job.get("run_id", ""), "problem": "manifest missing"})
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "COMPLETE":
            problems.append({"run_id": job["run_id"], "problem": "manifest status is not COMPLETE"})
        if manifest.get("run_id") != job.get("run_id"):
            problems.append({"run_id": job["run_id"], "problem": "index/manifest run_id mismatch"})
        if manifest.get("config_sha256") != config_hash:
            problems.append({"run_id": job["run_id"], "problem": "config hash mismatch"})
        for filename, key in (("summary.json", "summary_sha256"), ("predictions.csv.gz", "predictions_sha256"), ("provenance.json.gz", "provenance_sha256")):
            artifact = run_dir / filename
            if not artifact.is_file():
                problems.append({"run_id": job["run_id"], "problem": f"{filename} missing"})
            elif sha256(artifact) != manifest.get(key):
                problems.append({"run_id": job["run_id"], "problem": f"{filename} hash mismatch"})

    report = {
        "audit_type": "COMPLETED_DATASET_MODEL_PACKAGE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "index": str(index_path.relative_to(ROOT)),
        "dataset_id": index.get("dataset_id"),
        # Older/current runner indexes do not carry a top-level model list.
        # Derive it from the registered jobs so completed checkpoints remain
        # valid and no result-producing runner hash needs to change.
        "models": sorted(models),
        "expected_jobs": index.get("jobs") and len(index["jobs"]),
        "complete_jobs": sum(job.get("status") in {"COMPLETE", "SKIPPED_COMPLETE"} for job in index["jobs"]),
        "unique_registered_job_identities": len(set(identities)),
        "condition_counts": dict(sorted(conditions.items())),
        "phase_counts": dict(sorted(phases.items())),
        "current_config_sha256": config_hash,
        "problem_count": len(problems),
        "problems": problems,
        "performance_values_inspected": False,
        "status": "PASS" if not problems and len(set(identities)) == len(index["jobs"]) else "FAIL",
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
