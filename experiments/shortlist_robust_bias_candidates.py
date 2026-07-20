#!/usr/bin/env python3
"""Shortlist candidates that improve every bias-validation environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence


sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_validation_candidate import (  # noqa: E402
    load_validation_rows,
    sha256,
)


ENVIRONMENTS = ("iid", "lexical_ood", "template_ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Keep candidates with strictly positive bias reduction and "
            "sufficient target-probability retention in all three validation "
            "environments, before full semantic evaluation."
        )
    )
    parser.add_argument("--candidate-sets", required=True, type=Path)
    parser.add_argument("--iid-csv", required=True, type=Path)
    parser.add_argument("--lexical-ood-csv", required=True, type=Path)
    parser.add_argument("--template-ood-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--min-target-probability-retention", default=0.95, type=float
    )
    parser.add_argument(
        "--min-environment-bias-reduction", default=0.0, type=float
    )
    return parser.parse_args()


def load_candidates(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        candidates = json.load(handle)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate file must contain a non-empty JSON list.")
    ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) for candidate_id in ids):
        raise ValueError("Candidate IDs must be strings.")
    if len(set(ids)) != len(ids):
        raise ValueError("Candidate IDs must be unique.")
    return candidates


def index_rows(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    indexed = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in indexed:
            raise ValueError(f"Duplicate candidate row: {candidate_id}")
        indexed[candidate_id] = row
    return indexed


def robust_bias_shortlist(
    candidates: Sequence[Dict[str, object]],
    baselines: Dict[str, Dict[str, float]],
    rows_by_environment: Dict[str, Sequence[Dict[str, object]]],
    min_retention: float,
    min_reduction: float,
) -> List[Dict[str, object]]:
    if set(baselines) != set(ENVIRONMENTS) or set(
        rows_by_environment
    ) != set(ENVIRONMENTS):
        raise ValueError(f"Exactly these environments are required: {ENVIRONMENTS}")
    if not 0 <= min_retention <= 1:
        raise ValueError("Minimum retention must be in [0, 1].")
    indexed = {
        environment: index_rows(rows_by_environment[environment])
        for environment in ENVIRONMENTS
    }
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    for environment in ENVIRONMENTS:
        if set(indexed[environment]) != candidate_ids:
            raise ValueError(
                f"Candidate IDs differ in environment {environment}."
            )

    shortlisted = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        metrics = {}
        feasible = True
        for environment in ENVIRONMENTS:
            baseline = baselines[environment]
            row = indexed[environment][candidate_id]
            baseline_probability = baseline[
                "bias_mean_pair_target_probability"
            ]
            if baseline_probability <= 0:
                raise ValueError("Baseline target probability must be positive.")
            retention = (
                row["bias_mean_pair_target_probability"]
                / baseline_probability
            )
            reduction = row["bias_absolute_gap_reduction"]
            metrics[environment] = {
                "bias_absolute_gap_reduction": reduction,
                "target_probability_retention": retention,
            }
            feasible = feasible and reduction > min_reduction
            feasible = feasible and retention >= min_retention
        if feasible:
            value = dict(candidate)
            value["bias_validation"] = metrics
            shortlisted.append(value)
    return shortlisted


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    candidates = load_candidates(args.candidate_sets)
    paths = {
        "iid": args.iid_csv,
        "lexical_ood": args.lexical_ood_csv,
        "template_ood": args.template_ood_csv,
    }
    baselines = {}
    rows_by_environment = {}
    for environment, path in paths.items():
        baseline, rows = load_validation_rows(path)
        baselines[environment] = baseline
        rows_by_environment[environment] = rows
    shortlisted = robust_bias_shortlist(
        candidates,
        baselines,
        rows_by_environment,
        args.min_target_probability_retention,
        args.min_environment_bias_reduction,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "candidate_sets.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(shortlisted, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "protocol_version": "v3_bias_shortlist",
        "candidate_sets": str(args.candidate_sets.resolve()),
        "candidate_sets_sha256": sha256(args.candidate_sets),
        "validation_sources": {
            environment: {
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for environment, path in paths.items()
        },
        "environments": list(ENVIRONMENTS),
        "min_target_probability_retention": (
            args.min_target_probability_retention
        ),
        "min_environment_bias_reduction": (
            args.min_environment_bias_reduction
        ),
        "num_input_candidates": len(candidates),
        "num_shortlisted_candidates": len(shortlisted),
        "output_candidate_sets": str(output_path.resolve()),
        "output_candidate_sets_sha256": sha256(output_path),
        "next_step": (
            "run full semantic validation only on this shortlist; no candidate "
            "may be frozen before the semantic constraints are applied"
        ),
    }
    with (args.output_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
