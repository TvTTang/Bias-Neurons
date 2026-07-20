#!/usr/bin/env python3
"""Freeze a V3 candidate using robust multi-environment validation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_validation_candidate import (  # noqa: E402
    load_candidate_sets,
    load_validation_rows,
    sha256,
)


ENVIRONMENTS = ("iid", "lexical_ood", "template_ood")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze one direction-aware candidate using max-min validation "
            "across IID, lexical-OOD, and template-OOD environments."
        )
    )
    parser.add_argument("--candidate-sets", required=True, type=Path)
    parser.add_argument("--iid-csv", required=True, type=Path)
    parser.add_argument("--lexical-ood-csv", required=True, type=Path)
    parser.add_argument("--template-ood-csv", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument(
        "--min-target-probability-retention", default=0.95, type=float
    )
    parser.add_argument("--min-top1-agreement", default=0.995, type=float)
    parser.add_argument("--max-mean-kl", default=1e-4, type=float)
    parser.add_argument("--max-nll-increase", default=0.005, type=float)
    parser.add_argument(
        "--min-environment-bias-reduction", default=0.0, type=float
    )
    return parser.parse_args()


def index_rows(
    rows: Sequence[Dict[str, object]],
) -> Dict[str, Dict[str, object]]:
    indexed = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in indexed:
            raise ValueError(f"Duplicate validation candidate: {candidate_id}")
        indexed[candidate_id] = row
    return indexed


def assess_robust_candidates(
    baselines: Dict[str, Dict[str, float]],
    rows_by_environment: Dict[str, Sequence[Dict[str, object]]],
    candidate_sets: Dict[str, Dict[str, object]],
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    if set(baselines) != set(ENVIRONMENTS) or set(
        rows_by_environment
    ) != set(ENVIRONMENTS):
        raise ValueError(f"Exactly these environments are required: {ENVIRONMENTS}")
    indexed = {
        environment: index_rows(rows_by_environment[environment])
        for environment in ENVIRONMENTS
    }
    expected_ids = set(candidate_sets)
    for environment in ENVIRONMENTS:
        if set(indexed[environment]) != expected_ids:
            raise ValueError(
                f"Candidate IDs differ in environment {environment}."
            )
        if baselines[environment][
            "bias_mean_pair_target_probability"
        ] <= 0:
            raise ValueError(
                f"Baseline target probability is not positive in {environment}."
            )

    assessed = []
    for candidate_id, metadata in candidate_sets.items():
        environment_metrics = {}
        environment_checks = {}
        reductions = []
        for environment in ENVIRONMENTS:
            row = indexed[environment][candidate_id]
            baseline = baselines[environment]
            reduction = row["bias_absolute_gap_reduction"]
            retention = (
                row["bias_mean_pair_target_probability"]
                / baseline["bias_mean_pair_target_probability"]
            )
            checks = {
                "positive_bias_reduction": (
                    reduction
                    > thresholds["min_environment_bias_reduction"]
                ),
                "target_probability_retention": (
                    retention
                    >= thresholds["min_target_probability_retention"]
                ),
            }
            environment_metrics[environment] = {
                "bias_absolute_gap_reduction": reduction,
                "target_probability_retention": retention,
                "bias_mean_absolute_gap": row["bias_mean_absolute_gap"],
                "baseline_bias_mean_absolute_gap": baseline[
                    "bias_mean_absolute_gap"
                ],
                "num_bias_examples": int(row["num_bias_examples"]),
            }
            environment_checks[environment] = checks
            reductions.append(reduction)

        iid_row = indexed["iid"][candidate_id]
        semantic_checks = {
            "top1_agreement": (
                iid_row["semantic_baseline_top1_agreement"]
                >= thresholds["min_top1_agreement"]
            ),
            "mean_kl": (
                iid_row["semantic_mean_kl_from_baseline"]
                <= thresholds["max_mean_kl"]
            ),
            "nll_increase": (
                iid_row["semantic_nll_increase"]
                <= thresholds["max_nll_increase"]
            ),
        }
        scale = float(metadata["intervention_scale"])
        if not math.isfinite(scale) or scale < 0 or scale == 1:
            raise ValueError(
                f"Candidate {candidate_id} has an invalid V3 scale."
            )
        feasible = all(semantic_checks.values()) and all(
            all(checks.values()) for checks in environment_checks.values()
        )
        assessed.append(
            {
                "candidate_id": candidate_id,
                "num_neurons": int(iid_row["num_neurons"]),
                "intervention_scale": scale,
                "intervention_magnitude": abs(scale - 1.0),
                "stability_min": float(metadata["stability_min"]),
                "semantic_weight": float(metadata["semantic_weight"]),
                "worst_environment_bias_reduction": min(reductions),
                "mean_environment_bias_reduction": sum(reductions)
                / len(reductions),
                "environment_metrics": environment_metrics,
                "environment_checks": environment_checks,
                "semantic_metrics": {
                    "num_examples": int(iid_row["num_semantic_examples"]),
                    "top1_agreement": iid_row[
                        "semantic_baseline_top1_agreement"
                    ],
                    "mean_kl_from_baseline": iid_row[
                        "semantic_mean_kl_from_baseline"
                    ],
                    "nll_increase": iid_row["semantic_nll_increase"],
                },
                "semantic_checks": semantic_checks,
                "feasible": feasible,
            }
        )
    return assessed


def robust_selection_key(candidate: Dict[str, object]) -> Tuple[object, ...]:
    return (
        -candidate["worst_environment_bias_reduction"],
        -candidate["mean_environment_bias_reduction"],
        candidate["num_neurons"],
        candidate["intervention_magnitude"],
        -candidate["stability_min"],
        candidate["semantic_weight"],
        candidate["candidate_id"],
    )


def freeze_robust(
    dimension: str,
    baselines: Dict[str, Dict[str, float]],
    assessed: Sequence[Dict[str, object]],
    candidate_sets: Dict[str, Dict[str, object]],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    feasible = sorted(
        (candidate for candidate in assessed if candidate["feasible"]),
        key=robust_selection_key,
    )
    selected = feasible[0] if feasible else None
    if selected is None:
        status = "no_intervention"
        candidate_id = None
        neurons = []
        scale = 1.0
        reason = (
            "No candidate improved every validation environment while "
            "satisfying all target-retention and semantic constraints."
        )
    else:
        status = "intervention"
        candidate_id = selected["candidate_id"]
        neurons = candidate_sets[candidate_id]["neurons"]
        scale = selected["intervention_scale"]
        reason = (
            "Selected the feasible candidate with the largest worst-environment "
            "validation reduction using the pre-registered tie-break."
        )
    return {
        "protocol_version": "v3",
        "dimension": dimension,
        "status": status,
        "selected_candidate_id": candidate_id,
        "selected_neurons": neurons,
        "selected_intervention_scale": scale,
        "selection_reason": reason,
        "thresholds": thresholds,
        "environments": list(ENVIRONMENTS),
        "selection_order": [
            "maximum worst-environment bias reduction",
            "maximum mean-environment bias reduction",
            "fewer neurons",
            "smaller absolute distance of intervention scale from 1",
            "higher stability_min",
            "smaller semantic_weight",
            "lexicographically smaller candidate_id",
        ],
        "num_candidates": len(assessed),
        "num_feasible_candidates": len(feasible),
        "baseline_validation_metrics": baselines,
        "selected_validation": selected,
    }


def main() -> None:
    args = parse_args()
    if args.output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite frozen selection: {args.output_file}"
        )
    thresholds = {
        "min_target_probability_retention": (
            args.min_target_probability_retention
        ),
        "min_top1_agreement": args.min_top1_agreement,
        "max_mean_kl": args.max_mean_kl,
        "max_nll_increase": args.max_nll_increase,
        "min_environment_bias_reduction": (
            args.min_environment_bias_reduction
        ),
    }
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
    candidate_sets = load_candidate_sets(args.candidate_sets)
    assessed = assess_robust_candidates(
        baselines, rows_by_environment, candidate_sets, thresholds
    )
    result = freeze_robust(
        args.dimension,
        baselines,
        assessed,
        candidate_sets,
        thresholds,
    )
    result["candidate_sets_path"] = str(args.candidate_sets.resolve())
    result["candidate_sets_sha256"] = sha256(args.candidate_sets)
    result["validation_sources"] = {
        environment: {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }
        for environment, path in paths.items()
    }
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
