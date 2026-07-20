#!/usr/bin/env python3
"""Freeze one validation-selected candidate before test/OOD evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply pre-registered bias/semantic validation constraints and "
            "freeze one candidate, or no intervention when none is feasible."
        )
    )
    parser.add_argument("--candidate-sets", required=True, type=Path)
    parser.add_argument("--validation-csv", required=True, type=Path)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument(
        "--min-target-probability-retention", default=0.95, type=float
    )
    parser.add_argument("--min-top1-agreement", default=0.995, type=float)
    parser.add_argument("--max-mean-kl", default=1e-4, type=float)
    parser.add_argument("--max-nll-increase", default=0.005, type=float)
    parser.add_argument("--min-bias-reduction", default=0.0, type=float)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidate_sets(path: Path) -> Dict[str, Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise ValueError("Candidate file must contain a non-empty JSON list.")
    candidates = {}
    for value in values:
        candidate_id = value.get("candidate_id")
        neurons = value.get("neurons")
        if not isinstance(candidate_id, str) or candidate_id in candidates:
            raise ValueError("Candidate IDs must be unique strings.")
        if not isinstance(neurons, list) or not neurons:
            raise ValueError(f"Candidate {candidate_id} has no neurons.")
        candidates[candidate_id] = value
    return candidates


def load_validation_rows(
    path: Path,
) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("Validation CSV is empty.")
    baseline_rows = [
        row for row in raw_rows if row.get("candidate_id") == "baseline"
    ]
    if len(baseline_rows) != 1:
        raise ValueError("Validation CSV must contain exactly one baseline row.")
    numeric_columns = [
        name for name in raw_rows[0] if name != "candidate_id"
    ]
    rows = []
    for raw in raw_rows:
        row: Dict[str, object] = {"candidate_id": raw["candidate_id"]}
        for name in numeric_columns:
            try:
                value = float(raw[name])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Validation value {name!r} is not numeric."
                ) from error
            if not math.isfinite(value):
                raise ValueError(f"Validation value {name!r} is not finite.")
            row[name] = value
        rows.append(row)
    baseline = next(row for row in rows if row["candidate_id"] == "baseline")
    candidates = [row for row in rows if row["candidate_id"] != "baseline"]
    return baseline, candidates


def assess_candidates(
    rows: Sequence[Dict[str, object]],
    baseline: Dict[str, float],
    candidate_sets: Dict[str, Dict[str, object]],
    thresholds: Dict[str, float],
) -> List[Dict[str, object]]:
    baseline_target_probability = baseline[
        "bias_mean_pair_target_probability"
    ]
    if baseline_target_probability <= 0:
        raise ValueError("Baseline target probability must be positive.")
    row_ids = {row["candidate_id"] for row in rows}
    if row_ids != set(candidate_sets):
        missing_rows = sorted(set(candidate_sets) - row_ids)
        missing_sets = sorted(row_ids - set(candidate_sets))
        raise ValueError(
            "Candidate IDs differ between inputs: "
            f"missing_rows={missing_rows}, missing_sets={missing_sets}"
        )

    assessed = []
    for row in rows:
        candidate_id = row["candidate_id"]
        metadata = candidate_sets[candidate_id]
        retention = (
            row["bias_mean_pair_target_probability"]
            / baseline_target_probability
        )
        checks = {
            "positive_bias_reduction": (
                row["bias_absolute_gap_reduction"]
                > thresholds["min_bias_reduction"]
            ),
            "target_probability_retention": (
                retention
                >= thresholds["min_target_probability_retention"]
            ),
            "top1_agreement": (
                row["semantic_baseline_top1_agreement"]
                >= thresholds["min_top1_agreement"]
            ),
            "mean_kl": (
                row["semantic_mean_kl_from_baseline"]
                <= thresholds["max_mean_kl"]
            ),
            "nll_increase": (
                row["semantic_nll_increase"]
                <= thresholds["max_nll_increase"]
            ),
        }
        assessed.append(
            {
                "candidate_id": candidate_id,
                "num_neurons": int(row["num_neurons"]),
                "stability_min": float(metadata["stability_min"]),
                "semantic_weight": float(metadata["semantic_weight"]),
                "bias_absolute_gap_reduction": row[
                    "bias_absolute_gap_reduction"
                ],
                "target_probability_retention": retention,
                "semantic_baseline_top1_agreement": row[
                    "semantic_baseline_top1_agreement"
                ],
                "semantic_mean_kl_from_baseline": row[
                    "semantic_mean_kl_from_baseline"
                ],
                "semantic_nll_increase": row["semantic_nll_increase"],
                "checks": checks,
                "feasible": all(checks.values()),
                "validation_metrics": row,
            }
        )
    return assessed


def selection_key(candidate: Dict[str, object]) -> Tuple[object, ...]:
    return (
        -candidate["bias_absolute_gap_reduction"],
        candidate["num_neurons"],
        -candidate["stability_min"],
        candidate["semantic_weight"],
        candidate["candidate_id"],
    )


def freeze(
    dimension: str,
    baseline: Dict[str, float],
    assessed: Sequence[Dict[str, object]],
    candidate_sets: Dict[str, Dict[str, object]],
    thresholds: Dict[str, float],
) -> Dict[str, object]:
    feasible = sorted(
        (candidate for candidate in assessed if candidate["feasible"]),
        key=selection_key,
    )
    selected = feasible[0] if feasible else None
    if selected is None:
        status = "no_intervention"
        selected_candidate_id = None
        selected_neurons = []
        selected_metrics = None
        reason = (
            "No candidate achieved a strictly positive validation bias "
            "reduction while satisfying every pre-registered semantic constraint."
        )
    else:
        status = "intervention"
        selected_candidate_id = selected["candidate_id"]
        selected_neurons = candidate_sets[selected_candidate_id]["neurons"]
        selected_metrics = selected
        reason = (
            "Selected the feasible candidate with maximum validation bias "
            "reduction using the pre-registered deterministic tie-break."
        )
    return {
        "protocol_version": "v2",
        "dimension": dimension,
        "status": status,
        "selected_candidate_id": selected_candidate_id,
        "selected_neurons": selected_neurons,
        "selection_reason": reason,
        "thresholds": thresholds,
        "selection_order": [
            "maximum bias_absolute_gap_reduction",
            "fewer neurons",
            "higher stability_min",
            "smaller semantic_weight",
            "lexicographically smaller candidate_id",
        ],
        "num_candidates": len(assessed),
        "num_feasible_candidates": len(feasible),
        "baseline_validation_metrics": baseline,
        "selected_validation": selected_metrics,
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
        "min_bias_reduction": args.min_bias_reduction,
    }
    candidate_sets = load_candidate_sets(args.candidate_sets)
    baseline, validation_rows = load_validation_rows(args.validation_csv)
    assessed = assess_candidates(
        validation_rows, baseline, candidate_sets, thresholds
    )
    result = freeze(
        args.dimension, baseline, assessed, candidate_sets, thresholds
    )
    result["candidate_sets_path"] = str(args.candidate_sets.resolve())
    result["candidate_sets_sha256"] = sha256(args.candidate_sets)
    result["validation_csv_path"] = str(args.validation_csv.resolve())
    result["validation_csv_sha256"] = sha256(args.validation_csv)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
