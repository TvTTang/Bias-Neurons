#!/usr/bin/env python3
"""Create bias/semantic multi-objective neuron candidate sets."""

from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Neuron = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank FFN neurons by bias utility minus a semantic-importance "
            "penalty and emit a validation-tunable candidate grid."
        )
    )
    parser.add_argument("--bias-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--bias-name", nargs="+", required=True)
    parser.add_argument("--semantic-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--semantic-weight",
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
        type=float,
    )
    parser.add_argument(
        "--num-neurons",
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64),
        type=int,
    )
    parser.add_argument(
        "--bias-aggregation",
        choices=("mean", "min"),
        default="mean",
        help="How to combine normalized bias utilities across dimensions.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_metric_csv(path: Path, metric: str) -> Dict[Neuron, float]:
    values = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "neuron", metric}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} lacks required columns: {sorted(required)}")
        for row in reader:
            key = (int(row["layer"]), int(row["neuron"]))
            if key in values:
                raise ValueError(f"Duplicate neuron {key} in {path}.")
            value = float(row[metric])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Metric {metric} must be finite and non-negative.")
            values[key] = value
    if not values:
        raise ValueError(f"No neuron rows found in {path}.")
    return values


def zero_origin_percentiles(values: Dict[Neuron, float]) -> Dict[Neuron, float]:
    """Map the minimum/tied-zero group to 0 and the maximum to at most 1."""
    ordered = sorted(values.values())
    denominator = max(1, len(ordered) - 1)
    return {
        key: bisect.bisect_left(ordered, value) / denominator
        for key, value in values.items()
    }


def validate_same_neurons(
    named_values: Sequence[Tuple[str, Dict[Neuron, float]]]
) -> List[Neuron]:
    reference_name, reference = named_values[0]
    reference_keys = set(reference)
    for name, values in named_values[1:]:
        if set(values) != reference_keys:
            missing = len(reference_keys - set(values))
            extra = len(set(values) - reference_keys)
            raise ValueError(
                f"Neuron index mismatch for {name} relative to {reference_name}: "
                f"missing={missing}, extra={extra}."
            )
    return sorted(reference_keys)


def pareto_frontier(rows: Sequence[Dict[str, object]]) -> set:
    """Return neurons not dominated by higher bias and lower semantic cost."""
    ordered = sorted(
        rows,
        key=lambda row: (
            -row["bias_utility"],
            row["semantic_cost"],
            row["layer"],
            row["neuron"],
        ),
    )
    frontier = set()
    best_cost_at_strictly_higher_bias = float("inf")
    index = 0
    while index < len(ordered):
        bias_utility = ordered[index]["bias_utility"]
        group = []
        while (
            index < len(ordered)
            and ordered[index]["bias_utility"] == bias_utility
        ):
            group.append(ordered[index])
            index += 1
        group_best_cost = min(row["semantic_cost"] for row in group)
        if group_best_cost < best_cost_at_strictly_higher_bias:
            for row in group:
                if row["semantic_cost"] == group_best_cost:
                    frontier.add((row["layer"], row["neuron"]))
        best_cost_at_strictly_higher_bias = min(
            best_cost_at_strictly_higher_bias, group_best_cost
        )
    return frontier


def build_rows(
    bias_values: Sequence[Tuple[str, Dict[Neuron, float]]],
    semantic_values: Dict[Neuron, float],
    aggregation: str,
) -> List[Dict[str, object]]:
    keys = validate_same_neurons(
        [*bias_values, ("semantic", semantic_values)]
    )
    normalized_bias = [
        (name, zero_origin_percentiles(values))
        for name, values in bias_values
    ]
    normalized_semantic = zero_origin_percentiles(semantic_values)
    rows = []
    for layer, neuron in keys:
        key = (layer, neuron)
        components = [values[key] for _, values in normalized_bias]
        if aggregation == "mean":
            bias_utility = sum(components) / len(components)
        elif aggregation == "min":
            bias_utility = min(components)
        else:
            raise ValueError(f"Unsupported bias aggregation: {aggregation}")
        row = {
            "layer": layer,
            "neuron": neuron,
            "bias_utility": bias_utility,
            "semantic_cost": normalized_semantic[key],
            "semantic_mean_abs_ig": semantic_values[key],
        }
        for name, raw_values in bias_values:
            row[f"bias_raw_{name}"] = raw_values[key]
        for name, normalized_values in normalized_bias:
            row[f"bias_percentile_{name}"] = normalized_values[key]
        rows.append(row)
    frontier = pareto_frontier(rows)
    for row in rows:
        row["pareto_frontier"] = (
            row["layer"], row["neuron"]
        ) in frontier
    return rows


def candidate_grid(
    rows: Sequence[Dict[str, object]],
    semantic_weights: Iterable[float],
    neuron_counts: Iterable[int],
) -> List[Dict[str, object]]:
    candidates = []
    for semantic_weight in semantic_weights:
        if semantic_weight < 0:
            raise ValueError("Semantic weights must be non-negative.")
        ranked = sorted(
            rows,
            key=lambda row: (
                row["bias_utility"]
                - semantic_weight * row["semantic_cost"],
                row["bias_utility"],
                -row["semantic_cost"],
                -row["layer"],
                -row["neuron"],
            ),
            reverse=True,
        )
        for count in neuron_counts:
            if count <= 0 or count > len(rows):
                raise ValueError("Requested neuron counts are out of range.")
            selected = ranked[:count]
            candidate_id = (
                f"lambda-{semantic_weight:g}_k-{count}"
                .replace(".", "p")
            )
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "semantic_weight": semantic_weight,
                    "num_neurons": count,
                    "mean_bias_utility": sum(
                        row["bias_utility"] for row in selected
                    )
                    / count,
                    "mean_semantic_cost": sum(
                        row["semantic_cost"] for row in selected
                    )
                    / count,
                    "pareto_fraction": sum(
                        bool(row["pareto_frontier"]) for row in selected
                    )
                    / count,
                    "neurons": [
                        [row["layer"], row["neuron"]] for row in selected
                    ],
                }
            )
    return candidates


def main() -> None:
    args = parse_args()
    if len(args.bias_csv) != len(args.bias_name):
        raise ValueError("--bias-csv and --bias-name must have equal lengths.")
    if len(set(args.bias_name)) != len(args.bias_name):
        raise ValueError("--bias-name values must be unique.")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )

    bias_values = [
        (name, load_metric_csv(path, "abs_mean_zero_imputed"))
        for name, path in zip(args.bias_name, args.bias_csv)
    ]
    semantic_values = load_metric_csv(
        args.semantic_csv, "semantic_mean_abs_ig"
    )
    rows = build_rows(
        bias_values=bias_values,
        semantic_values=semantic_values,
        aggregation=args.bias_aggregation,
    )
    candidates = candidate_grid(
        rows,
        semantic_weights=args.semantic_weight,
        neuron_counts=args.num_neurons,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "joint_neuron_scores.csv"
    fixed_fields = [
        "layer",
        "neuron",
        "bias_utility",
        "semantic_cost",
        "semantic_mean_abs_ig",
        "pareto_frontier",
    ]
    dynamic_fields = [
        f"{prefix}_{name}"
        for prefix in ("bias_raw", "bias_percentile")
        for name in args.bias_name
    ]
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[*fixed_fields, *dynamic_fields]
        )
        writer.writeheader()
        writer.writerows(rows)

    candidates_path = args.output_dir / "candidate_sets.json"
    with candidates_path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest = {
        "method": (
            "rank-normalized bias utility minus lambda times rank-normalized "
            "neutral semantic importance"
        ),
        "selection_protocol": (
            "choose lambda and num_neurons on validation only; evaluate the "
            "chosen candidate once on test"
        ),
        "bias_aggregation": args.bias_aggregation,
        "bias_sources": [
            {
                "name": name,
                "path": str(path.resolve()),
                "sha256": sha256(path),
            }
            for name, path in zip(args.bias_name, args.bias_csv)
        ],
        "semantic_source": {
            "path": str(args.semantic_csv.resolve()),
            "sha256": sha256(args.semantic_csv),
        },
        "total_neurons": len(rows),
        "pareto_frontier_size": sum(
            bool(row["pareto_frontier"]) for row in rows
        ),
        "semantic_weights": list(args.semantic_weight),
        "neuron_counts": list(args.num_neurons),
        "num_candidates": len(candidates),
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": sha256(scores_path),
        "candidates_path": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
    }
    with (args.output_dir / "manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
