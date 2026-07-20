#!/usr/bin/env python3
"""Select semantic-safe bias neurons with bag/template bootstrap stability."""

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
            "Build v2 neuron candidates from train-only bag/template bootstrap "
            "stability, support, sign consistency, and neutral semantic cost."
        )
    )
    parser.add_argument("--gap-jsonl", required=True, type=Path)
    parser.add_argument("--semantic-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metric",
        default="auto",
        choices=("auto", "ig2_gold_gap", "ig_gold_gap"),
    )
    parser.add_argument("--num-layers", default=12, type=int)
    parser.add_argument("--neurons-per-layer", default=3072, type=int)
    parser.add_argument("--bootstrap-rounds", default=200, type=int)
    parser.add_argument("--bootstrap-top-m", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--stability-min",
        nargs="+",
        default=(0.25, 0.5, 0.75),
        type=float,
    )
    parser.add_argument(
        "--semantic-weight",
        nargs="+",
        default=(0.0, 0.25, 0.5, 1.0, 2.0),
        type=float,
    )
    parser.add_argument(
        "--num-neurons",
        nargs="+",
        default=(1, 2, 4, 8, 16, 32),
        type=int,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_metric(result: Dict[str, object], requested: str) -> str:
    if requested != "auto":
        if requested not in result:
            raise ValueError(f"Metric {requested!r} is absent.")
        return requested
    for candidate in ("ig2_gold_gap", "ig_gold_gap"):
        if candidate in result:
            return candidate
    raise ValueError("No supported gap metric exists.")


def load_gap_tensor(
    path: Path,
    metric: str,
    num_layers: int,
    neurons_per_layer: int,
    np: object,
) -> Tuple[object, str]:
    total_neurons = num_layers * neurons_per_layer
    bags = []
    resolved_metric = metric
    template_count = None
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            bag = json.loads(line)
            if not isinstance(bag, list) or not bag:
                raise ValueError(f"Line {line_number} is not a valid bag.")
            if template_count is None:
                template_count = len(bag)
            elif len(bag) != template_count:
                raise ValueError("Bags have inconsistent template counts.")
            values = np.zeros((len(bag), total_neurons), dtype=np.float32)
            for template_index, example in enumerate(bag):
                result = example[1]
                if resolved_metric == "auto":
                    resolved_metric = resolve_metric(result, resolved_metric)
                seen = set()
                for layer, neuron, raw_value in result[resolved_metric]:
                    layer = int(layer)
                    neuron = int(neuron)
                    value = float(raw_value)
                    if not math.isfinite(value):
                        raise ValueError("Gap attribution is not finite.")
                    if not 0 <= layer < num_layers or not (
                        0 <= neuron < neurons_per_layer
                    ):
                        raise ValueError("Neuron index is out of range.")
                    index = layer * neurons_per_layer + neuron
                    if index in seen:
                        raise ValueError("Neuron occurs twice in one example.")
                    seen.add(index)
                    values[template_index, index] = value
            bags.append(values)
    if not bags:
        raise ValueError("Gap file contains no bags.")
    return np.stack(bags, axis=0), resolved_metric


def direction_strength(unit_means: object, np: object) -> object:
    positive = (unit_means > 0).sum(axis=0)
    negative = (unit_means < 0).sum(axis=0)
    signed = positive + negative
    return np.divide(
        np.abs(positive - negative),
        signed,
        out=np.zeros_like(signed, dtype=np.float64),
        where=signed > 0,
    )


def zero_origin_percentiles(values: object, np: object) -> object:
    ordered = np.sort(values)
    denominator = max(1, len(values) - 1)
    return np.searchsorted(ordered, values, side="left") / denominator


def bootstrap_top_frequency(
    unit_means: object,
    rounds: int,
    top_m: int,
    seed: int,
    np: object,
) -> object:
    if rounds <= 0:
        raise ValueError("Bootstrap rounds must be positive.")
    units, neurons = unit_means.shape
    if not 0 < top_m <= neurons:
        raise ValueError("bootstrap-top-m is out of range.")
    rng = np.random.RandomState(seed)
    counts = np.zeros(neurons, dtype=np.int32)
    for _ in range(rounds):
        sampled = rng.randint(0, units, size=units)
        scores = np.abs(unit_means[sampled].mean(axis=0))
        selected = np.argpartition(scores, neurons - top_m)[-top_m:]
        counts[selected] += 1
    return counts / rounds


def load_semantic(path: Path) -> Dict[Tuple[int, int], float]:
    values = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"layer", "neuron", "semantic_mean_abs_ig"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("Semantic CSV lacks required columns.")
        for row in reader:
            key = (int(row["layer"]), int(row["neuron"]))
            if key in values:
                raise ValueError(f"Duplicate semantic neuron: {key}")
            value = float(row["semantic_mean_abs_ig"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("Semantic importance must be finite and non-negative.")
            values[key] = value
    return values


def candidate_grid(
    rows: Sequence[Dict[str, object]],
    stability_thresholds: Sequence[float],
    semantic_weights: Sequence[float],
    neuron_counts: Sequence[int],
) -> List[Dict[str, object]]:
    candidates = []
    for stability_min in stability_thresholds:
        if not 0 <= stability_min <= 1:
            raise ValueError("Stability thresholds must be in [0, 1].")
        eligible = [
            row
            for row in rows
            if row["dual_axis_stability"] >= stability_min
            and row["stable_bias_utility"] > 0
        ]
        for semantic_weight in semantic_weights:
            if semantic_weight < 0:
                raise ValueError("Semantic weights must be non-negative.")
            ranked = sorted(
                eligible,
                key=lambda row: (
                    row["stable_bias_utility"]
                    - semantic_weight * row["semantic_cost"],
                    row["stable_bias_utility"],
                    -row["semantic_cost"],
                    -row["layer"],
                    -row["neuron"],
                ),
                reverse=True,
            )
            for count in neuron_counts:
                if count <= 0:
                    raise ValueError("Neuron counts must be positive.")
                if count > len(ranked):
                    continue
                selected = ranked[:count]
                candidate_id = (
                    f"stability-{stability_min:g}_lambda-{semantic_weight:g}_k-{count}"
                    .replace(".", "p")
                )
                candidates.append(
                    {
                        "candidate_id": candidate_id,
                        "stability_min": stability_min,
                        "semantic_weight": semantic_weight,
                        "num_neurons": count,
                        "mean_stable_bias_utility": sum(
                            row["stable_bias_utility"] for row in selected
                        )
                        / count,
                        "mean_semantic_cost": sum(
                            row["semantic_cost"] for row in selected
                        )
                        / count,
                        "mean_dual_axis_stability": sum(
                            row["dual_axis_stability"] for row in selected
                        )
                        / count,
                        "neurons": [
                            [row["layer"], row["neuron"]] for row in selected
                        ],
                    }
                )
    if not candidates:
        raise ValueError("No v2 candidate satisfies the stability grid.")
    return candidates


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    import numpy as np

    tensor, metric = load_gap_tensor(
        args.gap_jsonl,
        args.metric,
        args.num_layers,
        args.neurons_per_layer,
        np,
    )
    num_bags, num_templates, total_neurons = tensor.shape
    bag_means = tensor.mean(axis=1)
    template_means = tensor.mean(axis=0)
    overall_mean = tensor.mean(axis=(0, 1))
    overall_mean_abs = np.abs(tensor).mean(axis=(0, 1))
    bag_support = (tensor != 0).any(axis=1).mean(axis=0)
    bag_direction = direction_strength(bag_means, np)
    template_direction = direction_strength(template_means, np)
    bag_frequency = bootstrap_top_frequency(
        bag_means,
        args.bootstrap_rounds,
        args.bootstrap_top_m,
        args.seed,
        np,
    )
    template_frequency = bootstrap_top_frequency(
        template_means,
        args.bootstrap_rounds,
        args.bootstrap_top_m,
        args.seed + 1,
        np,
    )
    dual_stability = np.minimum(bag_frequency, template_frequency)
    bias_percentile = zero_origin_percentiles(np.abs(overall_mean), np)

    semantic = load_semantic(args.semantic_csv)
    expected_keys = {
        divmod(index, args.neurons_per_layer) for index in range(total_neurons)
    }
    if set(semantic) != expected_keys:
        raise ValueError("Semantic CSV neuron index does not match the gap tensor.")
    semantic_raw = np.array(
        [
            semantic[divmod(index, args.neurons_per_layer)]
            for index in range(total_neurons)
        ],
        dtype=np.float64,
    )
    semantic_cost = zero_origin_percentiles(semantic_raw, np)
    stable_bias_utility = (
        bias_percentile
        * np.sqrt(bag_support)
        * bag_direction
        * template_direction
        * dual_stability
    )

    rows = []
    for index in range(total_neurons):
        layer, neuron = divmod(index, args.neurons_per_layer)
        rows.append(
            {
                "layer": layer,
                "neuron": neuron,
                "mean_signed_gap": float(overall_mean[index]),
                "mean_abs_gap": float(overall_mean_abs[index]),
                "bias_percentile": float(bias_percentile[index]),
                "bag_support_rate": float(bag_support[index]),
                "bag_direction_strength": float(bag_direction[index]),
                "template_direction_strength": float(template_direction[index]),
                "bag_bootstrap_frequency": float(bag_frequency[index]),
                "template_bootstrap_frequency": float(template_frequency[index]),
                "dual_axis_stability": float(dual_stability[index]),
                "stable_bias_utility": float(stable_bias_utility[index]),
                "semantic_mean_abs_ig": float(semantic_raw[index]),
                "semantic_cost": float(semantic_cost[index]),
            }
        )
    candidates = candidate_grid(
        rows,
        args.stability_min,
        args.semantic_weight,
        args.num_neurons,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_path = args.output_dir / "stable_neuron_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    candidates_path = args.output_dir / "candidate_sets.json"
    with candidates_path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "method": (
            "train-only bias percentile times sqrt bag support times bag/template "
            "direction strength times minimum bag/template bootstrap frequency, "
            "minus lambda times neutral semantic percentile"
        ),
        "gap_jsonl": str(args.gap_jsonl.resolve()),
        "gap_sha256": sha256(args.gap_jsonl),
        "semantic_csv": str(args.semantic_csv.resolve()),
        "semantic_sha256": sha256(args.semantic_csv),
        "metric": metric,
        "num_bags": num_bags,
        "num_templates": num_templates,
        "total_neurons": total_neurons,
        "bootstrap_rounds": args.bootstrap_rounds,
        "bootstrap_top_m": args.bootstrap_top_m,
        "seed": args.seed,
        "stability_thresholds": list(args.stability_min),
        "semantic_weights": list(args.semantic_weight),
        "neuron_counts": list(args.num_neurons),
        "num_candidates": len(candidates),
        "neurons_with_nonzero_stable_utility": int(
            (stable_bias_utility > 0).sum()
        ),
        "scores_path": str(scores_path.resolve()),
        "scores_sha256": sha256(scores_path),
        "candidates_path": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "selection_protocol": (
            "choose stability_min, lambda, and K on validation only; freeze "
            "before any v2 test/OOD evaluation"
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
