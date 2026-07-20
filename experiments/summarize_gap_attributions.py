#!/usr/bin/env python3
"""Summarize sparse released gap attributions over the complete FFN index."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from array import array
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one row per FFN neuron from a filtered IG/IG2 gap JSONL. "
            "Absent sparse entries are explicitly treated as zero."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metric",
        default="auto",
        choices=("auto", "ig2_gold_gap", "ig_gold_gap"),
    )
    parser.add_argument("--num-layers", default=12, type=int)
    parser.add_argument("--neurons-per-layer", default=3072, type=int)
    parser.add_argument("--top-k", default=100, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relation_name(path: Path) -> str:
    name = path.name
    if "-base-" in name:
        name = name.split("-base-", maxsplit=1)[1]
    return name.split(".rlt.jsonl", maxsplit=1)[0]


def resolve_metric(result: Dict[str, object], requested: str) -> str:
    if requested != "auto":
        if requested not in result:
            raise ValueError(f"Metric {requested!r} is absent from the result.")
        return requested
    for candidate in ("ig2_gold_gap", "ig_gold_gap"):
        if candidate in result:
            return candidate
    raise ValueError("No supported gap attribution field exists in the input.")


def main() -> None:
    args = parse_args()
    if args.num_layers <= 0 or args.neurons_per_layer <= 0:
        raise ValueError("Model dimensions must be positive.")
    size = args.num_layers * args.neurons_per_layer

    sums = array("d", [0.0]) * size
    abs_sums = array("d", [0.0]) * size
    sum_squares = array("d", [0.0]) * size
    support = array("I", [0]) * size
    positive = array("I", [0]) * size
    negative = array("I", [0]) * size
    bag_support = array("I", [0]) * size

    metric = args.metric
    num_bags = 0
    num_examples = 0
    bag_sizes: List[int] = []

    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            bag = json.loads(line)
            if not isinstance(bag, list) or not bag:
                raise ValueError(f"Line {line_number} is not a non-empty result bag.")
            num_bags += 1
            bag_sizes.append(len(bag))
            seen_in_bag = set()
            for example in bag:
                result = example[1]
                if metric == "auto":
                    metric = resolve_metric(result, metric)
                triplets = result[metric]
                num_examples += 1
                seen_in_example = set()
                for triplet in triplets:
                    layer, neuron, raw_value = triplet
                    layer = int(layer)
                    neuron = int(neuron)
                    value = float(raw_value)
                    if not 0 <= layer < args.num_layers:
                        raise ValueError(f"Layer index out of range: {layer}")
                    if not 0 <= neuron < args.neurons_per_layer:
                        raise ValueError(f"Neuron index out of range: {neuron}")
                    if not math.isfinite(value):
                        raise ValueError("Attribution contains a non-finite value.")
                    index = layer * args.neurons_per_layer + neuron
                    if index in seen_in_example:
                        raise ValueError("A neuron occurs twice in one sparse example.")
                    seen_in_example.add(index)
                    seen_in_bag.add(index)
                    sums[index] += value
                    abs_sums[index] += abs(value)
                    sum_squares[index] += value * value
                    support[index] += 1
                    if value > 0:
                        positive[index] += 1
                    elif value < 0:
                        negative[index] += 1
            for index in seen_in_bag:
                bag_support[index] += 1

    if num_examples == 0:
        raise ValueError("The input contains no examples.")

    rows = []
    for index in range(size):
        layer, neuron = divmod(index, args.neurons_per_layer)
        count = support[index]
        mean_all = sums[index] / num_examples
        abs_mean_all = abs_sums[index] / num_examples
        variance = max(0.0, sum_squares[index] / num_examples - mean_all * mean_all)
        signed_count = positive[index] + negative[index]
        sign_consistency = (
            max(positive[index], negative[index]) / signed_count
            if signed_count
            else 0.0
        )
        rows.append(
            {
                "layer": layer,
                "neuron": neuron,
                "mean_zero_imputed": mean_all,
                "abs_mean_zero_imputed": abs_mean_all,
                "std_zero_imputed": math.sqrt(variance),
                "mean_when_retained": sums[index] / count if count else 0.0,
                "support_count": count,
                "support_rate": count / num_examples,
                "bag_support_count": bag_support[index],
                "bag_support_rate": bag_support[index] / num_bags,
                "positive_count": positive[index],
                "negative_count": negative[index],
                "sign_consistency": sign_consistency,
            }
        )

    ranked_indices = sorted(
        range(size),
        key=lambda index: (
            rows[index]["abs_mean_zero_imputed"],
            rows[index]["support_rate"],
            -rows[index]["layer"],
            -rows[index]["neuron"],
        ),
        reverse=True,
    )
    for rank, index in enumerate(ranked_indices, start=1):
        rows[index]["global_rank"] = rank

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "bias_attribution.csv"
    fieldnames = [
        "layer",
        "neuron",
        "global_rank",
        "mean_zero_imputed",
        "abs_mean_zero_imputed",
        "std_zero_imputed",
        "mean_when_retained",
        "support_count",
        "support_rate",
        "bag_support_count",
        "bag_support_rate",
        "positive_count",
        "negative_count",
        "sign_consistency",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    top_rows = [rows[index] for index in ranked_indices[: args.top_k]]
    summary = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "relation": relation_name(args.input),
        "metric": metric,
        "representation": (
            "released filtered sparse gap; absent layer-neuron entries are zero-imputed"
        ),
        "num_layers": args.num_layers,
        "neurons_per_layer": args.neurons_per_layer,
        "total_neurons": size,
        "num_bags": num_bags,
        "num_examples": num_examples,
        "bag_size_min": min(bag_sizes),
        "bag_size_max": max(bag_sizes),
        "retained_triplets": int(sum(support)),
        "neurons_with_support": sum(1 for count in support if count),
        "top_neurons": top_rows,
    }
    with (args.output_dir / "bias_attribution_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
