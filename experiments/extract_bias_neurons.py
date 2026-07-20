#!/usr/bin/env python3
"""Extract relation-level bias neurons from an IG/IG2 gap JSONL file.

This is a parameterized, streaming-compatible replacement for the hard-coded
``bias_neuron_src/2_get_bn_bias_*.py`` scripts.  Its defaults reproduce the
released post-processing rule without changing the underlying algorithm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


Position = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract bag- and relation-level bias neurons from gap JSONL."
    )
    parser.add_argument("--input", required=True, type=Path, help="Gap JSONL file.")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Directory for JSON summaries."
    )
    parser.add_argument(
        "--metric",
        default="auto",
        choices=("auto", "ig2_gold_gap", "ig_gold_gap"),
        help="Gap field. 'auto' supports both released naming variants.",
    )
    parser.add_argument("--threshold-ratio", type=float, default=0.2)
    parser.add_argument("--mode-ratio-bag", type=float, default=0.7)
    parser.add_argument("--mode-ratio-rel", type=float, default=0.1)
    parser.add_argument("--min-bag-count", type=int, default=3)
    parser.add_argument("--adaptive-rounds", type=int, default=6)
    parser.add_argument("--target-min", type=float, default=2.0)
    parser.add_argument("--target-max", type=float, default=5.0)
    return parser.parse_args()


def position_key(position: Position) -> str:
    return f"{position[0]}@{position[1]}"


def parse_position(key: str) -> Position:
    layer, neuron = key.split("@", maxsplit=1)
    return int(layer), int(neuron)


def select_by_frequency(
    counts: Counter[str],
    total: int,
    ratio: float,
    minimum: int = 0,
) -> List[Position]:
    threshold = max(total * ratio, minimum)
    return [
        parse_position(key)
        for key, count in counts.items()
        if count >= threshold
    ]


def resolve_metric(result: Dict[str, object], requested: str) -> str:
    if requested != "auto":
        if requested not in result:
            raise ValueError(f"Metric {requested!r} is absent from the input result.")
        return requested
    for candidate in ("ig2_gold_gap", "ig_gold_gap"):
        if candidate in result:
            return candidate
    raise ValueError("Neither 'ig2_gold_gap' nor 'ig_gold_gap' exists in the input.")


def filtered_positions(
    triplets: Sequence[Sequence[object]], threshold_ratio: float
) -> Iterable[Position]:
    if not triplets:
        return ()
    values = [float(triplet[2]) for triplet in triplets]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Attribution triplets contain non-finite values.")
    threshold = max(values) * threshold_ratio
    return (
        (int(triplet[0]), int(triplet[1]))
        for triplet, value in zip(triplets, values)
        if value >= threshold
    )


def read_bag_counts(
    path: Path, metric: str, threshold_ratio: float
) -> Tuple[List[Counter[str]], List[int], str]:
    bag_counts: List[Counter[str]] = []
    bag_sizes: List[int] = []
    resolved_metric = metric

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            bag = json.loads(line)
            if not isinstance(bag, list) or not bag:
                raise ValueError(f"Line {line_number} is not a non-empty result bag.")
            counts: Counter[str] = Counter()
            for example in bag:
                result = example[1]
                if resolved_metric == "auto":
                    resolved_metric = resolve_metric(result, resolved_metric)
                triplets = result[resolved_metric]
                counts.update(
                    position_key(position)
                    for position in filtered_positions(triplets, threshold_ratio)
                )
            bag_counts.append(counts)
            bag_sizes.append(len(bag))

    if not bag_counts:
        raise ValueError("The input JSONL contains no result bags.")
    return bag_counts, bag_sizes, resolved_metric


def choose_bag_neurons(
    bag_counts: Sequence[Counter[str]],
    bag_sizes: Sequence[int],
    initial_ratio: float,
    minimum: int,
    adaptive_rounds: int,
    target_min: float,
    target_max: float,
) -> Tuple[
    float,
    float,
    List[List[Position]],
    bool,
    List[Dict[str, float]],
]:
    ratio = initial_ratio
    bag_neurons: List[List[Position]] = []
    average = 0.0
    trajectory = []

    for round_index in range(adaptive_rounds):
        bag_neurons = [
            select_by_frequency(counts, size, ratio, minimum)
            for counts, size in zip(bag_counts, bag_sizes)
        ]
        average = sum(map(len, bag_neurons)) / len(bag_neurons)
        trajectory.append(
            {
                "round": round_index + 1,
                "mode_ratio_bag": ratio,
                "average_neurons_per_bag": average,
            }
        )
        if average < target_min:
            ratio -= 0.05
        elif average > target_max:
            ratio += 0.05
        else:
            break
    converged = target_min <= average <= target_max
    selected_ratio = trajectory[-1]["mode_ratio_bag"]
    return selected_ratio, average, bag_neurons, converged, trajectory


def relation_name(path: Path) -> str:
    name = path.name
    if "-base-" in name:
        name = name.split("-base-", maxsplit=1)[1]
    return name.split(".rlt.jsonl", maxsplit=1)[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not 0 <= args.threshold_ratio <= 1:
        raise ValueError("--threshold-ratio must be in [0, 1].")
    if not 0 <= args.mode_ratio_bag <= 1 or not 0 <= args.mode_ratio_rel <= 1:
        raise ValueError("Mode ratios must be in [0, 1].")

    bag_counts, bag_sizes, metric = read_bag_counts(
        args.input, args.metric, args.threshold_ratio
    )
    bag_ratio, average, bag_neurons, converged, trajectory = choose_bag_neurons(
        bag_counts=bag_counts,
        bag_sizes=bag_sizes,
        initial_ratio=args.mode_ratio_bag,
        minimum=args.min_bag_count,
        adaptive_rounds=args.adaptive_rounds,
        target_min=args.target_min,
        target_max=args.target_max,
    )

    relation_counts: Counter[str] = Counter(
        position_key(position)
        for neurons in bag_neurons
        for position in neurons
    )
    relation_neurons = select_by_frequency(
        relation_counts, len(bag_neurons), args.mode_ratio_rel
    )
    relation_neurons.sort()
    layer_counts = Counter(layer for layer, _ in relation_neurons)

    relation = relation_name(args.input)
    summary = {
        "source": str(args.input.resolve()),
        "source_sha256": sha256(args.input),
        "relation": relation,
        "metric": metric,
        "num_bags": len(bag_neurons),
        "bag_size_min": min(bag_sizes),
        "bag_size_max": max(bag_sizes),
        "threshold_ratio": args.threshold_ratio,
        "mode_ratio_bag_initial": args.mode_ratio_bag,
        "mode_ratio_bag_selected": bag_ratio,
        "mode_ratio_rel": args.mode_ratio_rel,
        "min_bag_count": args.min_bag_count,
        "target_neurons_per_bag_min": args.target_min,
        "target_neurons_per_bag_max": args.target_max,
        "selection_converged": converged,
        "selection_trajectory": trajectory,
        "average_neurons_per_bag": average,
        "relation_neuron_count": len(relation_neurons),
        "relation_neurons": relation_neurons,
        "layer_counts": dict(sorted(layer_counts.items())),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        f"bn_bag-{relation}.json": bag_neurons,
        f"bn_rel-{relation}.json": relation_neurons,
        f"summary-{relation}.json": summary,
    }
    for filename, value in outputs.items():
        with (args.output_dir / filename).open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
