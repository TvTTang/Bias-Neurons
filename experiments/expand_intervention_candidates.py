#!/usr/bin/env python3
"""Expand neuron sets into V3 direction-and-strength candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create V3 candidates by pairing train-selected neuron sets with "
            "pre-registered non-negative activation scales."
        )
    )
    parser.add_argument("--candidate-sets", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--intervention-scale",
        nargs="+",
        default=(0.0, 0.25, 0.5, 0.75, 1.25, 1.5, 2.0),
        type=float,
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_number(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def load_candidates(path: Path) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        candidates = json.load(handle)
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Candidate file must contain a non-empty JSON list.")
    return candidates


def canonical_neurons(candidate: Dict[str, object]) -> Tuple[Tuple[int, int], ...]:
    neurons = candidate.get("neurons")
    if not isinstance(neurons, list) or not neurons:
        raise ValueError("Every source candidate must contain neurons.")
    parsed = []
    for neuron in neurons:
        if (
            not isinstance(neuron, list)
            or len(neuron) != 2
            or not all(isinstance(index, int) for index in neuron)
        ):
            raise ValueError("Invalid neuron index.")
        parsed.append((neuron[0], neuron[1]))
    if len(set(parsed)) != len(parsed):
        raise ValueError("A source candidate contains duplicate neurons.")
    return tuple(parsed)


def unique_source_candidates(
    candidates: Sequence[Dict[str, object]],
) -> List[Dict[str, object]]:
    """Keep one deterministic representative for each ordered neuron set."""
    representatives = {}
    for candidate in candidates:
        neurons = canonical_neurons(candidate)
        key = (
            -float(candidate["stability_min"]),
            float(candidate["semantic_weight"]),
            str(candidate["candidate_id"]),
        )
        previous = representatives.get(neurons)
        if previous is None or key < previous[0]:
            representatives[neurons] = (key, candidate)
    return [
        value[1]
        for _, value in sorted(
            representatives.items(),
            key=lambda item: (
                len(item[0]),
                item[1][0],
                item[0],
            ),
        )
    ]


def expand_candidates(
    candidates: Sequence[Dict[str, object]],
    scales: Sequence[float],
) -> List[Dict[str, object]]:
    if not scales:
        raise ValueError("At least one intervention scale is required.")
    normalized_scales = []
    for scale in scales:
        scale = float(scale)
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("Intervention scales must be finite and non-negative.")
        if scale == 1:
            raise ValueError("Scale 1 is the baseline and must not be a candidate.")
        if scale not in normalized_scales:
            normalized_scales.append(scale)
    expanded = []
    for candidate in unique_source_candidates(candidates):
        for scale in normalized_scales:
            value = dict(candidate)
            value["source_candidate_id"] = candidate["candidate_id"]
            value["candidate_id"] = (
                f"{candidate['candidate_id']}_scale-{format_number(scale)}"
            )
            value["intervention_scale"] = scale
            expanded.append(value)
    return expanded


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    source = load_candidates(args.candidate_sets)
    candidates = expand_candidates(source, args.intervention_scale)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "candidate_sets.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(candidates, handle, indent=2, sort_keys=True)
        handle.write("\n")
    manifest = {
        "method": (
            "pair each unique V2 train-selected neuron set with a "
            "pre-registered non-negative activation scale"
        ),
        "source_candidate_sets": str(args.candidate_sets.resolve()),
        "source_candidate_sets_sha256": sha256(args.candidate_sets),
        "intervention_scales": list(args.intervention_scale),
        "num_source_candidate_ids": len(source),
        "num_unique_neuron_sets": len(unique_source_candidates(source)),
        "num_candidates": len(candidates),
        "candidate_sets": str(output_path.resolve()),
        "candidate_sets_sha256": sha256(output_path),
        "selection_protocol": (
            "evaluate on IID, lexical-OOD, and template-OOD validation "
            "environments; freeze before any new sealed evaluation"
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
