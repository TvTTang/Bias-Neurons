#!/usr/bin/env python3
"""Create paired lexical/template splits for bias-neuron experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split paired demographic data along both modifier-bag and prompt-template "
            "axes. Matching examples for both demographic groups always stay together."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--group1", required=True)
    parser.add_argument("--group2", required=True)
    parser.add_argument("--modifier", default="N")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--bag-ratios",
        default=(0.6, 0.2, 0.2),
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--template-ratios",
        default=(10 / 17, 3 / 17, 4 / 17),
        type=float,
        nargs=3,
        metavar=("TRAIN", "VAL", "TEST"),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_counts(total: int, ratios: Sequence[float]) -> Tuple[int, int, int]:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios):
        raise ValueError("Exactly three positive split ratios are required.")
    ratio_sum = sum(ratios)
    normalized = [ratio / ratio_sum for ratio in ratios]
    raw = [total * ratio for ratio in normalized]
    counts = [int(value) for value in raw]
    remaining = total - sum(counts)
    order = sorted(
        range(3), key=lambda index: (raw[index] - counts[index], -index), reverse=True
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]


def shuffled_partitions(
    total: int, ratios: Sequence[float], seed: int
) -> Dict[str, List[int]]:
    indices = list(range(total))
    random.Random(seed).shuffle(indices)
    train_count, val_count, test_count = allocate_counts(total, ratios)
    train_end = train_count
    val_end = train_end + val_count
    partitions = {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end : val_end + test_count],
    }
    flattened = [index for values in partitions.values() for index in values]
    if len(flattened) != total or len(set(flattened)) != total:
        raise AssertionError("Generated partitions are not disjoint and exhaustive.")
    return partitions


def load_pair(
    data_root: Path, dimension: str, group1: str, group2: str, modifier: str
) -> Tuple[Path, Path, List[object], List[object]]:
    directory = data_root / dimension
    path1 = directory / f"{group1}_{modifier}_data.json"
    path2 = directory / f"{group2}_{modifier}_data.json"
    with path1.open("r", encoding="utf-8") as handle:
        data1 = json.load(handle)
    with path2.open("r", encoding="utf-8") as handle:
        data2 = json.load(handle)
    return path1, path2, data1, data2


def validate_pair(data1: Sequence[object], data2: Sequence[object]) -> int:
    if len(data1) != len(data2) or not data1:
        raise ValueError("Demographic files must contain the same non-zero bag count.")
    template_count = len(data1[0])
    if template_count == 0:
        raise ValueError("Bags must contain at least one prompt template.")

    for bag_index, (bag1, bag2) in enumerate(zip(data1, data2)):
        if len(bag1) != template_count or len(bag2) != template_count:
            raise ValueError(f"Bag {bag_index} has an inconsistent template count.")
        for template_index, (example1, example2) in enumerate(zip(bag1, bag2)):
            if example1[0] != example2[0]:
                raise ValueError(
                    "Prompt mismatch at bag "
                    f"{bag_index}, template {template_index}: "
                    f"{example1[0]!r} != {example2[0]!r}"
                )
    return template_count


def subset(
    data: Sequence[object], bag_indices: Sequence[int], template_indices: Sequence[int]
) -> List[object]:
    return [
        [data[bag_index][template_index] for template_index in template_indices]
        for bag_index in bag_indices
    ]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )

    path1, path2, data1, data2 = load_pair(
        args.data_root, args.dimension, args.group1, args.group2, args.modifier
    )
    template_count = validate_pair(data1, data2)
    bag_parts = shuffled_partitions(len(data1), args.bag_ratios, args.seed)
    template_parts = shuffled_partitions(
        template_count, args.template_ratios, args.seed + 1
    )

    split_axes = {
        "train": ("train", "train"),
        "val": ("val", "val"),
        "test": ("test", "test"),
        "val_lexical_ood": ("val", "train"),
        "val_template_ood": ("train", "val"),
        "test_lexical_ood": ("test", "train"),
        "test_template_ood": ("train", "test"),
    }
    split_summary = {}
    for split_name, (bag_part, template_part) in split_axes.items():
        bag_indices = bag_parts[bag_part]
        template_indices = template_parts[template_part]
        split_root = args.output_dir / split_name / args.dimension
        write_json(
            split_root / f"{args.group1}_{args.modifier}_data.json",
            subset(data1, bag_indices, template_indices),
        )
        write_json(
            split_root / f"{args.group2}_{args.modifier}_data.json",
            subset(data2, bag_indices, template_indices),
        )
        split_summary[split_name] = {
            "bag_partition": bag_part,
            "template_partition": template_part,
            "num_bags": len(bag_indices),
            "templates_per_bag": len(template_indices),
            "examples_per_group": len(bag_indices) * len(template_indices),
        }

    manifest = {
        "dimension": args.dimension,
        "groups": [args.group1, args.group2],
        "modifier": args.modifier,
        "seed": args.seed,
        "bag_ratios": list(args.bag_ratios),
        "template_ratios": list(args.template_ratios),
        "source_files": {
            args.group1: {
                "path": str(path1.resolve()),
                "sha256": file_sha256(path1),
            },
            args.group2: {
                "path": str(path2.resolve()),
                "sha256": file_sha256(path2),
            },
        },
        "bag_partitions": bag_parts,
        "template_partitions": template_parts,
        "splits": split_summary,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
