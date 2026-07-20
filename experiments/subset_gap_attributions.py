#!/usr/bin/env python3
"""Subset a gap-attribution JSONL using paired bag/template split indices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply an audited paired-split manifest to an existing gap JSONL "
            "without recomputing integrated gradients."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--output-manifest", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_indices(
    manifest: Dict[str, object], split: str
) -> Tuple[List[int], List[int]]:
    splits = manifest.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise ValueError(f"Split {split!r} does not exist in the manifest.")
    split_record = splits[split]
    bag_partition = split_record.get("bag_partition")
    template_partition = split_record.get("template_partition")
    bag_partitions = manifest.get("bag_partitions")
    template_partitions = manifest.get("template_partitions")
    if (
        not isinstance(bag_partitions, dict)
        or bag_partition not in bag_partitions
        or not isinstance(template_partitions, dict)
        or template_partition not in template_partitions
    ):
        raise ValueError("Split partition references are invalid.")
    bag_indices = bag_partitions[bag_partition]
    template_indices = template_partitions[template_partition]
    if not all(isinstance(index, int) and index >= 0 for index in bag_indices):
        raise ValueError("Bag indices must be non-negative integers.")
    if not all(
        isinstance(index, int) and index >= 0 for index in template_indices
    ):
        raise ValueError("Template indices must be non-negative integers.")
    if len(set(bag_indices)) != len(bag_indices) or len(
        set(template_indices)
    ) != len(template_indices):
        raise ValueError("Split indices contain duplicates.")
    return list(bag_indices), list(template_indices)


def partition_size(partitions: object, label: str) -> int:
    if not isinstance(partitions, dict):
        raise ValueError(f"{label} partitions are absent.")
    flattened = [
        index
        for indices in partitions.values()
        for index in indices
    ]
    if len(flattened) != len(set(flattened)):
        raise ValueError(f"{label} partitions are not disjoint.")
    if set(flattened) != set(range(len(flattened))):
        raise ValueError(f"{label} partitions are not exhaustive contiguous indices.")
    return len(flattened)


def subset_gap(
    input_path: Path,
    split_manifest_path: Path,
    split: str,
    output_path: Path,
) -> Dict[str, object]:
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite output: {output_path}")
    with split_manifest_path.open("r", encoding="utf-8") as handle:
        split_manifest = json.load(handle)
    bag_indices, template_indices = resolve_indices(split_manifest, split)
    expected_bags = partition_size(
        split_manifest.get("bag_partitions"), "Bag"
    )
    expected_templates = partition_size(
        split_manifest.get("template_partitions"), "Template"
    )
    selected_bags = set(bag_indices)
    source_digest = hashlib.sha256()
    selected_original_indices = []
    source_bag_count = 0
    source_template_count = None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        raise FileExistsError(f"Refusing to overwrite partial output: {partial_path}")

    try:
        with input_path.open("rb") as source, partial_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as output:
            for bag_index, raw_line in enumerate(source):
                source_digest.update(raw_line)
                bag = json.loads(raw_line)
                if not isinstance(bag, list) or not bag:
                    raise ValueError(f"Source bag {bag_index} is empty or invalid.")
                if source_template_count is None:
                    source_template_count = len(bag)
                elif len(bag) != source_template_count:
                    raise ValueError("Source bags have inconsistent template counts.")
                if bag_index in selected_bags:
                    if max(template_indices) >= len(bag):
                        raise ValueError("A template index exceeds the source bag size.")
                    selected = [bag[index] for index in template_indices]
                    output.write(json.dumps(selected, separators=(",", ":")) + "\n")
                    selected_original_indices.append(bag_index)
                source_bag_count += 1
        if source_bag_count != expected_bags:
            raise ValueError(
                f"Source has {source_bag_count} bags, manifest expects {expected_bags}."
            )
        if source_template_count != expected_templates:
            raise ValueError(
                "Source has "
                f"{source_template_count} templates per bag, manifest expects "
                f"{expected_templates}."
            )
        if set(selected_original_indices) != selected_bags:
            raise ValueError("Not all selected bag indices were found in the source.")
        partial_path.replace(output_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise

    return {
        "source_path": str(input_path.resolve()),
        "source_sha256": source_digest.hexdigest(),
        "split_manifest_path": str(split_manifest_path.resolve()),
        "split_manifest_sha256": sha256(split_manifest_path),
        "split": split,
        "source_num_bags": source_bag_count,
        "source_templates_per_bag": source_template_count,
        "selected_bag_indices_source_order": selected_original_indices,
        "selected_template_indices_manifest_order": template_indices,
        "output_num_bags": len(selected_original_indices),
        "output_templates_per_bag": len(template_indices),
        "output_num_examples": len(selected_original_indices)
        * len(template_indices),
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.output_manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite output manifest: {manifest_path}"
        )
    summary = subset_gap(
        input_path=args.input,
        split_manifest_path=args.split_manifest,
        split=args.split,
        output_path=args.output,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
