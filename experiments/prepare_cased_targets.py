#!/usr/bin/env python3
"""Prepare auditable target-case corrections for a cased MLM tokenizer."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Correct only MLM gold target strings while preserving every prompt, "
            "bag, template, and relation label."
        )
    )
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument(
        "--group",
        nargs="+",
        required=True,
        metavar="FILE_LABEL=TARGET_TOKEN",
    )
    parser.add_argument("--modifier", default="N")
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_group_mappings(values: Sequence[str]) -> List[Tuple[str, str]]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid group mapping {value!r}; expected old=new.")
        source, target = value.split("=", maxsplit=1)
        if not source or not target:
            raise ValueError(f"Invalid empty group mapping: {value!r}.")
        mappings.append((source, target))
    if len({source for source, _ in mappings}) != len(mappings):
        raise ValueError("Source group labels must be unique.")
    return mappings


def replace_targets(
    bags: Sequence[object], expected_target: str, corrected_target: str
) -> List[object]:
    corrected = deepcopy(bags)
    count = 0
    for bag_index, bag in enumerate(corrected):
        if not isinstance(bag, list) or not bag:
            raise ValueError(f"Bag {bag_index} is empty or invalid.")
        for template_index, example in enumerate(bag):
            if not isinstance(example, list) or len(example) < 3:
                raise ValueError(
                    f"Example {bag_index}/{template_index} is invalid."
                )
            if example[1] != expected_target:
                raise ValueError(
                    f"Unexpected target {example[1]!r} at "
                    f"{bag_index}/{template_index}; expected "
                    f"{expected_target!r}."
                )
            example[1] = corrected_target
            count += 1
    if count == 0:
        raise ValueError("No targets were corrected.")
    return corrected


def validate_single_token(tokenizer: object, target: str) -> int:
    tokens = tokenizer.tokenize(target)
    target_id = tokenizer.convert_tokens_to_ids(target)
    if tokens != [target]:
        raise ValueError(
            f"Corrected target {target!r} is not preserved as one WordPiece: {tokens}."
        )
    if target_id == tokenizer.unk_token_id:
        raise ValueError(f"Corrected target {target!r} maps to [UNK].")
    return int(target_id)


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(
            f"Output root is not empty; refusing to overwrite: {args.output_root}"
        )
    mappings = parse_group_mappings(args.group)

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        str(args.tokenizer_path), local_files_only=True
    )
    target_ids: Dict[str, int] = {}
    for _, corrected_target in mappings:
        target_ids[corrected_target] = validate_single_token(
            tokenizer, corrected_target
        )
    if len(set(target_ids.values())) != len(target_ids):
        raise ValueError("Corrected demographic targets map to duplicate token IDs.")

    records = []
    for source_group, corrected_target in mappings:
        source_path = (
            args.data_root
            / args.dimension
            / f"{source_group}_{args.modifier}_data.json"
        )
        with source_path.open("r", encoding="utf-8") as handle:
            source_data = json.load(handle)
        corrected_data = replace_targets(
            source_data, source_group, corrected_target
        )
        output_path = (
            args.output_root
            / args.dimension
            / f"{source_group}_{args.modifier}_data.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(corrected_data, handle, indent=2)
            handle.write("\n")
        records.append(
            {
                "source_group": source_group,
                "corrected_target": corrected_target,
                "corrected_target_id": target_ids[corrected_target],
                "source_path": str(source_path.resolve()),
                "source_sha256": sha256(source_path),
                "output_path": str(output_path.resolve()),
                "output_sha256": sha256(output_path),
                "num_bags": len(corrected_data),
                "templates_per_bag": len(corrected_data[0]),
                "num_examples": sum(len(bag) for bag in corrected_data),
            }
        )

    manifest = {
        "operation": (
            "gold-target case correction only; prompts, bags, templates, and "
            "relation labels are unchanged"
        ),
        "dimension": args.dimension,
        "modifier": args.modifier,
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "groups": records,
    }
    manifest_path = args.output_root / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
