#!/usr/bin/env python3
"""Build deterministic neutral masked-token calibration sets from WikiText."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Tuple


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")
WIKITEXT_MARKUP = (
    (" @-@ ", "-"),
    (" @,@ ", ", "),
    (" @.@ ", ". "),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build neutral masked-token calibration data from the raw WikiText "
            "train/validation/test files."
        )
    )
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", required=True, type=Path)
    parser.add_argument("--test-file", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--demographic-dict", required=True, type=Path)
    parser.add_argument("--lexicon-files", nargs="*", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-size", default=512, type=int)
    parser.add_argument("--validation-size", default=512, type=int)
    parser.add_argument("--test-size", default=1000, type=int)
    parser.add_argument("--min-tokens", default=8, type=int)
    parser.add_argument("--max-tokens", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--do-lower-case",
        action="store_true",
        help="Enable only for an uncased tokenizer; cased models keep the default false.",
    )
    parser.add_argument(
        "--source-url",
        default=(
            "https://s3.amazonaws.com/research.metamind.io/"
            "wikitext/wikitext-103-raw-v1.zip"
        ),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(text: str) -> str:
    text = " ".join(text.strip().split())
    for source, target in WIKITEXT_MARKUP:
        text = text.replace(source, target)
    return " ".join(text.split())


def sentence_candidates(lines: Iterable[str]) -> Iterable[str]:
    for raw_line in lines:
        line = normalize_text(raw_line)
        if not line or (line.startswith("=") and line.endswith("=")):
            continue
        for sentence in SENTENCE_BOUNDARY.split(line):
            sentence = sentence.strip()
            if sentence:
                yield sentence


def flatten_json_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from flatten_json_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from flatten_json_strings(item)


def load_filter_terms(
    demographic_dict: Path, lexicon_files: Sequence[Path]
) -> Tuple[List[str], Dict[str, str]]:
    sources = [demographic_dict, *lexicon_files]
    terms = set()
    hashes = {}
    for path in sources:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        hashes[str(path.resolve())] = sha256(path)
        for term in flatten_json_strings(value):
            normalized = normalize_text(term).lower()
            if normalized:
                terms.add(normalized)
    return sorted(terms), hashes


def compile_filter_pattern(terms: Sequence[str]) -> Optional[Pattern[str]]:
    if not terms:
        return None
    alternatives = "|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)", flags=re.IGNORECASE)


def hash_score(seed: int, split: str, text: str) -> int:
    payload = f"{seed}\0{split}\0{text}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def valid_mask_positions(tokens: Sequence[str]) -> List[int]:
    positions = []
    for index, token in enumerate(tokens):
        if token == "[UNK]" or token.startswith("##"):
            continue
        if not any(character.isalnum() for character in token):
            continue
        if index + 1 < len(tokens) and tokens[index + 1].startswith("##"):
            continue
        positions.append(index)
    return positions


def select_sentences(
    path: Path,
    split: str,
    sample_size: int,
    tokenizer: object,
    filter_pattern: Optional[Pattern[str]],
    min_tokens: int,
    max_tokens: int,
    seed: int,
) -> Tuple[List[Tuple[int, str, List[str], List[int]]], Dict[str, int]]:
    heap: List[Tuple[int, str, List[str], List[int]]] = []
    stats = {
        "sentences_seen": 0,
        "filtered_sensitive": 0,
        "filtered_length": 0,
        "filtered_no_maskable_token": 0,
        "eligible": 0,
    }
    with path.open("r", encoding="utf-8") as handle:
        for text in sentence_candidates(handle):
            stats["sentences_seen"] += 1
            if filter_pattern is not None and filter_pattern.search(text):
                stats["filtered_sensitive"] += 1
                continue
            tokens = tokenizer.tokenize(text)
            if not min_tokens <= len(tokens) <= max_tokens:
                stats["filtered_length"] += 1
                continue
            positions = valid_mask_positions(tokens)
            if not positions:
                stats["filtered_no_maskable_token"] += 1
                continue
            stats["eligible"] += 1
            score = hash_score(seed, split, text)
            item = (-score, text, tokens, positions)
            if len(heap) < sample_size:
                heapq.heappush(heap, item)
            elif score < -heap[0][0]:
                heapq.heapreplace(heap, item)
    selected = [(-negative, text, tokens, positions) for negative, text, tokens, positions in heap]
    selected.sort(key=lambda item: item[0])
    if len(selected) != sample_size:
        raise ValueError(
            f"{split} produced only {len(selected)} eligible samples; "
            f"{sample_size} were requested."
        )
    return selected, stats


def build_records(
    selected: Sequence[Tuple[int, str, List[str], List[int]]],
    split: str,
    tokenizer: object,
    seed: int,
) -> List[Dict[str, object]]:
    records = []
    for score, text, tokens, positions in selected:
        mask_selector = hash_score(seed + 1, split, text)
        mask_index = positions[mask_selector % len(positions)]
        target_token = tokens[mask_index]
        target_id = tokenizer.convert_tokens_to_ids(target_token)
        masked_tokens = list(tokens)
        masked_tokens[mask_index] = tokenizer.mask_token
        record_id = hashlib.sha256(
            f"{split}\0{text}\0{mask_index}".encode("utf-8")
        ).hexdigest()[:20]
        records.append(
            {
                "id": record_id,
                "source_split": split,
                "source_hash_score": score,
                "text": text,
                "tokens": tokens,
                "masked_tokens": masked_tokens,
                "mask_index": mask_index,
                "target_token": target_token,
                "target_id": target_id,
            }
        )
    return records


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    if args.min_tokens <= 0 or args.max_tokens < args.min_tokens:
        raise ValueError("Invalid token-length bounds.")
    if min(args.train_size, args.validation_size, args.test_size) <= 0:
        raise ValueError("Requested split sizes must be positive.")

    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(
        str(args.tokenizer_path),
        do_lower_case=args.do_lower_case,
        local_files_only=True,
    )
    terms, term_hashes = load_filter_terms(
        args.demographic_dict, args.lexicon_files
    )
    filter_pattern = compile_filter_pattern(terms)

    sources = {
        "train": (args.train_file, args.train_size),
        "validation": (args.validation_file, args.validation_size),
        "test": (args.test_file, args.test_size),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_splits = {}
    for split, (path, sample_size) in sources.items():
        selected, filtering_stats = select_sentences(
            path=path,
            split=split,
            sample_size=sample_size,
            tokenizer=tokenizer,
            filter_pattern=filter_pattern,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            seed=args.seed,
        )
        records = build_records(selected, split, tokenizer, args.seed)
        output_path = args.output_dir / f"{split}.jsonl"
        write_jsonl(output_path, records)
        manifest_splits[split] = {
            "source_path": str(path.resolve()),
            "source_sha256": sha256(path),
            "output_path": str(output_path.resolve()),
            "output_sha256": sha256(output_path),
            "sample_size": len(records),
            "filtering": filtering_stats,
            "token_length_min": min(len(record["tokens"]) for record in records),
            "token_length_max": max(len(record["tokens"]) for record in records),
        }

    manifest = {
        "dataset": "WikiText-103-raw-v1",
        "source_url": args.source_url,
        "seed": args.seed,
        "tokenizer_path": str(args.tokenizer_path.resolve()),
        "tokenizer_vocab_size": len(tokenizer),
        "do_lower_case": args.do_lower_case,
        "min_tokens": args.min_tokens,
        "max_tokens": args.max_tokens,
        "filter_term_count": len(terms),
        "filter_source_sha256": term_hashes,
        "splits": manifest_splits,
    }
    manifest_path = args.output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
