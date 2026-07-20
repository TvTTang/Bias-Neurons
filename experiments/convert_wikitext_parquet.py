#!/usr/bin/env python3
"""Convert pinned WikiText parquet shards to raw text with an audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional, Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream the text column from one or more WikiText parquet shards "
            "into the raw line-oriented format used by the calibration builder."
        )
    )
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--batch-size", default=65536, type=int)
    parser.add_argument("--expected-rows", type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_shards(
    input_paths: Sequence[Path],
    output_path: Path,
    batch_size: int,
    expected_rows: Optional[int] = None,
    parquet_module: object = None,
) -> Dict[str, object]:
    if not input_paths:
        raise ValueError("At least one input shard is required.")
    if batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    if parquet_module is None:
        try:
            import pyarrow.parquet as parquet_module
        except ImportError as error:
            raise RuntimeError(
                "pyarrow is required to convert parquet shards."
            ) from error

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(output_path.name + ".partial")
    if partial_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing partial output: {partial_path}"
        )

    source_records = []
    row_count = 0
    try:
        with partial_path.open("w", encoding="utf-8", newline="\n") as output:
            for input_path in input_paths:
                source_records.append(
                    {
                        "path": str(input_path.resolve()),
                        "sha256": sha256(input_path),
                        "size_bytes": input_path.stat().st_size,
                    }
                )
                parquet_file = parquet_module.ParquetFile(input_path)
                shard_rows = 0
                for batch in parquet_file.iter_batches(
                    batch_size=batch_size, columns=["text"]
                ):
                    for text in batch.column(0).to_pylist():
                        if not isinstance(text, str):
                            raise ValueError(
                                f"Non-string text value in shard: {input_path}"
                            )
                        output.write(text)
                        if not text.endswith(("\n", "\r")):
                            output.write("\n")
                        row_count += 1
                        shard_rows += 1
                source_records[-1]["rows"] = shard_rows
        if expected_rows is not None and row_count != expected_rows:
            raise ValueError(
                f"Converted {row_count} rows, expected {expected_rows}."
            )
        os.replace(partial_path, output_path)
    except BaseException:
        if partial_path.exists():
            partial_path.unlink()
        raise

    return {
        "format": "WikiText parquet text column converted to line-oriented UTF-8",
        "sources": source_records,
        "output_path": str(output_path.resolve()),
        "output_sha256": sha256(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "row_count": row_count,
        "batch_size": batch_size,
    }


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest or args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}"
        )
    manifest = convert_shards(
        input_paths=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        expected_rows=args.expected_rows,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
