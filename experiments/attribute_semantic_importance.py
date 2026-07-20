#!/usr/bin/env python3
"""Attribute neutral masked-token predictions to every BERT FFN neuron."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute dense FFN integrated-gradient importance for gold masked "
            "tokens in a neutral semantic calibration JSONL."
        )
    )
    parser.add_argument("--calibration-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-seq-length", default=128, type=int)
    parser.add_argument("--batch-size", default=20, type=int)
    parser.add_argument("--num-batch", default=1, type=int)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument(
        "--do-lower-case",
        action="store_true",
        help="Enable only when the model/tokenizer is uncased.",
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--top-k", default=100, type=int)
    parser.add_argument("--progress-every", default=10, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path, limit: int = None) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")
            records.append(record)
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError("The calibration file contains no records.")
    return records


def validate_record(
    record: Dict[str, object], tokenizer: object
) -> Tuple[List[str], int, int]:
    masked_tokens = record.get("masked_tokens")
    mask_index = record.get("mask_index")
    target_id = record.get("target_id")
    if not isinstance(masked_tokens, list) or not all(
        isinstance(token, str) for token in masked_tokens
    ):
        raise ValueError("masked_tokens must be a list of strings.")
    if not isinstance(mask_index, int) or not 0 <= mask_index < len(masked_tokens):
        raise ValueError("mask_index is out of range.")
    if masked_tokens[mask_index] != tokenizer.mask_token:
        raise ValueError("mask_index does not point to the tokenizer mask token.")
    if masked_tokens.count(tokenizer.mask_token) != 1:
        raise ValueError("Each calibration example must contain exactly one mask.")
    if not isinstance(target_id, int) or target_id < 0:
        raise ValueError("target_id must be a non-negative integer.")
    target_token = record.get("target_token")
    if tokenizer.convert_tokens_to_ids(target_token) != target_id:
        raise ValueError("target_token and target_id disagree.")
    return masked_tokens, mask_index, target_id


def make_model_inputs(
    record: Dict[str, object],
    tokenizer: object,
    max_seq_length: int,
) -> Tuple[List[int], List[int], List[int], int, int]:
    masked_tokens, mask_index, target_id = validate_record(record, tokenizer)
    tokens = [tokenizer.cls_token, *masked_tokens, tokenizer.sep_token]
    if len(tokens) > max_seq_length:
        raise ValueError(
            f"Calibration sequence has {len(tokens)} tokens, exceeding "
            f"--max-seq-length={max_seq_length}."
        )
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    attention_mask = [1] * len(input_ids)
    token_type_ids = [0] * len(input_ids)
    padding = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding
    attention_mask += [0] * padding
    token_type_ids += [0] * padding
    return input_ids, attention_mask, token_type_ids, mask_index + 1, target_id


def rank_rows(
    signed_sums: Sequence[float],
    abs_sums: Sequence[float],
    sum_squares: Sequence[float],
    count: int,
    neurons_per_layer: int,
) -> List[Dict[str, object]]:
    if count <= 0:
        raise ValueError("At least one attributed example is required.")
    if not (
        len(signed_sums) == len(abs_sums) == len(sum_squares)
        and len(signed_sums) % neurons_per_layer == 0
    ):
        raise ValueError("Attribution arrays do not match the model dimensions.")
    rows = []
    for index, (signed_sum, abs_sum, sum_square) in enumerate(
        zip(signed_sums, abs_sums, sum_squares)
    ):
        layer, neuron = divmod(index, neurons_per_layer)
        mean_abs = abs_sum / count
        mean_square = sum_square / count
        rows.append(
            {
                "layer": layer,
                "neuron": neuron,
                "semantic_mean_abs_ig": mean_abs,
                "semantic_rms_ig": math.sqrt(max(0.0, mean_square)),
                "semantic_mean_signed_ig": signed_sum / count,
                "semantic_std_abs_ig": math.sqrt(
                    max(0.0, mean_square - mean_abs * mean_abs)
                ),
            }
        )
    ranked = sorted(
        range(len(rows)),
        key=lambda index: (
            rows[index]["semantic_mean_abs_ig"],
            -rows[index]["layer"],
            -rows[index]["neuron"],
        ),
        reverse=True,
    )
    for rank, index in enumerate(ranked, start=1):
        rows[index]["semantic_rank"] = rank
    return rows


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    if args.batch_size <= 0 or args.num_batch <= 0:
        raise ValueError("Integration batch parameters must be positive.")
    if args.max_seq_length <= 2:
        raise ValueError("--max-seq-length must leave room for special tokens.")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import BertTokenizer

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "bias_neuron_src"))
    from custom_bert_bias import BertForMaskedLM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        if "," in args.gpus:
            raise ValueError("This runner intentionally supports exactly one GPU.")
        device = torch.device(f"cuda:{args.gpus}")
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer.from_pretrained(
        str(args.model_path),
        do_lower_case=args.do_lower_case,
        local_files_only=True,
    )
    model = BertForMaskedLM.from_pretrained(str(args.model_path))
    model.to(device)
    model.eval()

    records = load_records(args.calibration_file, args.limit)
    num_layers = model.bert.config.num_hidden_layers
    neurons_per_layer = model.bert.config.intermediate_size
    total_neurons = num_layers * neurons_per_layer
    signed_sums = np.zeros(total_neurons, dtype=np.float64)
    abs_sums = np.zeros(total_neurons, dtype=np.float64)
    sum_squares = np.zeros(total_neurons, dtype=np.float64)
    correct = 0
    gold_probability_sum = 0.0
    negative_log_likelihood_sum = 0.0
    num_points = args.batch_size * args.num_batch
    started = time.perf_counter()

    for example_index, record in enumerate(records, start=1):
        (
            input_ids,
            attention_mask,
            token_type_ids,
            target_position,
            target_id,
        ) = make_model_inputs(record, tokenizer, args.max_seq_length)
        input_ids_tensor = torch.tensor(
            input_ids, dtype=torch.long, device=device
        ).unsqueeze(0)
        attention_mask_tensor = torch.tensor(
            attention_mask, dtype=torch.long, device=device
        ).unsqueeze(0)
        token_type_ids_tensor = torch.tensor(
            token_type_ids, dtype=torch.long, device=device
        ).unsqueeze(0)

        for layer in range(num_layers):
            activation, logits = model(
                input_ids=input_ids_tensor,
                attention_mask=attention_mask_tensor,
                token_type_ids=token_type_ids_tensor,
                tgt_pos=target_position,
                tgt_layer=layer,
            )
            if layer == 0:
                probabilities = functional.softmax(logits, dim=1)
                gold_probability = float(probabilities[0, target_id].item())
                gold_probability_sum += gold_probability
                negative_log_likelihood_sum -= math.log(
                    max(gold_probability, 1e-45)
                )
                correct += int(int(torch.argmax(logits[0]).item()) == target_id)

            baseline = torch.zeros_like(activation)
            step = activation / num_points
            scaled = torch.cat(
                [baseline + step * point for point in range(num_points)], dim=0
            )
            scaled.requires_grad_(True)
            integrated_gradient = None
            for batch_index in range(args.num_batch):
                batch_activation = scaled[
                    batch_index * args.batch_size :
                    (batch_index + 1) * args.batch_size
                ]
                _, gradient = model(
                    input_ids=input_ids_tensor,
                    attention_mask=attention_mask_tensor,
                    token_type_ids=token_type_ids_tensor,
                    tgt_pos=target_position,
                    tgt_layer=layer,
                    tmp_score=batch_activation,
                    tgt_label=target_id,
                )
                gradient = gradient.sum(dim=0)
                integrated_gradient = (
                    gradient
                    if integrated_gradient is None
                    else integrated_gradient + gradient
                )
            integrated_gradient = integrated_gradient * step[0]
            values = integrated_gradient.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
            offset = layer * neurons_per_layer
            signed_sums[offset : offset + neurons_per_layer] += values
            abs_sums[offset : offset + neurons_per_layer] += np.abs(values)
            sum_squares[offset : offset + neurons_per_layer] += values * values

        if args.progress_every > 0 and (
            example_index % args.progress_every == 0
            or example_index == len(records)
        ):
            elapsed = time.perf_counter() - started
            print(
                f"attributed={example_index}/{len(records)} "
                f"elapsed_seconds={elapsed:.3f}",
                flush=True,
            )

    rows = rank_rows(
        signed_sums,
        abs_sums,
        sum_squares,
        len(records),
        neurons_per_layer,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "semantic_importance.csv"
    fieldnames = [
        "layer",
        "neuron",
        "semantic_rank",
        "semantic_mean_abs_ig",
        "semantic_rms_ig",
        "semantic_mean_signed_ig",
        "semantic_std_abs_ig",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ranked_rows = sorted(rows, key=lambda row: row["semantic_rank"])
    elapsed = time.perf_counter() - started
    summary = {
        "calibration_file": str(args.calibration_file.resolve()),
        "calibration_sha256": sha256(args.calibration_file),
        "model_path": str(args.model_path.resolve()),
        "device": str(device),
        "seed": args.seed,
        "do_lower_case": args.do_lower_case,
        "num_examples": len(records),
        "num_layers": num_layers,
        "neurons_per_layer": neurons_per_layer,
        "total_neurons": total_neurons,
        "target": "gold masked-token probability",
        "activation_baseline": "all-zero FFN activation at the masked position",
        "integration_rule": (
            "left Riemann sum matching the released bias attribution; "
            "endpoint excluded"
        ),
        "integration_points": num_points,
        "batch_size": args.batch_size,
        "num_batch": args.num_batch,
        "baseline_top1_accuracy": correct / len(records),
        "baseline_mean_gold_probability": gold_probability_sum / len(records),
        "baseline_mean_negative_log_likelihood": (
            negative_log_likelihood_sum / len(records)
        ),
        "elapsed_seconds": elapsed,
        "output_csv": str(csv_path.resolve()),
        "output_csv_sha256": sha256(csv_path),
        "top_neurons": ranked_rows[: args.top_k],
    }
    with (args.output_dir / "semantic_importance_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
