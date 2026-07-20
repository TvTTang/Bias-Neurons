#!/usr/bin/env python3
"""Evaluate neuron-removal candidates on paired bias and neutral MLM data."""

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
            "Evaluate candidate FFN neuron sets using paired demographic "
            "probability gaps and neutral masked-token preservation."
        )
    )
    parser.add_argument("--candidate-sets", required=True, type=Path)
    parser.add_argument("--candidate-id", nargs="*")
    parser.add_argument("--bias-data-root", required=True, type=Path)
    parser.add_argument("--dimension", required=True)
    parser.add_argument("--group1", required=True)
    parser.add_argument("--group2", required=True)
    parser.add_argument("--modifier", default="N")
    parser.add_argument("--semantic-file", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-seq-length", default=128, type=int)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--do-lower-case", action="store_true")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--bias-limit", type=int)
    parser.add_argument("--semantic-limit", type=int)
    parser.add_argument("--progress-every", default=100, type=int)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_candidates(
    path: Path, selected_ids: Sequence[str] = None
) -> List[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        values = json.load(handle)
    if not isinstance(values, list) or not values:
        raise ValueError("Candidate file must contain a non-empty JSON list.")
    requested = set(selected_ids) if selected_ids else None
    candidates = []
    seen_ids = set()
    for value in values:
        candidate_id = value.get("candidate_id")
        neurons = value.get("neurons")
        if not isinstance(candidate_id, str) or candidate_id in seen_ids:
            raise ValueError("Candidate IDs must be unique strings.")
        seen_ids.add(candidate_id)
        if requested is not None and candidate_id not in requested:
            continue
        if not isinstance(neurons, list) or not neurons:
            raise ValueError(f"Candidate {candidate_id} has no neurons.")
        parsed = []
        for neuron in neurons:
            if (
                not isinstance(neuron, list)
                or len(neuron) != 2
                or not all(isinstance(index, int) for index in neuron)
            ):
                raise ValueError(f"Invalid neuron in candidate {candidate_id}.")
            parsed.append((neuron[0], neuron[1]))
        if len(set(parsed)) != len(parsed):
            raise ValueError(f"Candidate {candidate_id} contains duplicate neurons.")
        candidates.append({"candidate_id": candidate_id, "neurons": parsed})
    if requested is not None:
        missing = requested - {item["candidate_id"] for item in candidates}
        if missing:
            raise ValueError(f"Requested candidate IDs are absent: {sorted(missing)}")
    if not candidates:
        raise ValueError("No candidates were selected.")
    return candidates


def deduplicate_candidates(
    candidates: Sequence[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], Dict[str, str]]:
    unique = []
    key_to_representative = {}
    candidate_to_representative = {}
    for candidate in candidates:
        key = tuple(candidate["neurons"])
        representative = key_to_representative.get(key)
        if representative is None:
            representative = candidate["candidate_id"]
            key_to_representative[key] = representative
            unique.append(candidate)
        candidate_to_representative[candidate["candidate_id"]] = representative
    return unique, candidate_to_representative


def load_bias_pairs(
    data_root: Path,
    dimension: str,
    group1: str,
    group2: str,
    modifier: str,
    limit: int = None,
) -> List[Tuple[str, str, str]]:
    paths = [
        data_root / dimension / f"{group}_{modifier}_data.json"
        for group in (group1, group2)
    ]
    data = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            data.append(json.load(handle))
    if len(data[0]) != len(data[1]) or not data[0]:
        raise ValueError("Paired demographic files have different bag counts.")
    pairs = []
    for bag_index, (bag1, bag2) in enumerate(zip(*data)):
        if len(bag1) != len(bag2):
            raise ValueError(f"Paired bag {bag_index} has different sizes.")
        for example1, example2 in zip(bag1, bag2):
            if example1[0] != example2[0]:
                raise ValueError("Paired demographic prompts differ.")
            pairs.append((example1[0], example1[1], example2[1]))
            if limit is not None and len(pairs) >= limit:
                return pairs
    return pairs


def load_semantic_records(path: Path, limit: int = None) -> List[Dict[str, object]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
            if limit is not None and len(records) >= limit:
                break
    if not records:
        raise ValueError("Semantic calibration file is empty.")
    return records


def prepare_prompt(
    tokens: Sequence[str],
    tokenizer: object,
    max_seq_length: int,
) -> Tuple[List[int], List[int], List[int], int]:
    model_tokens = [tokenizer.cls_token, *tokens, tokenizer.sep_token]
    if model_tokens.count(tokenizer.mask_token) != 1:
        raise ValueError("Each prompt must contain exactly one mask token.")
    if len(model_tokens) > max_seq_length:
        raise ValueError("Prompt exceeds --max-seq-length.")
    target_position = model_tokens.index(tokenizer.mask_token)
    input_ids = tokenizer.convert_tokens_to_ids(model_tokens)
    attention_mask = [1] * len(input_ids)
    padding = max_seq_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding
    attention_mask += [0] * padding
    token_type_ids = [0] * max_seq_length
    return input_ids, attention_mask, token_type_ids, target_position


def summarize_bias(
    signed_sum: float,
    absolute_sum: float,
    square_sum: float,
    target_probability_sum: float,
    count: int,
) -> Dict[str, float]:
    return {
        "num_bias_examples": count,
        "bias_mean_signed_gap": signed_sum / count,
        "bias_mean_absolute_gap": absolute_sum / count,
        "bias_rms_gap": math.sqrt(square_sum / count),
        "bias_mean_pair_target_probability": target_probability_sum / count,
    }


def summarize_semantic(
    gold_probability_sum: float,
    negative_log_likelihood_sum: float,
    correct: int,
    baseline_top1_agreement: int,
    kl_from_baseline_sum: float,
    count: int,
) -> Dict[str, float]:
    return {
        "num_semantic_examples": count,
        "semantic_mean_gold_probability": gold_probability_sum / count,
        "semantic_mean_negative_log_likelihood": (
            negative_log_likelihood_sum / count
        ),
        "semantic_top1_accuracy": correct / count,
        "semantic_baseline_top1_agreement": baseline_top1_agreement / count,
        "semantic_mean_kl_from_baseline": kl_from_baseline_sum / count,
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    if "," in args.gpus:
        raise ValueError("This evaluator intentionally supports one GPU.")

    import numpy as np
    import torch
    import torch.nn.functional as functional
    from transformers import BertTokenizer

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "bias_neuron_src")
    )
    from custom_bert_bias import BertForMaskedLM

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
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

    candidates = load_candidates(args.candidate_sets, args.candidate_id)
    unique_candidates, candidate_to_representative = deduplicate_candidates(
        candidates
    )
    methods = [{"candidate_id": "baseline", "neurons": None}, *unique_candidates]
    bias_pairs = load_bias_pairs(
        args.bias_data_root,
        args.dimension,
        args.group1,
        args.group2,
        args.modifier,
        args.bias_limit,
    )
    semantic_records = load_semantic_records(
        args.semantic_file, args.semantic_limit
    )
    bias_accumulators = {
        method["candidate_id"]: [0.0, 0.0, 0.0, 0.0, 0]
        for method in methods
    }
    semantic_accumulators = {
        method["candidate_id"]: [0.0, 0.0, 0, 0, 0.0, 0]
        for method in methods
    }
    started = time.perf_counter()

    def logits_for(tokens, neurons):
        ids, mask, types, position = prepare_prompt(
            tokens, tokenizer, args.max_seq_length
        )
        with torch.no_grad():
            _, logits = model(
                input_ids=torch.tensor(ids, device=device).unsqueeze(0),
                attention_mask=torch.tensor(mask, device=device).unsqueeze(0),
                token_type_ids=torch.tensor(types, device=device).unsqueeze(0),
                tgt_pos=position,
                tgt_layer=0,
                imp_pos=neurons,
                imp_op="remove" if neurons is not None else None,
            )
        return logits

    for example_index, (text, target1, target2) in enumerate(
        bias_pairs, start=1
    ):
        tokens = tokenizer.tokenize(text)
        target_ids = tokenizer.convert_tokens_to_ids([target1, target2])
        if (
            tokenizer.unk_token_id in target_ids
            or target_ids[0] == target_ids[1]
        ):
            raise ValueError("Bias targets are unknown or map to the same token.")
        for method in methods:
            logits = logits_for(tokens, method["neurons"])
            probabilities = functional.softmax(logits[0], dim=0)
            probability1 = float(probabilities[target_ids[0]].item())
            probability2 = float(probabilities[target_ids[1]].item())
            gap = probability1 - probability2
            accumulator = bias_accumulators[method["candidate_id"]]
            accumulator[0] += gap
            accumulator[1] += abs(gap)
            accumulator[2] += gap * gap
            accumulator[3] += (probability1 + probability2) / 2
            accumulator[4] += 1
        if args.progress_every > 0 and example_index % args.progress_every == 0:
            print(f"bias_examples={example_index}/{len(bias_pairs)}", flush=True)

    for example_index, record in enumerate(semantic_records, start=1):
        tokens = record["masked_tokens"]
        target_id = int(record["target_id"])
        baseline_probabilities = None
        baseline_prediction = None
        for method in methods:
            logits = logits_for(tokens, method["neurons"])
            probabilities = functional.softmax(logits[0], dim=0)
            prediction = int(torch.argmax(logits[0]).item())
            gold_probability = float(probabilities[target_id].item())
            if method["candidate_id"] == "baseline":
                baseline_probabilities = probabilities
                baseline_prediction = prediction
                kl_from_baseline = 0.0
            else:
                kl_from_baseline = float(
                    torch.sum(
                        baseline_probabilities
                        * (
                            torch.log(baseline_probabilities.clamp_min(1e-45))
                            - torch.log(probabilities.clamp_min(1e-45))
                        )
                    ).item()
                )
            accumulator = semantic_accumulators[method["candidate_id"]]
            accumulator[0] += gold_probability
            accumulator[1] -= math.log(max(gold_probability, 1e-45))
            accumulator[2] += int(prediction == target_id)
            accumulator[3] += int(prediction == baseline_prediction)
            accumulator[4] += kl_from_baseline
            accumulator[5] += 1
        if args.progress_every > 0 and example_index % args.progress_every == 0:
            print(
                f"semantic_examples={example_index}/{len(semantic_records)}",
                flush=True,
            )

    representative_rows = {}
    for method in methods:
        candidate_id = method["candidate_id"]
        bias = summarize_bias(*bias_accumulators[candidate_id])
        semantic = summarize_semantic(*semantic_accumulators[candidate_id])
        representative_rows[candidate_id] = {
            "candidate_id": candidate_id,
            "num_neurons": len(method["neurons"] or []),
            **bias,
            **semantic,
        }
    rows = [representative_rows["baseline"]]
    for candidate in candidates:
        representative = candidate_to_representative[candidate["candidate_id"]]
        row = dict(representative_rows[representative])
        row["candidate_id"] = candidate["candidate_id"]
        rows.append(row)
    baseline = rows[0]
    for row in rows:
        row["bias_absolute_gap_reduction"] = (
            1
            - row["bias_mean_absolute_gap"]
            / baseline["bias_mean_absolute_gap"]
            if baseline["bias_mean_absolute_gap"] > 0
            else 0.0
        )
        row["semantic_nll_increase"] = (
            row["semantic_mean_negative_log_likelihood"]
            - baseline["semantic_mean_negative_log_likelihood"]
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "candidate_evaluation.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "model_path": str(args.model_path.resolve()),
        "device": str(device),
        "do_lower_case": args.do_lower_case,
        "intervention": "set selected FFN activations to zero at [MASK]",
        "candidate_sets_path": str(args.candidate_sets.resolve()),
        "candidate_sets_sha256": sha256(args.candidate_sets),
        "num_candidate_ids": len(candidates),
        "num_unique_neuron_sets": len(unique_candidates),
        "bias_data_root": str(args.bias_data_root.resolve()),
        "semantic_file": str(args.semantic_file.resolve()),
        "semantic_file_sha256": sha256(args.semantic_file),
        "elapsed_seconds": time.perf_counter() - started,
        "results": rows,
        "output_csv": str(csv_path.resolve()),
        "output_csv_sha256": sha256(csv_path),
    }
    with (args.output_dir / "candidate_evaluation_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
