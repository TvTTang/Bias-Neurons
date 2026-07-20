#!/usr/bin/env python3
"""Numerically verify legacy and generic FFN intervention equivalence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check that scale=0 equals legacy remove and scale=2 equals "
            "legacy enhance for one deterministic masked prompt."
        )
    )
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layer", default=10, type=int)
    parser.add_argument("--neuron", default=198, type=int)
    parser.add_argument("--tolerance", default=0.0, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from transformers import BertTokenizer

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "bias_neuron_src")
    )
    from custom_bert_bias import BertForMaskedLM

    tokenizer = BertTokenizer.from_pretrained(
        str(args.model_path),
        do_lower_case=False,
        local_files_only=True,
    )
    tokens = tokenizer.tokenize("The gender of this person is [MASK].")
    model_tokens = [tokenizer.cls_token, *tokens, tokenizer.sep_token]
    target_position = model_tokens.index(tokenizer.mask_token)
    input_ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(model_tokens)],
        device=args.device,
    )
    attention_mask = torch.ones_like(input_ids)
    token_type_ids = torch.zeros_like(input_ids)
    model = BertForMaskedLM.from_pretrained(str(args.model_path))
    model.to(args.device)
    model.eval()
    neuron = [(args.layer, args.neuron)]

    def logits(operation, scale=None):
        with torch.no_grad():
            _, values = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                tgt_pos=target_position,
                tgt_layer=0,
                imp_pos=neuron,
                imp_op=operation,
                imp_scale=scale,
            )
        return values

    remove_difference = float(
        torch.max(torch.abs(logits("remove") - logits("scale", 0.0))).item()
    )
    enhance_difference = float(
        torch.max(torch.abs(logits("enhance") - logits("scale", 2.0))).item()
    )
    result = {
        "device": args.device,
        "model_path": str(args.model_path.resolve()),
        "neuron": [args.layer, args.neuron],
        "remove_vs_scale_0_max_abs_difference": remove_difference,
        "enhance_vs_scale_2_max_abs_difference": enhance_difference,
        "tolerance": args.tolerance,
        "passed": (
            remove_difference <= args.tolerance
            and enhance_difference <= args.tolerance
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
