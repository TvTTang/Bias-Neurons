#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: $0 <demographic_dimension> <demographic1> <demographic2> <modifier>" >&2
    echo "Example: $0 ethnicity black white N" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
BERT_MODEL_PATH="${BERT_MODEL_PATH:-${SCRIPT_DIR}/handload-bert-base-cased}"
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/bias_neuron_data}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/bias_results_modifier}"

# AutoDL single-GPU instances expose the first physical GPU as device 0.
# After CUDA_VISIBLE_DEVICES is applied, PyTorch also addresses it as cuda:0.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/1_analyze_mlm_bias.py" \
    --bert_model_path "${BERT_MODEL_PATH}" \
    --demographic_dimension "$1" \
    --demographic1 "$2" \
    --demographic2 "$3" \
    --modifier "$4" \
    --data_path "${DATA_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --gpus 0 \
    --max_seq_length 128 \
    --get_ig2_gold \
    --get_base \
    --get_ig2_gold_gap_filtered \
    --batch_size 20 \
    --num_batch 1 \
    --debug 100000
