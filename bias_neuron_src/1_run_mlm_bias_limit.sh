#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

DEMOGRAPHIC_DIMENSION="${1:-ethnicity}"
DEMOGRAPHIC1="${2:-black}"
DEMOGRAPHIC2="${3:-white}"
MODIFIER="${4:-N}"

exec bash "${SCRIPT_DIR}/1_run_mlm_bias_modifier_limit.sh" \
    "${DEMOGRAPHIC_DIMENSION}" \
    "${DEMOGRAPHIC1}" \
    "${DEMOGRAPHIC2}" \
    "${MODIFIER}"
