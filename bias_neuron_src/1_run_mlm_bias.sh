#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Default AutoDL run: ethnicity bias between black and white with modifier N.
# Pass four arguments to override these values.
DEMOGRAPHIC_DIMENSION="${1:-ethnicity}"
DEMOGRAPHIC1="${2:-black}"
DEMOGRAPHIC2="${3:-white}"
MODIFIER="${4:-N}"

exec bash "${SCRIPT_DIR}/1_run_mlm_bias_modifier.sh" \
    "${DEMOGRAPHIC_DIMENSION}" \
    "${DEMOGRAPHIC1}" \
    "${DEMOGRAPHIC2}" \
    "${MODIFIER}"
