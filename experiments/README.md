# Reproducible experiment utilities

The scripts in this directory parameterize the released post-processing code.
They do not change the IG2 attribution or bias-neuron selection algorithm.

## Extract the released bias-neuron baseline

```bash
python experiments/extract_bias_neurons.py \
  --input /path/to/Modifier-ethnicity-N-filtered-gap-rm-base-black-white.rlt.jsonl \
  --output-dir /path/to/analysis
```

The defaults reproduce the released thresholds:

- attribution threshold ratio: `0.2`
- initial within-bag mode ratio: `0.7`
- relation mode ratio: `0.1`
- minimum within-bag count: `3`
- adaptive target: two to five neurons per bag

The output includes bag-level neurons, relation-level neurons, the source file
SHA-256, selected thresholds, layer counts, and a machine-readable summary.

## Disk-efficient attribution runs

`bias_neuron_src/1_run_mlm_bias_limit.sh` uses the released gap-only runner. It
keeps the IG2 computation and filtering unchanged but does not persist the two
multi-gigabyte per-demographic intermediate JSONL files.

```bash
PYTHON_BIN=/path/to/python \
BERT_MODEL_PATH=/path/to/bert-base-cased \
DATA_PATH=/path/to/bias_neuron_data \
OUTPUT_DIR=/path/to/output \
bash bias_neuron_src/1_run_mlm_bias_limit.sh gender male female N
```

Before using this runner for full experiments, its gap JSONL was compared
against the full runner on the same GPU smoke input: the structures, SHA-256,
and all 10,491 attribution values were identical (`max_abs_diff = 0.0`).
