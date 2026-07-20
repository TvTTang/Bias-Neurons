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
