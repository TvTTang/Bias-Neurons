# V3 direction-aware robust-validation results

V3 keeps the V2 train-only neuron localization but expands the intervention
from removal-only to a pre-registered activation-scale grid:
`0, 0.25, 0.5, 0.75, 1.25, 1.5, 2`. A candidate must:

1. reduce the paired absolute target-probability gap on IID, lexical-OOD, and
   template-OOD validation environments;
2. retain at least 95% of the paired target probability in every environment;
3. satisfy the frozen WikiText-103 semantic constraints; and
4. maximize its worst-environment validation reduction.

The existing V2 test splits were not used for V3 selection or evaluation.

## Frozen validation decisions

| Dimension | Neurons | Scale | IID reduction | Lexical-OOD reduction | Template-OOD reduction | Semantic top-1 agreement |
|---|---:|---:|---:|---:|---:|---:|
| Ethnicity | 2 | 0.00 | 0.645% | 12.925% | 0.912% | 100.000% |
| Gender | 8 | 1.25 | 12.290% | 7.793% | 11.609% | 99.805% |
| Religion | 2 | 0.50 | 4.420% | 0.537% | 4.299% | 100.000% |

The gender result is the principal V3 finding. Every V2 removal candidate
failed gender validation, whereas increasing eight stable activations by 25%
improves all three validation environments with negligible semantic change.
This demonstrates that neuron importance and intervention direction are
distinct questions.

## Sealed external CrowS-Pairs evaluation

After freezing all three selections, V3 was evaluated once on the official
CrowS-Pairs CSV using its released common-token pseudo-log-likelihood rule.
Lower CrowS score is better.

| Dimension | Pairs | Baseline | V3 | Reduction (percentage points) | Good/bad flips | McNemar p |
|---|---:|---:|---:|---:|---:|---:|
| Ethnicity (`race-color`) | 516 | 54.651% | 55.039% | -0.388 | 0 / 2 | 0.500 |
| Gender | 262 | 57.634% | 57.634% | 0.000 | 0 / 0 | 1.000 |
| Religion | 105 | 66.667% | 66.667% | 0.000 | 0 / 0 | 1.000 |

The paired bootstrap 95% interval for intervention-minus-baseline CrowS score
is `[0.000, 0.969]` percentage points for ethnicity and exactly `[0, 0]` for
gender and religion. V3 therefore does not improve sentence-level CrowS-Pairs
bias. The ethnicity binary score worsens by two pairs, but the change is not
statistically significant.

## Interpretation

The internal and external metrics probe different behavior:

- the project benchmark measures a demographic target-token probability gap
  in controlled masked templates;
- CrowS-Pairs measures preference between natural sentence pairs using the
  pseudo-likelihood of unchanged context tokens.

The frozen neurons strongly affect the former but rarely change the rank order
of CrowS-Pairs sentence scores. The defensible claim is therefore targeted,
not universal: direction-aware neuron intervention improves robust
template-based demographic parity while preserving neutral MLM behavior, but
does not transfer to sentence-level CrowS-Pairs.

CrowS-Pairs is retained as a supplementary external benchmark. Its official
repository warns of substantial noise and reliability problems, so it should
not be treated as the sole measure of social bias.

Exact metrics and artifact hashes are recorded in
`experiments/results/v3_frozen_external_results.json`.
