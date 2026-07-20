# V2 frozen-selection results

V2 ranks neurons using train-only bag/template bootstrap stability, directional
consistency, bias attribution, and neutral semantic cost. Hyperparameters are
chosen on validation data under fixed semantic constraints. Test and OOD data
were accessed only after the selections were frozen.

## Frozen decisions

| Dimension | Decision | Candidate | Neurons | Validation gap reduction |
|---|---|---|---|---:|
| Ethnicity | intervene | `stability-0p75_lambda-0p25_k-2` | `(8,1249), (7,324)` | 0.645% |
| Gender | no intervention | — | — | 0.000% |
| Religion | intervene | `stability-0p75_lambda-0p5_k-2` | `(8,1113), (8,1728)` | 1.999% |

Gender abstains because every V2 removal candidate increased its validation
bias gap. This is a frozen negative result, not a post-test exclusion.

## Held-out bias results

Positive values indicate a smaller mean absolute paired target-probability gap.

| Dimension | Split | Examples | Baseline gap | V2 gap | Relative reduction | Target-probability retention |
|---|---|---:|---:|---:|---:|---:|
| Ethnicity | IID test | 80 | 0.002679 | 0.002620 | 2.193% | 97.827% |
| Ethnicity | lexical OOD | 200 | 0.007398 | 0.006491 | 12.263% | 95.641% |
| Ethnicity | template OOD | 240 | 0.002745 | 0.002564 | 6.575% | 97.492% |
| Religion | IID test | 80 | 0.074741 | 0.074033 | 0.946% | 99.296% |
| Religion | lexical OOD | 200 | 0.032852 | 0.034852 | -6.090% | 100.442% |
| Religion | template OOD | 240 | 0.078544 | 0.077767 | 0.988% | 99.255% |

Ethnicity generalizes across both OOD axes. Religion generalizes across
templates but fails under lexical shift, so V2 should not be described as
universally robust.

## Held-out semantic preservation

The semantic test uses 1,000 WikiText-103 examples. OOD runs use one semantic
example only as an evaluator placeholder; semantic claims come exclusively from
the full IID semantic test below.

| Dimension | Top-1 agreement | Mean KL from baseline | NLL increase |
|---|---:|---:|---:|
| Ethnicity | 99.9% | 1.722e-5 | -1.764e-4 |
| Religion | 100.0% | 7.215e-6 | -1.597e-4 |

Exact metrics, frozen-manifest hashes, and result-file hashes are recorded in
`experiments/results/v2_frozen_results.json`.

## Scientific conclusion and next step

V2 supplies evidence that stable, semantic-safe neurons can support reliable
ethnicity mitigation with only two FFN units. It also exposes two limitations:
removal-only intervention is unsuitable for gender, and religion selection
does not survive lexical shift.

The next method iteration should remain train/validation-only and add
direction-aware intervention plus lexical leave-one-group-out stability. V2
must remain unchanged as a reported baseline; its test results must not be used
to retune V2.
