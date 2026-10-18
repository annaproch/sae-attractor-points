# Related-background pilot

The user ran this five-step synthetic pilot with the default settings of
`scripts/backgrounds.py`. It completed 784 trajectories in 280 matched groups
in about 4.6 seconds. The checkpoint SHA-256 matches pilot-003. Full states and
contrasts remain saved locally; metadata and summary metrics are plain text.

## Findings

| Step | Nonzero trajectories | Equal-amplitude background mode | Matched decoded-norm mode |
| --- | ---: | ---: | ---: |
| 1 | 302 | 176 | 126 |
| 2 | 111 | 83 | 28 |
| 3 | 24 | 24 | 0 |
| 4 | 9 | 9 | 0 |
| 5 | 2 | 2 | 0 |

The largest active support dropped from 247 features at initialization to two
at every subsequent saved step. Across all probes, only features 120, 2499,
4969, 6712, 11341 and 12692 were active at step 1. Feature 4969 disappeared
by step 2. None of the single-target groups had a nonzero contrast at step 5.

There were 66 nonzero pair interaction contrasts at step 1 (38 equal-amplitude,
28 norm-matched), and 59 at step 2 (31 and 28 respectively), using a contrast
norm tolerance of 1e-6. These contrasts measure AB minus A minus B plus the
background trajectory. Their existence alone does not show an effect beyond
what gated affine dynamics predict.

Numerical sensitivity check: raising the tolerance to 1e-5, 1e-4 or 1e-3 gives
62 pair contrasts at step 1 and 57 at step 2, rather than 66 and 59. The excluded
contrasts are consistent with float32 roundoff and should not be counted as
substantive interactions. Counts at steps 3–5 remain 10, 4 and 2.

Only two joint-addition conditions remained nonzero at step 5:

- Group 221: 120+6712, full union of 240 neighbors, equal-amplitude background,
  target strength 1.25. Endpoint: feature 120 = 4.19823 and feature 6712 = 4.27798.
  Background, A-only and B-only were all zero at that step.
- Group 277: 120+2499, full union of 245 neighbors, equal-amplitude background,
  target strength 1.25. Endpoint: feature 120 = 3.47463. Background, A-only and
  B-only were all zero at that step.

Their initial decoded background norms were 40.66 and 35.99; the total joint
initialization norms were 46.62 and 42.63. The corresponding norm-matched runs
did not survive to step 5. Thus the observed longer persistence occurs in the
large-total-input conditions, and is not demonstrated at the tested matched
norm. No later steps were computed, so neither survival nor collapse beyond
step 5 is established for these two conditions.

### Closer inspection of the survivors

Both joint trajectories shrink after the first step. For 120+6712, the joint
amplitudes are (11.010, 11.219), (9.008, 9.338), (7.228, 7.550),
(5.635, 5.863), and (4.198, 4.278) at steps 1–5. The background path turns
off at step 3; A-only and B-only turn off at step 5. The joint contrast is
within 6e-6 of zero for steps 1–2, then reaches norm 3.378 at step 3 as the
background path crosses its gates. Joint survival is therefore a concrete
example of delayed threshold crossing, not observed runaway growth.

For 120+2499, the joint path starts at (10.835, 11.589) at step 1, shrinks
through (5.016, 4.519) at step 4, and loses feature 2499 at step 5. It is
not maintaining a stable two-feature state.

The two-coordinate decoder-to-encoder feedback matrices have largest
eigenvalues 0.92354 (120/6712) and 0.91504 (120/2499), with negative affine
biases for both coordinates. Within these fixed supports, this supports the
interpretation of decaying activity with mutual reinforcement delaying
switch-off. This restricted calculation does not establish the behavior of
the full map if other features become active, and no extra rollout was run.

## Interpretation and limits

Related synthetic backgrounds change short-term responses and pair interactions,
but most tested activity vanishes quickly. Pair decoder cosines are 0.1942
(120/6712) and 0.1484 (120/2499), with positive directed couplings in both
directions. The observed interactions have not been shown to add predictive
value beyond these static quantities or first-step gating.

This pilot uses only two target strengths (0.75/1.25 times isolated boundaries)
and one background strength (0.25 times A's boundary). Matching uses the nearest
single neighbor as the reference; that is a deliberately small synthetic input,
not a natural-activation calibration. Counts include duplicated/reused controls
and must not be treated as independent statistical samples. Pair pools are
unions of per-target top-128 pools, so full pair pools exceed 128 members.

A useful follow-up would sweep decoded background norm while holding subset
membership fixed, comparing the two surviving full pools against small subsets
at the same total norm. That would help separate total-input effects from
companion identity. This follow-up has not been run. These results do not assign
concept meanings or establish mechanisms in the LLM.

## Reproduce and verify

Launch command is in the repository README; choose a fresh output directory.
Regenerate the analysis without running any SAE steps:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m scripts.analyze_backgrounds runs/background-pilot-001
```

All stored contrasts were checked against their saved component states at
steps 0–5. Pair initialization contrasts are exactly zero. All nonempty
norm-matched initialization norms match their recorded references within
relative tolerance 2e-6. Six implementation tests pass.
