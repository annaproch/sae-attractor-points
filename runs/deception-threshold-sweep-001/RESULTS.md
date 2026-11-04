# Threshold sweep: results and observations

## Setup and status

All 22 saved last-prompt-token activations (3 honest, 19 deceptive; ambiguous
responses excluded) use the same original z0. Each iteration multiplies the
SAE's learned per-feature thresholds by alpha. Surviving activation values are
unchanged. There is no top-k selection, normalization, or fixed feature set.
Labels describe subsequent generated responses, not the SAE trajectories.
Every state is saved, including stopping states. Existing numerical stopping
checks are unchanged; fixed-point candidates are not proofs of convergence.

Inspected multipliers: 1, 2, 4, 5, 6, 8. The 4x run was interrupted at step
7,500: its manifest still says running and it has no completed endpoint summary.
Its outcomes below are reconstructed from saved states and per-step CSV records.
The other five runs completed. Binary states remain local and gitignored.

## Outcomes

| Alpha | Honest (3) | Deceptive (19) |
|---|---|---|
| 1 | 3 growth limit, steps 12–14 | 19 growth limit, steps 11–14 |
| 2 | 3 growth limit, steps 19–22 | 19 growth limit, steps 18–67 |
| 4 | 2 growth limit (44, 51); 1 approximate period-15 cycle | 16 growth limit (30–52); 2 fixed-point candidates (94, 104); 1 approximate period-15 cycle |
| 5 | 3 growth limit (44–50) | 19 growth limit (32–49) |
| 6 | 2 fixed-point candidates; 1 growth limit (47) | 18 fixed-point candidates; 1 growth limit (43) |
| 8 | 3 fixed-point candidates | 19 fixed-point candidates |

Growth means code norm exceeded 1e6 times max(initial norm, 1). It is a safety
stop, not a converged endpoint. No zero-code collapse was observed at 8x.

## Feature counts and turnover

Median number of active features, honest / deceptive:

| Alpha | Step 1 | Step 3 | Step 5 | Step 10 |
|---|---|---|---|---|
| 1 | 67 / 64 | 73 / 74 | 172 / 189 | 3963 / 6741 |
| 2 | 29 / 26 | 22 / 19 | 18 / 17 | 40 / 16 |
| 4 | 13 / 12 | 9 / 9 | 7 / 7 | 7 / 6 |
| 5 | 10 / 10 | 8 / 8 | 5 / 5 | 2 / 2 |
| 6 | 9 / 9 | 6 / 6 | 3 / 2 | 2 / 2 |
| 8 | 6 / 6 | 4 / 4 | 2 / 2 | 2 / 2 |

Initial medians were 73 honest and 71 deceptive. At step 10, mean fraction of
original support retained was 49.0% / 48.9% at 1x, 15.0% / 13.0% at 2x, and
3.6% / 4.4% at 4x. Sparse states can therefore involve substantial replacement,
not just removal of weak initial features.

## What is interesting at 4x?

1. **The early contraction is almost identical across labels.** Both have median
   seven features at step 5 and six at step 20. Original-support retention is
   also similar. There is no clear early feature-count separation.
2. **A shared cycle crosses labels.** Honest probe 16 (artist admitting delays)
   and deceptive probe 7 (suspect claiming to be home watching TV) approach the
   same approximate 15-step cycle. In inspected late windows, relative vector
   differences after lag 15 were below 1e-6. Aligning the two trajectories by a
   two-step phase offset also gave relative differences below 1e-6. The runner
   only checks period 2, explaining why these continued to step 7,500.
3. **Cycle membership varies while a core persists.** Persistent IDs are
   {5,27,45,192,551,2063,3149,4460,16042}; IDs 218 and 11078 turn on/off.
   One cycle's counts are 10,10,10,9,9,11,11,11,9,9,11,11,11,9,9.
   None of these cycle features were initially active in either example.
4. **Only deceptive examples reached fixed-point candidates, but this is weak
   evidence.** Probe 5 stopped at step 104 with {22,174,181,269,299,680,1322};
   probe 12 stopped at step 94 with {22,95,148,174,680,1782,3539}. They share
   {22,174,680}, retain no initially active features, and are distinct endpoints.
   This is 2/19 deceptive versus 0/3 honest, far too little evidence to call
   these deception-specific attractors.
5. **Overall bounded outcomes do not separate the labels.** Counting the observed
   cycles and fixed-point candidates gives 1/3 honest versus 3/19 deceptive.
   Divergence occurs in both groups, and the same cycle attracts both labels.

The strongest observation is loss of initial support and convergence of two
opposite-label examples onto a common cycle. This argues against interpreting
that cycle as a unique deception signature. Paths into it could still differ;
no systematic path-based classifier or significance test has been performed.
Feature IDs have not been assigned semantic interpretations.

## Dominant features in divergent states

At 1x, recurring high-amplitude IDs are 16,624,474,25,48,248,381,285. IDs 16,
624,474 appear in the final top ten for all three honest examples and 17/19,
16/19,17/19 deceptive examples, respectively.

At 2x, another recurring set is 504,1263,4467,1277,218,509, while some examples
still amplify the 1x set. Feature 504 is in the final top ten for 3/3 honest
and 8/19 deceptive examples. These are post-hoc observations, not established
label-specific features. Final states are taken at different stopping steps.
For the inspected examples, the top eight features account for only about
2–10% of squared magnitude at 1x and 1–3% at 2x: divergence is broad.

At 5x, both labels initially reach a median of two active features at step 10,
then all diverge to approximately 10,700–12,000 active features. Recurrent
leaders include 504,1263,4467,2475,134,167,1277,44.

## Sparse endpoints at 6x and 8x

At 6x, 19 examples (2 honest, 17 deceptive) reach the {107,401} fixed-point
candidate, with norm approximately 2192.57. Deceptive probe 4 reaches {107}
alone, norm approximately 923.89. Honest probe 20 (teenager) and deceptive probe
10 (civilian's fainting excuse) diverge at steps 47 and 43; both have final
leaders 1263,504,2475,167,2569,62,4467,1277 in that order. Fixed-point candidates
stop at steps 48–81.

At 8x, all 22 reach fixed-point candidates within 43–79 steps. Twenty (all three
honest, 17 deceptive) reach nearly identical {107,401} endpoints, norm about
2192.58. Deceptive probes 4 and 19 retain {107} alone, norm about 923.9. The
surviving features were already active in each example's initial code.

## Interpretation

Stability is not monotonic with threshold: 4x has four bounded trajectories,
5x has none, 6x has twenty, and 8x has twenty-two. Thresholding changes which
features survive and can redirect later dynamics. Two active features early
in a trajectory do not guarantee stability, as 5x demonstrates.

The evidence supports threshold-sensitive dynamics and substantial shared
behavior across labels, not a robust honest/deceptive separation. There are
only three honest examples, thresholds were explored on the same pilot, and
all attractors described here belong to the SAE with modified threshold gates.
