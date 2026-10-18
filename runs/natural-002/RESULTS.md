# Natural-input continuation

All 17 unfinished natural-input trajectories reached the nonzero fixed-point
candidate stopping criterion. Total iteration counts, including the original
200 steps, ranged from 406 to 1260 (median 985). None reached zero or the growth
guardrail during this continuation. The other 83 original trajectories had
already hit the growth limit and were not continued.

The endpoints fall into six active-support groups:

| Features | Trajectories | Solved amplitudes | Local spectral radius |
| --- | ---: | --- | ---: |
| 6810 | 4 | 10.1099 | 0.99319 |
| 2620, 6810 | 2 | 4.5865, 9.5943 | 0.99338 |
| 6810, 7507 | 5 | 9.3411, 7.5652 | 0.99321 |
| 2620, 6810, 7507 | 3 | 4.0788, 8.8909, 7.4844 | 0.99339 |
| 2620 | 2 | 6.1877 | 0.97962 |
| 2620, 7507 | 1 | 5.5463, 7.6788 | 0.97968 |

For each observed support, we solved the affine fixed-point equations using
the released weights promoted to float64. Every solution satisfies its active
and inactive threshold inequalities with positive margins, and every active
feedback matrix has spectral radius below one. Thus these are six locally
asymptotically stable nonzero fixed points of the mathematical map represented
by these weights. This is stronger evidence than small step size alone. It
does not establish global attraction or semantic meaning.

The runner used float32 and stopped after five sufficiently small steps
(atol=rtol=1e-5). Saved endpoint residuals are about 2.3e-5 to 1.3e-4;
distances to the solved points reach 0.0314. Slow contraction, particularly
the approximately 0.993 feedback eigenvalues, explains why tiny step sizes
can coexist with a remaining endpoint error. Grouping is by active support
and the corresponding solved fixed point, not strict equality of saved codes.

The original natural codes averaged 87.07 active features; the convergent
endpoints contain only 1–3. These are highly simplified persistent mixtures,
not demonstrated preservation of the original text's meaning. Feature semantics
and perturbation-return behavior have not been tested here. No extra trajectories
were run for this analysis.

Full numeric results are in `analysis.json`. Every computed trajectory state
remains in `states/`; `continuation.json` maps rows back to the original probes.
