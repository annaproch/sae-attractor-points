# Active-feature spectral analysis

Read-only analysis of the saved trajectories and matching Gemma Scope 2 SAE.
No modified rollouts were run. Results below were computed with float64 matrix
products and NumPy eigenvalue decomposition; checkpoint weights are float32.

## Method and interpretation

For row-vector codes, the update is
`z_next = JumpReLU(z @ W_dec @ W_enc + b_dec @ W_enc + b_enc)`.
For a fixed active-feature set S, inspect
`A_S = W_dec[S, :] @ W_enc[:, S]` and its spectral radius
`rho = max(abs(eigvals(A_S)))`.

A radius below 1 implies stability of the affine recurrence with that fixed
support. A valid SAE fixed point must also satisfy the activation thresholds.
Changing supports, threshold discontinuities, and transient amplification mean
this is not a global convergence guarantee. These are SAE iterations on the
last prompt token, not model generation steps.

## Full active sets

None of the 132 active sets (22 examples, steps 0–5) had rho below 1.

| Step | Honest min / median / max | Deceptive min / median / max |
|---|---|---|
| 0 | 1.436 / 1.556 / 1.599 | 1.342 / 1.457 / 1.667 |
| 1 | 1.355 / 1.524 / 1.542 | 1.256 / 1.372 / 1.495 |
| 2 | 1.261 / 1.537 / 1.546 | 1.225 / 1.356 / 1.478 |
| 3 | 1.319 / 1.533 / 1.590 | 1.278 / 1.393 / 1.701 |
| 4 | 1.542 / 1.740 / 1.819 | 1.338 / 1.680 / 2.381 |
| 5 | 1.756 / 2.095 / 2.445 | 1.471 / 2.071 / 3.714 |

There are only 3 honest versus 19 deceptive examples. These results do not
establish a label-specific stability pattern. All original rollouts hit the
growth limit at steps 11–14; a finite growth-limit stop is not itself proof of
indefinite mathematical divergence.

## Strongest-feature subsets at step 0

Features were ranked by raw code activation, not decoder-weighted contribution.
Each row uses the corresponding principal submatrix of A_S.

| Retained features | Radius range | Number below 1 |
|---|---|---|
| Top 1 | 0.816–0.869 | 22/22 |
| Top 2 | 0.876–0.926 | 22/22 |
| Top 3 | 0.931–0.933 | 22/22 |
| Top 4 | 0.936–0.998 | 22/22 |
| Top 5 | 1.003–1.005 | 0/22 |
| Top 10 | 1.003–1.222 | 0/22 |
| Top 20 | 1.184–1.296 | 0/22 |
| Top 40 | 1.167–1.342 | 0/22 |
| Top 60 | 1.194–1.509 | 0/22 |

Interactions among a few strong features already produce instability: weak
features are not the sole cause. Selection by activation is not a search for
the largest stable subset, and spectral radius need not vary monotonically
with subset size.

These are subsets of existing trajectories, not new top-k trajectories.
At step 5, only 21/22 top-1 subsets have radius below 1, illustrating that
dynamic selection can encounter unstable features.

## Candidate experiments (not run)

1. **Frozen initial top-3 support:** select the three strongest initial feature
   IDs, start with only those features, and zero all other coordinates after
   EVERY encoding. Retained magnitudes can change. This keeps the permitted
   feature identities fixed; thresholding can still deactivate them.
   The initial three-feature matrices all have radius around 0.93, but actual
   thresholded convergence must still be checked.
2. **Initial-only pruning:** prune z0 once, then allow every feature to activate
   in later iterations. This is a different experiment and is NOT frozen
   support. The top-3 matrix does not describe its subsequent dynamics.
3. **Dynamic top-k:** retain the k largest activations after every encoding,
   allowing feature identities to change. Initial subset stability does not
   guarantee stability of this switched process.
4. **Update shrinkage:** use `z_next = alpha * encode(decode(z))`.
   For an unchanged support, eigenvalues scale by alpha. Alpha=0.5 makes all
   measured initial-support radii less than 1; alpha=0.25 does so for all
   measured full supports through step 5. Modified trajectories can visit
   different supports, so neither value is a global guarantee. This changes
   the original dynamics and should be compared with the unmodified baseline.

All 22 initial matrices have a positive real eigenvalue above 1. Ordinary
averaging `z_next = (1-alpha)*z + alpha*encode(decode(z))`, with 0<alpha<=1,
cannot stabilize that fixed-support mode: its eigenvalue is
`1-alpha+alpha*lambda > 1`. Averaging and update shrinkage are distinct.

Suggested next comparison: unchanged baseline versus shrinkage, with frozen
top-3 as a more restrictive diagnostic. Track norms, supports, reconstruction
error, and actual convergence rather than relying on eigenvalues alone.
