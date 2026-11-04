# SAE project: conversation and results summary

Updated 2026-10-18. This summarizes decisions and findings rather than preserving
a verbatim transcript. Detailed methods and bibliography are in
[SAE_INTERACTION_PLAN.md](SAE_INTERACTION_PLAN.md).

## Research question and intended paper

Can repeated SAE decoding and encoding reveal useful feature interactions,
replacement, context sensitivity, or stable states beyond static similarities
and one-step responses?

The user meant **Navigating the Latent Space Dynamics of Neural Models**, not
the Gemma Scope release paper. My earlier answer about the training objective
of Gemma Scope answered the wrong paper.

The intended paper argues for local contraction and convergence to attractors,
not universal zero collapse or explosion. Table 3 motivates contraction through
sigmoid saturation or inactive ReLU derivatives. Section 6.1 proposes LLM SAE
analysis as future work. [Paper, Table 3 and §6.1](https://arxiv.org/html/2505.22785v4#A3.T3)

Our assessment: a sparse Jacobian need not have gain below one. Our JumpReLU
checkpoint differs from the table's examples and is discontinuous at its gates.
Synthetic seeds need not lie near training activations. We therefore cannot
assume global contraction or meaningful nonzero attractors. Zero is a measured
fixed point here; growth elsewhere does not refute a local contraction claim.

## Implementation and controls

- Checkpoint: Gemma Scope base Gemma 2 2B, layer 12, width 16,384, average L0 82.
- Released weights encode raw activations directly. Do not subtract decoder
  bias before encoding or normalize intermediate trajectory states.
- All nonzero float32 state entries are retained locally, without top-k
  truncation. Binary states and checkpoint weights are Git-ignored.
- Six tests cover released inference, trajectory handling, matched backgrounds,
  neighbor selection, reproducibility and contrast behavior. Saved contrasts and
  initialization norms were independently checked.
- No real-activation dataset, verified semantic feature labels or LLM causal
  interventions have been used. Feature IDs are checkpoint-specific.

## Isolated-feature results

[pilot-003 report](runs/pilot-003/RESULTS.md): 128 features at three amplitudes
gave 384 probes: 251 collapsed to zero and 133 reached the growth guardrail.
No nonzero fixed point was found. The guardrail is not proof of unbounded growth.

The most interesting observations:

- 66 growing paths initially shrank. Feature 120 at the largest strength fell
  from norm 81.38 to about 52.89 before growing, with early recruits 6712/2499.
- Growing endpoints became dense: median 4,286 active features. Their last-step
  direction cosine had median 0.9925, while norms multiplied by median 16.07.
- Endpoint-dominant features recurred: 7488 in 30 paths, 1468 in 25, 14396 in 9.
  The seed was never the largest endpoint component. Endpoint pair cosine median
  0.5103 argues against one shared terminal direction.
- Twenty collapsing paths recruited other features. Features 11341 and 12692
  provided longer transients, reaching zero at steps 22 and 20 respectively.

Interpretation: several expanding mixtures and structured transients, with no
established semantic attractor or asymptotic direction.

## Related-background pilot

The user launched the implemented pilot. See
[background-pilot-001 report](runs/background-pilot-001/RESULTS.md).
It used 784 trajectories in 280 groups over five steps: targets 120/11341/12692,
pairs 120+6712 and 120+2499, nearest and sampled positive-cosine neighbor subsets,
full pools, empty controls, and equal-amplitude versus decoded-norm-matched
backgrounds. Pair pools are unions and can exceed 128 neighbors.

Nonzero trajectory counts at steps 1–5 were **302, 111, 24, 9, 2**. All
norm-matched trajectories reached zero by step 3. Only full-pool, equal-amplitude
joint additions of 120+6712 and 120+2499 survived step 5 at target strength 1.25.
Counts include duplicate controls and are not independent statistical samples.

Both survivors were shrinking. For 120+6712, joint amplitudes declined from
(11.01, 11.22) at step 1 to (4.20, 4.28) at step 5. Background alone switched
off at step 3, separate additions by step 5. The pair contrast was approximately
zero for two steps and became substantial when paths crossed different gates.
The 120+2499 joint path lost feature 2499 at step 5.

The largest eigenvalues of the restricted two-feature feedback matrices are
0.92354 and 0.91504, with negative biases. This supports delayed switch-off
within those supports, not a guarantee for the full map. No later steps ran.

Numerical correction: the initial 1e-6 tolerance counted 66/59 pair contrasts at
steps 1/2. Tolerances from 1e-5 to 1e-3 give **62/57**; the excluded effects are
consistent with float32 roundoff. Later counts remain 10/4/2.

## Decisions and next questions

Prefer synthetic related backgrounds before real activations, retain complete
trajectories for later visualization, and keep explanations concise without
LaTeX. The user prefers to launch experiments themselves; supply commands.

The proposed next test holds decoded initialization norm fixed while changing
companion identity, with a norm sweep for fixed subsets. This separates total
input magnitude from specific companion effects. It has not been run.
Interpretability value ultimately requires prediction beyond decoder cosine,
directed coupling and first-step gating, validated on examples or interventions.

Commits: `94b3d4c` contains the initial pilots and plan; `e00666e` contains the
background implementation and initial analysis. Subsequent trajectory commentary
and this summary are documentation updates, not new experiments.
