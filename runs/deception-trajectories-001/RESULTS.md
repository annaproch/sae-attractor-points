# Deception trajectory pilot: results so far

## Completed steps

1. Selected 30 Apollo roleplaying scenarios with seed 42. Generated our own
   responses using `google/gemma-3-4b-it`, revision
   `093f9f388b31de276ce2de164bdc2081324b9767`, CPU float32, greedy decoding,
   maximum 256 new tokens. None of the 30 responses hit the token limit.
   Reference honest/deceptive completions were excluded from model inputs.
2. Judged saved responses through the API with `gpt-4o-2024-08-06`, using Apollo's
   original scenario/question/response rubric. Scores 1–2 are honest, 3–5
   ambiguous, and 6–7 deceptive. Saved scores, explanations, API model details,
   usage and input/rubric hashes. Results: **19 deceptive, 3 honest, 8 ambiguous**.
3. Excluded ambiguous examples and extracted **22 × 2560** residual vectors
   after zero-indexed block 17. Replayed saved prompt token IDs at the same
   model revision and precision. Position: the final prompt token, including
   the chat generation prefix; no generated response tokens were included.
4. Loaded the matching Gemma Scope 2 SAE:
   `google/gemma-scope-2-4b-it/resid_post/layer_17_width_16k_l0_medium`.
   It has 16,384 JumpReLU features and a configured target L0 of 60. Recorded
   checkpoint/config hashes. Reused the existing `SAE` and `rollout()` code,
   adding support for the released lowercase safetensors parameter names.
   Applied no extra input centering, normalization, or top-k pruning.
5. Encoded each activation to z0, then iterated `z_next = encode(decode(z))`,
   with a cap of 10,000 steps and independent stopping for every example.
   Saved every sparse float32 state and its probe IDs, including z0 and the
   state that triggered stopping.

## Results

| Measurement | Result |
|---|---:|
| Initial mean nonzero features | 72.64 |
| Initial mean relative reconstruction error | 0.05360 (5.36%) |
| Growth-limit stops | 22 / 22 |
| Honest growth-limit stops | 3 / 3 |
| Deceptive growth-limit stops | 19 / 19 |
| Stopping iterations | 11–14 |
| Fixed-point / period-2 candidates | 0 / 0 |
| Saved example-state pairs, including z0 | 298 |

Relative reconstruction error is `norm(decode(encode(x)) - x) / norm(x)`,
averaged across examples. The growth limit is a code norm exceeding
`1e6 * max(initial code norm, 1)`.

Other stopping checks: nonfinite entries, five consecutive small steps
(`distance <= 1e-5 + 1e-5 * previous norm`), period-2 recurrence within the
same tolerance, and the iteration cap. Convergence checks identify numerical
candidates, not proofs. All trajectories stopped on growth before reaching
any other check.

## Validation and artifacts

Loader compatibility and existing rollout tests passed. Independently replayed
all 298 saved states against the SAE recurrence, verified row IDs through each
example's terminal step, and checked saved endpoints match the final states.

- `../deception-pilot-002/`: prompts and generated responses.
- `../deception-pilot-002/judging/`: authoritative API labels and explanations.
  The original run's top-level labels and summary remain manual/unlabeled.
- `../deception-activations-001/`: activations, aligned examples/labels, provenance.
- `probes.json`: source IDs, labels and initial reconstruction measurements.
- `states/`: all code states with row mappings.
- `states_steps.csv`: per-step sparsity, norms, movement and support changes.
- `states_endpoints.json`: stop reasons, iterations and fresh residuals.
- `manifest.json`, `summary.json`: provenance and aggregate results.

Checkpoint weights and binary activation/trajectory arrays remain local under
existing Git ignore rules; text metadata and results are committed. The recorded
hashes identify the exact local artifacts. Decoded vectors can be recomputed
from saved codes and the decoder.

## Interpretation and next steps

This is a completed pipeline check. Both labels exhibit runaway code-norm growth;
we have not established a difference between them. The three honest examples
are too few for reliable held-out detection estimates.

Next compare early steps (before runaway growth dominates): support persistence,
feature recruitment, relative movement, norm growth and reconstruction error.
Compare any apparent separation with raw activations and initial SAE codes.
Collect a larger, better-balanced sample before fitting/evaluating probes, and
keep related scenarios together in any train/test split. No predictive evaluation
or causal intervention has been performed yet.

## Operational notes

An early generation attempt exhausted the original container's memory. Generation
now requires an explicit flag and a memory check, accounts for reclaimable inactive
file cache, and limits CPU threads. The successful run used the 32 GiB container.
SAE iterations load only the SAE, not Gemma. Judging and extraction are separate
commands; judging resumes completed API calls after interruptions.
