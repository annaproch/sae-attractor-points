# Steering trajectory pilot

Run from the project root:

```sh
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m scripts.steering_pilot \
  --run --output runs/steering-pilot-001 \
  --alphas 0 1 2 4 8 --limit 5 --max-new-tokens 96 --steps 100
```

Uses pinned Gemma 3 4B IT CPU float32 and the existing layer-17 16k medium SAE. Keeps the existing 24 GiB available-memory guard and two Torch threads. Run sequentially in the 32 GiB container; the LLM is released before loading the SAE. No new dependencies or SAE downloads are needed. Model weights must already be cached or accessible through Hugging Face.

The contrastive statement pairs are adapted from https://github.com/jonathanbostock/activation-steering-sweep. That experiment reports steering at layer 25; this pilot extracts a fresh difference-of-means vector at layer 17 because our matching SAE is there. Published strengths do not establish effective strengths here. The source's hawk contrast is retained for comparability; it is not a correct taxonomic ground-truth label for caracaras.

Five prompts comprise three caracara questions and two unrelated quality checks. Conditions are baseline plus contrastive and norm-matched seeded Gaussian directions at each nonzero alpha (45 generations by default). The same direction is applied to every prompt and generated position, with greedy decoding. Capture is the actual last prompt token after intervention on the first generation forward pass. It predicts a sustained intervention; it does not isolate a one-token causal effect. No steering is applied during the subsequent SAE iterations.

Outputs:

- `responses.jsonl`: full continuations, exact token IDs, condition, strength, truncation flag.
- `labels.csv`: blank fields for independently reviewing target effect and degeneration. Wrong answers, refusals, or truncated answers are not automatically degeneration.
- `steering_vector.npy`, `random_vector.npy`, `contrastive_prompts.json`: reproducible interventions.
- `activations.npy`, individual activation files, `probes.json`: aligned post-intervention last-prompt-token residuals.
- `states/`, `states_steps.csv`, `states_endpoints.*`: every sparse float32 trajectory state, original SAE thresholds and existing independent stopping checks.
- `manifest.json`, `summary.json`: provenance, configuration and endpoint records; no inferred success labels.

First inspect responses and label coherent target steering, coherent unsuccessful steering, and degeneration separately. Compare paired trajectories for the same prompt. Compare with random controls and initial activation/intervention norms; larger norms alone are not evidence of a useful trajectory predictor. These three near-duplicate target prompts are exploratory, not a held-out validation set. Use additional prompts and directions before claiming predictive performance.

If generation completed but trajectories were interrupted, use a fresh trajectory directory (preserve or move any partial `states` outputs first) and run:

```sh
OPENBLAS_NUM_THREADS=2 .venv/bin/python -m scripts.steering_pilot \
  --trajectories-only --output runs/steering-pilot-001 --steps 100
```

This accepts only a `generation_complete` manifest and verifies the SAE hash. Generation itself does not resume; a fresh output directory is required for a new run.

Validation: hook tests verify all-position edits, first-pass-only capture, tuple handling, and layer mismatch rejection; existing rollout and extraction tests pass. CLI preflight passes. Full model generation is left for the user's container run.

## Batched generation and resume

Generation now batches up to three conditions for the same prompt by default (`--batch-size 1` through `4` supported). Each row receives its own steering vector and strength; prompt lengths are identical within a batch. Finished continuations are trimmed at their first EOS so padding from longer-running rows is not saved as generated content. Batch timing is printed. Greedy decoding remains enabled, but batching may produce small numerical differences from serial execution.

To switch an active serial run: stop it with Ctrl-C and wait for the process to exit, then rerun the original command with `--resume --batch-size 3`. Completed JSONL responses and aligned individual activation files are retained; unfinished generation is repeated. Saved vectors are reused. Model revision, SAE hash, seed, strengths, prompt limit and token limit must match. Do not run both processes together.

This supersedes the earlier statement that generation cannot resume. Only manifests in `generating` state are accepted for generation resume. Original older runs without a saved prompt limit are assumed to have used the default five prompts. A malformed partial JSONL line is rejected rather than silently discarded. Manual labels already present are preserved. The existing `--trajectories-only` path remains available after generation completes.

Validation after batching: 13 focused tests pass, including batch-versus-serial hook equivalence, per-row capture, first-pass-only capture and EOS trimming. Full-model throughput has not been benchmarked.
