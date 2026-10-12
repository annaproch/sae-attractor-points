# Compact research handoff

## Current user request and preferences

- Latest request: update Markdown with the subset experiment and interpretability rationale, then commit project files.
- Plan update is complete. Next experiment is planned, not run. Do not launch new experiments merely because this handoff exists.
- No LaTeX in user-facing responses. Keep replies and tool outputs concise. User noticed excessive context consumption from web results and large printed arrays.
- Retain full trajectories for later visualizations. User prefers synthetic related backgrounds before real activations.

## Files

- `SAE_INTERACTION_PLAN.md`: detailed plan, implementation status, complete bibliography of works referenced in written conversation (20 papers plus one research post).
- `scripts/trajectories.py`: NumPy/SciPy CPU round trips, sparse trajectory storage, stopping and candidate perturbation checks.
- `scripts/analyze_trajectories.py`: reads saved states and summarizes transient behavior.
- `runs/pilot-003/RESULTS.md`: main measured findings, reproducibility instructions and caveats.
- `runs/pilot-003/trajectory_analysis.json`: computed metrics and detailed selected trajectories.
- `.venv/`: Python environment, NumPy 2.5.4 and SciPy 1.18.1; dependencies in `requirements.txt`.
- `tests/test_trajectories.py`: three passing tests; saved states also replayed successfully at steps 1, 2, 5.

## Exact checkpoint and implementation

Local: `checkpoints/gemma-layer12-16k-l0-82.npz`.
HF: `google/gemma-scope-2b-pt-res`, `layer_12/width_16k/average_l0_82/params.npz`.
Revision: `fd571b47c1c64851e9b1989792367b9babb4af63`.
SHA-256: `afae57c7fdfe6faace4b97d9fe9a184deb08bda8852a4c40b308cf6c72ed8384`.
16,384 features, 2,304 activation dimensions, JumpReLU. Base Gemma 2 2B model; LLM itself not downloaded.

Published released-weight equations: `encode(x) = JumpReLU(x @ W_enc + b_enc)`, `decode(z) = z @ W_dec + b_dec`. Training-time normalization and centering are already folded into released weights. Do NOT subtract b_dec before encoding. Do NOT normalize intermediate states.

## Completed experiments

- `pilot-001`: incomplete final serialization; superseded by `pilot-002`.
- `pilot-002`: 32 random features, 3 threshold-relative amplitudes; all 96 probes plus zero baseline collapse to zero.
- `pilot-003`: 128 random features (seed 42), amplitudes 1.25/4/16 times each isolated self-activation boundary. Among 384 feature probes: 251 collapse, 133 hit growth limit; zero baseline also stays zero. No nonzero fixed points found.
- Growth guardrail: code norm exceeds `1e6 * max(initial norm, 1)`. This is arbitrary stopping, NOT proof of unbounded divergence.
- Candidate fixed points: five consecutive small steps, atol=rtol=1e-5. Three small perturbations of zero returned. No broad basin validation.
- All nonzero float32 code values saved at every step as CSR NPZ, with companion row/probe ID arrays. No top-k truncation. Weights and binary artifacts retained locally but Git-ignored.

## Interesting measured trajectory findings

- 66/133 growth trajectories initially shrink.
- Feature 120 at largest strength: norm 81.38 initially, about 52.89 at steps 4–5, then grows; 11 active features through step 8, 4,802 at cutoff. Early recruits: 6712 and 2499.
- Median final-step cosine in growing paths: 0.9925; median final-step norm multiplier: 16.07. Directional persistence, NOT a fixed attractor or proven asymptotic convergence.
- Final dominant feature 7488 occurs in 30 growth paths, 1468 in 25, 14396 in 9. Seed feature never remains largest. Endpoint pair cosine median 0.5103: heterogeneous directions.
- Growing endpoints have median 4,286 active features: late dynamics are far from sparse synthetic seeds.
- 20/251 collapsing paths recruit others. Feature 11341 recruits 20 (including 3993, 9230), then reaches zero at step 22; feature 12692 reaches zero at step 20. Median collapse time is 2.
- No semantic labels verified for these IDs; no real-activation or causal LLM validation.

## Agreed next experimental direction

Use positive signed decoder-cosine neighbors as synthetic backgrounds, around features 120, 11341, 12692. From a top-128 positive-neighbor pool, test nearest-ranked and reproducibly sampled subsets of 1/4/16/64; include the full pool as an additional condition. For pairs, use ranked neighbor union excluding tested features. Compare background, +A, +B, +A+B with the same background in all conditions. Save trajectories and baseline-corrected interaction contrasts. Focus steps 1–5, sweep modest synthetic strengths, and control decoded initialization norm across background sizes. Never imply these are natural activations. Optional recruit ablations for 120: 6712 and 2499.

Interpretability goal: diagnose replacement, reinforcement/suppression and context sensitivity; prioritize groups for joint inspection. Demonstrated value requires predicting shared activating examples or intervention effects beyond static and one-step baselines. Synthetic SAE dynamics alone do not establish concept meanings or model computation.

Novelty baseline: for affine decoder and gated linear encoder, round trips equal `gate(M*z+bias)`, where M contains directed decoder-to-encoder dot products. Within a gate region dynamics are affine. Need evidence beyond decoder cosine, directed coupling and one-step gating. PolySAE already studies learned feature interactions; our proposal probes unchanged SAEs.

Safety features: related research exists for Gemma-2-2B-IT, but none verified for our exact checkpoint. Latest two references added to plan. No cross-checkpoint feature-ID transfer.

## Operational notes

Use `OPENBLAS_NUM_THREADS=4 .venv/bin/python ...`. CPU only in current environment. Network sandbox requires escalation for package/checkpoint downloads; these were approved previously for specific commands. No subagents unless explicitly requested. Existing output directories are protected from overwrite by runner.
