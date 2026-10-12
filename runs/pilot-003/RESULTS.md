# Gemma Scope trajectory pilot

Completed 2026-10-12. No nonzero attractors were found in this synthetic sweep.

Checkpoint: `google/gemma-scope-2b-pt-res`, `layer_12/width_16k/average_l0_82/params.npz`, revision `fd571b47c1c64851e9b1989792367b9babb4af63`. Architecture: JumpReLU. Dictionary width 16,384, activation dimension 2,304. CPU float32, no additional normalization or centering.

## Results

128 randomly selected feature IDs (seed 42), each probed at three synthetic strengths. Starting strength is a multiple of the amplitude needed for the feature to survive its first isolated round trip. A zero-code baseline is additional.

| Strength multiplier | Feature probes | Reached zero | Hit growth limit |
|---|---:|---:|---:|
| 1.25 | 128 | 128 | 0 |
| 4 | 128 | 117 | 11 |
| 16 | 128 | 6 | 122 |
| Total | 384 | 251 | 133 |

The zero baseline also stays zero. All 252 fixed-point candidates including that baseline are exactly zero, with zero fresh round-trip residual. The 133 growth cases were stopped when code norm exceeded one million times the larger of initial norm and one; this is observed growth to a guardrail, not a proof of unbounded divergence. No period-two candidates were detected. Termination occurred by iteration 28 with a maximum allowed horizon of 200.

Three small nonnegative perturbations of the zero code returned to zero. This provides limited empirical local stability evidence, not a global contraction guarantee. Active-coordinate perturbations cannot be applied to zero because it has no active coordinates.

The earlier threshold-based pilot (`../pilot-002`) used 32 features at 1.25, 2 and 4 times their thresholds; all 96 feature probes and the zero baseline collapsed. That run motivated accounting for encoder biases when choosing starting amplitudes.

## Interpretation

### Inspection of full trajectories

Reproducible analysis: `scripts/analyze_trajectories.py runs/pilot-003`; numerical output: `trajectory_analysis.json`. These are descriptive in-sample findings from synthetic seeds, not held-out statistical tests.

1. **Initial contraction does not predict eventual collapse.** Of 133 growth trajectories, 66 shrink on their first step. Feature 120 at 16 times its self-activation boundary starts at norm 81.38, falls to about 52.89 at steps 4–5, and then grows. Its active count stays at 11 through step 8, rises to 197 at step 18, and reaches 4,802 at the cutoff. Early recruits include 6712 and 2499. This is a useful candidate for testing whether removing a recruit changes the subsequent growth.
2. **Growing trajectories become directionally persistent.** Median cosine between the final two states is 0.9925 (range 0.9651–0.9990), while the median norm multiplier is 16.07 per last step (range 9.19–22.13). This is compatible with increasingly stable expanding directions, but two-state agreement does not establish asymptotic directional convergence. No normalization was applied during rollout.
3. **Several different seeds end with the same dominant feature.** Feature 7488 dominates 30 of 133 growing endpoints, 1468 dominates 25, and 14396 dominates 9. The original seed feature is never the largest final component, although it remains nonzero in 73 cases. Pairwise endpoint-direction cosine has median 0.5103 and upper quartile 0.8613, so there is no evidence for one universal final direction. Shared dominant features are candidate hubs, not established semantic attractors.
4. **Growth destroys sparsity in this synthetic setting.** Growing endpoints have 2,397–5,537 active features (median 4,286) out of 16,384. This makes late dynamics a questionable proxy for natural sparse encodings and favors studying early steps.
5. **Some collapse paths contain structured transients.** Twenty of 251 collapsing trajectories recruit additional features. Feature 11341 at the largest strength recruits 20 active features, with prominent recruits 3993 and 9230. It gradually loses support and reaches zero at step 22. Feature 12692 also recruits 20 features and reaches zero at step 20. Most collapse is much faster: median time to first zero is two steps.

Priority follow-ups: inspect early recruitment around feature 120, test matched ablations of recruits 6712 and 2499, compare shared endpoint directions against initial decoder similarity, and repeat with real-context seeds. Semantic labels have not been retrieved or validated; feature IDs here are exact checkpoint-local indices.

These isolated synthetic feature codes show collapse at low strengths and rapid amplification at high strengths. We have not established meaningful semantic attractors. The behavior is consistent with thresholding and affine feedback repeatedly eliminating or amplifying activity; it must not be interpreted as the LLM's own reasoning dynamics.

The next useful analysis is of the transient states: which features are recruited before collapse or growth, and how recruitment changes around an amplitude boundary. Pairs and real-context codes may behave differently. No conclusions about those settings have been tested yet.

## Saved artifacts

- `manifest.json`: checkpoint SHA-256, revision, seed, precision, equations, thresholds and run configuration.
- `probes.json`: feature IDs, starting amplitudes and calibration mode.
- `states/step_XXXXX.npz`: exact nonzero float32 sparse codes at each step, including step zero.
- `states/probe_ids_XXXXX.npy`: row-to-probe mapping for that step; completed probes disappear from later files.
- `states_steps.csv`: per-step norms, active counts, drift, residuals and support changes.
- `states_endpoints.npz` and `states_endpoints.json`: final states and termination diagnostics.
- `perturbation_states/`, `perturbation_probes.json`, and `stability.json`: all local-stability probes and outcomes.
- `summary.json`: machine-readable results and endpoint grouping.

About 18 MB of artifacts are stored for this run. Binary trajectories are retained locally and ignored by Git; copy the full run folder when transferring experiments. Keep the pinned checkpoint to decode saved codes later. The failed `pilot-001` is incomplete and not an analysis source.

## Reproduce

From the repository root, install `requirements.txt` in a virtual environment and obtain the pinned public checkpoint. Then run:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python scripts/trajectories.py \
  --checkpoint checkpoints/gemma-layer12-16k-l0-82.npz \
  --output runs/reproduction-001 \
  --features 128 --steps 200 --seed 42 \
  --amplitude-mode self_activation --multipliers 1.25,4,16 \
  --revision fd571b47c1c64851e9b1989792367b9babb4af63
```

The output directory must not already exist. To read a state:

```python
import numpy as np
from scipy import sparse

codes = sparse.load_npz('runs/pilot-003/states/step_00001.npz')
probe_ids = np.load('runs/pilot-003/states/probe_ids_00001.npy')
```

Validation: three unit tests pass (released bias/threshold behavior, period-two detection and lossless storage, convergence on a contracting toy map). Saved states at steps 1, 2 and 5 were independently replayed through the checkpoint and matched the stored next states within float32 tolerance. No external SAE library equivalence test or real-activation calibration has yet been performed.
