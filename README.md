# SAE trajectory experiments

Can repeated sparse autoencoder (SAE) decoding and encoding reveal useful feature
interactions, stable states, or signals of language-model behavior? This repo
tests that question with synthetic feature codes, Wikipedia activations,
deception scenarios, and activation steering.

The operation is `z_next = encode(decode(z))`, repeated from an initial feature
code. These are **SAE dynamics**, separate from the language model's generation
steps. The project was motivated by
[Navigating the Latent Space Dynamics of Neural Models](https://proceedings.iclr.cc/paper_files/paper/2026/file/2a5a41a536d3ada8fbf61a9d6fbf18d2-Paper-Conference.pdf).

**Results so far:** we found six locally stable nonzero fixed points from natural
inputs, but most unmodified trajectories hit a growth limit. Raising SAE thresholds
stabilized most or all deception-pilot trajectories at selected multipliers, with
largely shared endpoints across honest and deceptive examples. We have not
demonstrated improved deception detection or steering-quality prediction beyond
simpler initial measurements.

## Experiments and results

The synthetic and Wikipedia experiments used **Gemma 2 2B, layer 12**, with the
16,384-feature Gemma Scope SAE (`average_l0_82`). Deception and steering used
**Gemma 3 4B IT, layer 17**, with its matching 16,384-feature Gemma Scope 2 SAE
(`l0_medium`). Feature IDs are specific to each checkpoint.

| Experiment | What we attempted | Result |
| --- | --- | --- |
| [Isolated features](runs/pilot-003/RESULTS.md) | Start from 128 individual features at three strengths; look for persistent states and recruitment of other features. | Of 384 probes, **251 reached zero and 133 hit the growth limit**. No nonzero fixed points. Some growing trajectories initially shrank. |
| [Companion features](runs/background-pilot-001/RESULTS.md) | Add synthetic backgrounds selected by positive decoder-vector cosine similarity; test reinforcement and pair interactions across 784 trajectories. | Only **2 remained nonzero at step 5**, both shrinking. All decoded-norm-matched conditions reached zero by step 3. Persistence independent of total input magnitude was not established. |
| [Wikipedia activations](runs/natural-002/RESULTS.md) | Encode real model activations from 100 paragraphs and follow their trajectories. | **83 hit the growth limit; 17 approached six locally stable nonzero fixed points** after 406–1,260 iterations. Endpoints contained only 1–3 features, versus about 87 initially. Semantic preservation was not demonstrated. |
| [Deception pilot](runs/deception-trajectories-001/RESULTS.md) | Compare prompt-activation trajectories preceding honest and deceptive answers to 30 Apollo roleplaying scenarios. | API judging gave **19 deceptive, 3 honest, 8 ambiguous** answers. All 22 non-ambiguous trajectories hit the growth limit within 11–14 steps. No reliable class distinction was established. |
| [Threshold sweep](runs/deception-threshold-sweep-001/RESULTS.md) | Raise activation thresholds during iteration on the same 22 deception-pilot inputs. | **20/22 reached fixed-point candidates at 6×; 22/22 at 8×**, mostly shared across labels. Stability was non-monotonic: all hit the growth limit at 5×. Opposite-label examples shared an approximate 15-step cycle at 4×. |
| [Steering pilot](runs/steering-pilot-001/analysis/OBSERVATIONS.md) | Compare a contrastive steering direction with a norm-matched random direction: five prompts, 45 generations. | Strong interventions often caused repetition or gibberish; clean target steering was not established. **All 45 SAE trajectories hit the growth limit, including baselines.** Early trajectory metrics did not outperform initial feature count or reconstruction error in the exploratory comparison. |

**What were the inputs?** The Wikipedia pilot used 100 Wikipedia-derived
paragraphs from WikiText-2's test split (seed 42), sampling one token per paragraph
after at least 32 context tokens. The deception pilot was separate: Gemma generated
answers to Apollo roleplaying scenarios, and an API judge labeled the actual
answers. We extracted the **last prompt token's activation, before the answer**,
for the 22 non-ambiguous cases. Companion features were synthetic co-activations
chosen by vector similarity; their semantic meanings were not verified.

**How did threshold stabilization work?** Each JumpReLU feature has a learned
activation threshold. We multiplied these thresholds by a common factor during
each round trip, keeping the original starting code, weights, and surviving
activation magnitudes unchanged. At 8×, all 22 deception trajectories settled to
numerical fixed-point candidates with only 1–2 active SAE features. This modified
the SAE dynamics; it was not retraining or stabilization of the language model.
The 3× run also completed with [5 fixed-point candidates and 17 growth stops](runs/deception-threshold-sweep-001/alpha-3/summary.json).
The 4× run was interrupted at step 7,500; its cycle observations come from saved
states, not a completed endpoint summary.

## What the results establish

- **Stable states exist, but are not yet semantic explanations.** For the six
  Wikipedia endpoint groups, solved fixed points satisfy the threshold conditions
  and have locally stable feedback. Feature meanings and preservation of the
  original text's information remain untested.
- **Sparsity does not guarantee convergence.** The
  [natural-input audit](runs/natural-001/AUDIT.md) found amplifying local feedback
  at every tested starting point. Independent implementation and precision checks
  found no error explaining the observed growth.
- **Behavioral usefulness remains unproven.** There are only three honest
  deception examples. Steering labels are provisional, prompts and directions
  are few, and no held-out predictive evaluation was performed. In the steering
  sample, initial feature count and reconstruction error already separated the
  provisionally clear degradation cases perfectly.

A growth-limit stop means the code norm exceeded `1e6 * max(initial norm, 1)`;
it is not proof of indefinite divergence. Numerical fixed-point candidates from
the threshold sweep have not received the mathematical validation used for the
six Wikipedia fixed points. The next validation would need larger, balanced
behavioral datasets and held-out comparisons against raw activations, initial
SAE codes, and one-step measurements.

## Running the experiments

Third-party text, datasets, and adapted prompts are described in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), including source links,
license information, and unresolved upstream terms.

Natural-input pilot text: `data/natural-text-001/texts.jsonl` contains 100
Wikipedia-derived paragraphs sampled from WikiText-2's test split (seed 42).
Source row IDs and a manifest are saved alongside the text. Activation extraction
and trajectories are complete in `runs/natural-001`, with the 17 unfinished
trajectories continued in `runs/natural-002`.
Recreate in a fresh directory with
`.venv/bin/python scripts/prepare_text.py --output data/natural-text-002`.

Run the natural-input pilot (requires Hugging Face access to `google/gemma-2-2b`
and a logged-in account or `HF_TOKEN`):

```sh
uv pip install --python .venv/bin/python -r requirements-natural.txt
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m scripts.natural --output runs/natural-reproduction-001
```

This downloads Gemma on first use, processes up to 100 paragraphs at 256 tokens,
samples one token per paragraph after 32 context tokens, and extracts residual-post
block 12 (`hidden_states[13]`). CPU float32 is the default; use `--device cuda`
for GPU bfloat16 or `--limit 2 --steps 5` for a smoke run. Saves raw activations,
text/token provenance, initial reconstruction errors, and complete SAE trajectories.
See the [implementation audit](runs/natural-001/AUDIT.md) for completed checks
and their limits.

Commands for the later experiments:

- [Deception generation, judging, extraction, and threshold sweeps](DECEPTION_PILOT.md)
- [Steering generation, batching, resume, and trajectories](STEERING_PILOT.md)

- [Earlier synthetic-phase discussion and paper comparison](CONVERSATION_SUMMARY.md)
- [Research plan and complete conversation bibliography](SAE_INTERACTION_PLAN.md)
- [Initial Gemma Scope results](runs/pilot-003/RESULTS.md)
- [Trajectory runner](scripts/trajectories.py)
- [Matched-background pilot runner](scripts/backgrounds.py)
- [Related-background pilot results](runs/background-pilot-001/RESULTS.md)

Full sparse trajectories are saved locally under `runs/`; binary artifacts and
checkpoint weights are excluded from Git. A Git checkout contains the text
metadata and reports, but does not include those binary artifacts.

Launch the related-background pilot from the repository root:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m scripts.backgrounds \
  --checkpoint checkpoints/gemma-layer12-16k-l0-82.npz \
  --output runs/background-reproduction-001
```

Defaults: five steps, targets 120/11341/12692, pairs 120+6712 and 120+2499,
top-128 positive neighbors per target, nearest and two seeded sampled subsets
of sizes 1/4/16/64, full-pool and empty-background controls, and target strengths
0.75/1.25 times isolated self-activation boundaries. Pair pools use the union
of the two ranked pools and may contain up to 256 features. Background code
amplitudes are equal across members, set to 0.25 times target A's boundary.
A second condition matches each nonempty background's decoded norm to that
family's nearest single neighbor. These are synthetic, uncalibrated inputs.

The runner preserves every state and single-target difference or pair interaction
contrast, with metadata in `groups.json`, `neighbor_pools.json`,
`static_baselines.json`, and `manifest.json`, and per-step metrics in `metrics.csv`.
Contrast rows are indexed by group ID; state rows by probe ID. The fixed five-step
horizon keeps all four conditions aligned. A nonfinite state or growth guardrail
aborts the run instead of inventing subsequent states. Existing output directories
are never overwritten; choose another output name for a rerun.

Implementation tests:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m unittest discover -s tests -v
```
