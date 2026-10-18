https://proceedings.iclr.cc/paper_files/paper/2026/file/2a5a41a536d3ada8fbf61a9d6fbf18d2-Paper-Conference.pdf

# SAE trajectory experiments

Natural-input pilot text: `data/natural-text-001/texts.jsonl` contains 100
Wikipedia-derived paragraphs sampled from WikiText-2's test split (seed 42).
Source row IDs and a manifest are saved alongside the text. This is text only;
Gemma tokenization and activation extraction have not been run.
Recreate in a fresh directory with
`.venv/bin/python scripts/prepare_text.py --output data/natural-text-002`.

Run the natural-input pilot (requires Hugging Face access to `google/gemma-2-2b`
and a logged-in account or `HF_TOKEN`):

```sh
uv pip install --python .venv/bin/python -r requirements-natural.txt
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m scripts.natural --output runs/natural-001
```

This downloads Gemma on first use, processes up to 100 paragraphs at 256 tokens,
samples one token per paragraph after 32 context tokens, and extracts residual-post
block 12 (`hidden_states[13]`). CPU float32 is the default; use `--device cuda`
for GPU bfloat16 or `--limit 2 --steps 5` for a smoke run. Saves raw activations,
text/token provenance, initial reconstruction errors, and complete SAE trajectories.
The new runner has not yet been validated with a real Gemma forward pass.

- [Conversation, findings and paper comparison](CONVERSATION_SUMMARY.md)
- [Research plan and complete conversation bibliography](SAE_INTERACTION_PLAN.md)
- [Initial Gemma Scope results](runs/pilot-003/RESULTS.md)
- [Trajectory runner](scripts/trajectories.py)
- [Matched-background pilot runner](scripts/backgrounds.py)
- [Related-background pilot results](runs/background-pilot-001/RESULTS.md)

The first synthetic sweep found zero-code collapse or growth, with no nonzero attractors. Full sparse trajectories are saved locally under `runs/`; binary artifacts and checkpoint weights are excluded from Git.

Launch the related-background pilot from the repository root:

```sh
OPENBLAS_NUM_THREADS=4 .venv/bin/python -m scripts.backgrounds \
  --checkpoint checkpoints/gemma-layer12-16k-l0-82.npz \
  --output runs/background-pilot-001
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
