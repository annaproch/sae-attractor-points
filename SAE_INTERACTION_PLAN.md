# SAE feature interactions through repeated reconstruction

Research plan, updated 2026-10-12. Initial synthetic single-feature experiments are now implemented and running; pair and real-context experiments remain planned.

Writing preference: use plain-text equations and notation, not LaTeX.

## Motivation and evidence

The starting paper is [Navigating the Latent Space Dynamics of Neural Models](https://arxiv.org/html/2505.22785v4), by Fumero, Moschella, Rodolà and Locatello (ICLR 2026). It studies repeated decoding and encoding, interpreting stable endpoints and trajectories as probes of an autoencoder. Section 6.1 explicitly proposes applying this to SAEs for LLM interpretability; it does not provide an empirical study of modern LLM SAEs. Appendix C discusses sparsity as a source of contractive pressure.

Our question: what do repeated round trips reveal about feature interactions, beyond static feature similarity?

Define `F(z) = encode(decode(z))`. Start with a sparse code and repeatedly apply F. These are dynamics of the SAE, not the LLM's own computation. A stable endpoint is not automatically a semantic concept, and an interaction is not automatically an LLM causal connection.

## Recommended pretrained SAE

Start with the original Gemma Scope release for the Gemma 2 2B base model:

- Repository: [`google/gemma-scope-2b-pt-res`](https://huggingface.co/google/gemma-scope-2b-pt-res).
- Exact checkpoint: [`layer_12/width_16k/average_l0_82/params.npz`](https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main/layer_12/width_16k/average_l0_82).
- Architecture: JumpReLU; 16,384 features; residual-stream SAE. The checkpoint folder is approximately 302 MB.
- Use SAELens's supported loader after verifying the exact checkpoint mapping. Pin its version and the checkpoint revision, rather than relying on a changing canonical alias.
- Verify the hook, activation dimension, normalization, biases and thresholds from the loaded configuration before collecting model activations.

Why this choice: moderate dictionary size, public weights, an established analysis ecosystem, and no L1 amplitude-shrinkage penalty of the classical ReLU SAE kind. JumpReLU still has thresholds and can produce nonlinear or unstable round-trip behavior. This is a practical pilot choice, not a claim that it is the best available SAE.

The synthetic phase needs only SAE weights. Real-context calibration and downstream validation need the matching Gemma model or compatible cached activations. Access to the base model may require accepting its terms; record that separately from public SAE access.

Smaller fallback: [GPT-2 Small residual SAEs](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted), with SAELens release `gpt2-small-res-jb` and SAE ID `blocks.8.hook_resid_pre`. This is useful for inexpensive end-to-end development, but its architecture and shrinkage behavior must be treated as a different experimental condition.

References: [Gemma Scope paper](https://arxiv.org/abs/2408.05147), [SAELens documentation](https://decoderesearch.github.io/SAELens/), [supported SAEs](https://decoderesearch.github.io/SAELens/sae_table/).

## What feature comparisons currently measure

There is no single feature-similarity metric. Use several complementary baselines:

| Comparison | What it measures | Limitation |
|---|---|---|
| Decoder-vector cosine | Whether two features write similar directions into the activation space | Symmetric; ignores encoder thresholds, magnitudes and actual use |
| Encoder-vector cosine | Whether features detect similar input directions | Ignores biases, thresholds and input distribution |
| Decoder-to-encoder dot product | How writing feature i changes the preactivation of feature j | Directed and local; gating and normalization affect realized responses |
| Activation correlation on shared tokens | Whether feature strengths vary together in observed contexts | Sensitive to rarity and shared context; association is not causation |
| Active-set overlap, Jaccard, conditional activation probability | Whether features fire together, including asymmetric dependence | Requires adequate event counts and frequency controls |
| Top-activating examples and semantic descriptions | Whether features appear to describe similar content | Descriptions are fallible; inspect examples |
| Matched intervention effects | Whether features have similar downstream effects | More expensive; depends on intervention strength and context |

Decoder cosine is an important starting point, not a complete answer. Direct cosine across different models or unaligned layers is generally not meaningful because their coordinate systems differ. Cross-SAE matching may need aligned activations, matching algorithms or subspace comparisons rather than one-to-one matches.

Relevant work: [The Geometry of Concepts](https://arxiv.org/abs/2410.19750) studies decoder geometry and co-occurrence structure; [Quantifying Feature Space Universality](https://github.com/wlg1/univ_feat_geom) uses activation matching and representation comparisons. A newer [semantic optimal transport preprint](https://arxiv.org/abs/2605.28567) compares activation-weighted distributions rather than only decoder vectors.

## Hypotheses

1. Some features recruit or replace other features during round trips despite low decoder cosine.
2. Some pairs trigger features neither member recruits alone, or suppress features recruited individually.
3. Different feature IDs can converge to similar representations, suggesting possible redundancy or ambiguity to investigate.
4. Small changes in mixture strength can cross gating boundaries and produce different trajectories.
5. Early trajectory instability may predict unreliable interpretations or interventions beyond ordinary reconstruction error.

These are hypotheses, not established results. A null result, or convergence explained by shrinkage and thresholds, is informative.

## Single-feature probes

Support arbitrary feature IDs and run a batched sweep over every feature at a small number of strengths. Include a zero-code trajectory as a baseline; decoder and encoder biases mean F(0) need not equal zero.

For feature i, initialize all entries to zero except `z[i] = a`. Save the initial state and every subsequent state.

Use several strengths. Prefer positive-activation quantiles measured on held-out real contexts, for example the 25th, 50th and 90th percentiles conditional on the feature firing. Report sample counts and mark rare or unobserved features as uncalibrated. Before real-data calibration, explicitly label threshold-relative or fixed-strength probes as synthetic. Do not assume a JumpReLU threshold is a typical activation magnitude.

Measure survival of i, amplitude gain or loss, recruited features, first switch time, step distance, decoded drift, and endpoint behavior. Preserve directionality: i recruiting j need not imply j recruits i.

## Pair probes and divergence

Yes: test arbitrary feature pairs, vary their strengths, and compare their joint trajectory with both individual trajectories. The useful notion of divergence is departure from independent evolution, not simply that two different starting points remain different.

For each pair i, j and strengths a, b, run four matched trajectories:

```text
z_pair(0) = a * unit_i + b * unit_j
z_i(0)    = a * unit_i
z_j(0)    = b * unit_j
z_zero(0) = 0

Each trajectory uses: z(t+1) = F(z(t))

interaction(t) = z_pair(t) - z_i(t) - z_j(t) + z_zero(t)
```

The zero baseline corrects for affine offsets: this interaction vanishes for any affine F, including after repeated composition. Save the signed interaction vector and absolute and relative norms. A relative metric must include an explicit denominator and epsilon; always retain absolute magnitudes so near-zero denominators cannot manufacture dramatic effects.

Also compare decoded states with the same four-way contrast. Different codes can decode almost identically, so code-space divergence alone can exaggerate functional differences.

Record:

- Pair-only recruitment: features activated by the pair but not by either matched individual trajectory, with zero-baseline activity flagged separately.
- Suppression or enhancement relative to the baseline-corrected independent prediction.
- First significant interaction step, maximum interaction, and interaction over time.
- Which starting features survive, disappear or exchange dominance.
- Whether the pair approaches a different endpoint or stable support.
- Sensitivity to small amplitude perturbations and mixture ratio.

Use a small two-dimensional amplitude grid. Include matched decoded-input norm controls: a pair has more input signal than a single feature, and crossing thresholds due to that alone is not evidence of a special semantic relationship. Report cancellation for negatively aligned decoder directions.

Separately investigate sensitivity by comparing two nearly identical initial mixtures. Track how their distance evolves relative to the initial perturbation. Call this sensitivity or finite-time amplification, not chaos without stronger evidence.

## Pair selection and scale

A 16,384-feature dictionary has 134,209,536 unordered distinct pairs. Long trajectories for every pair and amplitude combination are not a sensible first pass.

1. Smoke test 100 features and a few hundred pairs.
2. Sweep all single features with short trajectories.
3. Select pairs from decoder-cosine neighbors, directed decoder-to-encoder coupling, single-probe recruitment, and random frequency/norm-matched controls.
4. Add co-activation and semantic candidates once real contexts are available.
5. Start with about 10,000 selected pairs, using a cheap one-step screen before long rollouts.
6. Expand promising cases to amplitude grids and contextual experiments.

Compute candidate similarities in blocks rather than requiring full dense pair matrices. Keep selection reasons and sampling probabilities. Because targeted selection enriches interactions, report random-control results separately; do not infer a dictionary-wide interaction rate from selected pairs.

## Contextual validation

### Next experiment: backgrounds selected by decoder similarity

User-selected direction: use **related-feature backgrounds selected by decoder-vector cosine similarity**. Run this synthetic experiment before collecting real activations. Random backgrounds are optional matched controls, not the primary background type. The user ran the initial five-step pilot; see `runs/background-pilot-001/RESULTS.md`.

Implementation is available in `scripts/backgrounds.py`; launch instructions are
in `README.md`. Defaults include the three single
targets and pairs 120+6712 / 120+2499, two sampled subsets per size, empty-background
controls, two target strengths (0.75/1.25), and background strength 0.25 relative
to target A's isolated boundary. Background members have equal absolute code
amplitudes; the norm-matched condition uses the nearest single neighbor's decoded
contribution norm as a family-specific reference across sizes. Pair pools are
the union of each target's top-128 positive neighbors, sorted by best source
cosine with feature-ID tie breaking. All five steps and full contrasts are saved.
Six tests pass, including affine contrast cancellation, a gate-induced interaction,
pool exclusions, reproducibility, and background matching. Of 784 trajectories,
302/111/24/9/2 were nonzero at steps 1/2/3/4/5. All norm-matched trajectories
reached zero by step 3. The only step-5 survivors were joint additions of
120+6712 and 120+2499 with equal-amplitude full-pool backgrounds at target
strength 1.25. This does not establish persistence at matched total norm or
behavior beyond step 5. Static pair baselines and first-step outcomes are
recorded for comparison; predictive validation remains future work.

For each target feature, normalize decoder vectors for neighbor selection only, and rank other features by signed cosine similarity. Prefer positive-cosine neighbors; do not use absolute cosine, which would conflate aligned and opposing directions. Keep original checkpoint weights and raw round-trip dynamics unchanged. Save neighbor IDs, ranks and cosine scores. Verify zero-norm vectors and exclude the target itself.

Start with targets 120, 11341 and 12692, whose isolated trajectories have already been inspected. Use the top 128 positive-cosine neighbors as an exploratory candidate pool, recording the actual count if fewer qualify. Start with subsets of 1, 4, 16 and 64 neighbors: include nearest-ranked subsets and several reproducibly sampled subsets at each size. Save pool membership, sampling seeds and exact subset membership. Include the full candidate pool as an additional condition, not the primary background. Subsets let us distinguish effects of particular companions from effects that persist across a related group. These sizes are exploratory choices, not estimates of natural co-activation. Positive decoder similarity means similar writing directions, not proven semantic compatibility or natural joint activation.

For a single target A, compare background alone with the identical background + A. For a target pair A and B, form a deterministic ranked union of their decoder neighbors, excluding A and B, and select subsets from that pool. Reuse exactly the same background and amplitudes in all four conditions:

```text
background
background + A
background + B
background + A + B

interaction(t) = joint(t) - A_only(t) - B_only(t) + background_only(t)
```

Use modest synthetic amplitude sweeps around each feature's self-activation boundary, clearly labeled as uncalibrated on real data. Similar decoder directions reinforce one another, so record both code norm and decoded contribution norm (`decode(z) - decode(0)`). Compare equal per-feature amplitude backgrounds with a separate condition matching total decoded background contribution norm across background sizes; do not renormalize any state during rollout. Keep the background identical within a four-condition comparison, and report the total starting norms of every condition.

Primary outcomes: first-step recruitment/suppression, interaction persistence over steps 2–5, and sensitivity across related backgrounds and amplitudes. Save every state for later plots. Longer trajectories remain secondary because isolated probes often become extremely dense or collapse. Compare findings against decoder cosine and directed decoder-to-encoder coupling. Follow up feature 120's early recruits 6712 and 2499 with matched ablations to test whether they contribute to growth.

Safety-oriented targets are a possible later subset. No safety/refusal feature IDs have been verified for our exact layer-12 base-model checkpoint. Refusal findings in Gemma-2-2B-IT cannot be transferred by feature ID. Distinguish content-related features from demonstrated safety mechanisms.

### Later: real-activation validation

Synthetic one-feature and two-feature codes may be far from naturally occurring codes. Validate findings in actual encoded activations.

For a context, retain all background features and construct four versions: background only, background plus i, background plus j, and background plus both. Remove the original i and j entries before constructing this matched set. Apply the same four-trajectory contrast with the background trajectory replacing zero.

Test multiple contexts and realistic strengths. Later, patch matched decoded activations into the LLM and compare downstream output changes. Control reconstruction error and intervention magnitude. SAE consistency does not by itself demonstrate model-semantic or causal relevance.

## Save trajectories for later visualizations

Save numerical trajectories now; build visualizations later. The initial implementation uses lossless SciPy CSR NPZ state files, JSON metadata and CSV step metrics rather than the proposed Parquet layout below.

Suggested artifact layout:

```text
runs/<run_id>/
  manifest.json
  probes.parquet
  feature_stats.parquet
  states/shard_*.npz
  steps.parquet
  interactions.parquet
  endpoints.parquet
```

- `manifest.json`: schema version, exact checkpoint revision and file hash, library versions, model/hook, SAE architecture and preprocessing, random seeds, precision/device, convergence rules, amplitude rules, data provenance and sampling method.
- `probes.parquet`: probe ID, family, feature IDs, amplitudes, background/context reference, paired baseline IDs, selection reason and initialization specification.
- `states`: every step including step zero, stored as exact nonzero feature indices and float32 values with offsets and probe/step IDs. Do not silently truncate to top-k. Fall back to dense chunked storage when sparse storage becomes inefficient.
- `steps.parquet`: code and decoded norms, active counts, step residual, distance from initial state, original-feature strengths, support changes and validity flags.
- `interactions.parquet`: links to all four trajectories and scalar interaction metrics; signed vectors can be reconstructed from the saved states or cached for selected cases.
- `endpoints.parquet`: final step, stopping reason, convergence diagnostics and later cluster assignments. Preserve raw trajectories even when endpoints are clustered.
- `feature_stats.parquet`: decoder norms, thresholds, calibration quantiles, activation frequencies, counts, and source links for any semantic annotations.

Pinned decoder weights allow decoded states to be reconstructed later. Optionally cache decoded trajectories for selected cases. Do not store only 2D projections: that would prevent changing the visualization or distance metric later.

Pilot storage estimate: 10,000 trajectories with 101 states and 100 nonzeros per state need about 0.81 GB for int32 indices and float32 values, before metadata or compression. Four independent trajectories per pair could multiply this by four; reuse identical single and zero baselines. Actual activity may grow, so measure pilot storage before scaling.

## Numerical controls and stopping

- Begin with float32 inference and a horizon of 100 iterations; inspect longer rollouts selectively.
- Preserve the checkpoint's exact encode/decode behavior, including biases and preprocessing. Do not manually reapply normalization that the loader already handles.
- Check relative and absolute residuals over several consecutive steps, plus decoded-space residuals. A stable support is not necessarily a fixed point.
- Record collapse, growth, non-finite values, detected short cycles, convergence and maximum-step termination separately.
- Recheck threshold-sensitive cases at higher precision where possible.
- Use identity/affine toy maps to verify the four-way interaction calculation and a simple shrinkage map as a collapse baseline.
- Do not normalize codes after every step in the primary experiment. Such normalization creates a different dynamical system; if explored, label and store it separately.
- Keep magnitude and direction metrics separate. Cosine is undefined or unreliable near zero; mark such cases rather than assigning a misleading value.

## Main interpretation risks

Sparsity does not guarantee contraction. A sparse active transformation can amplify its surviving directions. JumpReLU threshold boundaries can produce abrupt changes; TopK introduces feature competition. Classical L1 shrinkage can cause trivial collapse. See [gated SAEs](https://arxiv.org/abs/2404.16014) and [JumpReLU SAEs](https://arxiv.org/abs/2407.14435).

A faithful SAE may preserve a continuum of feature strengths and therefore have approximately fixed regions rather than isolated attractors. Long rollouts can magnify tiny reconstruction biases. For this reason, prioritize short-trajectory interactions and their incremental value over static baselines; treat endpoint interpretation as a separate empirical question.

## Later visualizations

- Feature interaction graph with directed edges, amplitude selectors and static-similarity overlays.
- Four-trajectory comparison for a selected pair, with recruited/suppressed features listed at each step.
- Pair amplitude heatmaps showing interaction onset and endpoint changes.
- Code and decoded-state trajectories projected using a shared fitted projection, with projection distortion made inspectable.
- Context browser linking trajectory changes to real activating text and eventual intervention effects.

Success criterion: identify reproducible, context-validated interactions that static decoder cosine and activation co-occurrence do not already explain. Save null results and failures as well as visually interesting trajectories.

## Implementation and initial runs

Implementation: `scripts/trajectories.py`; tests: `tests/test_trajectories.py`.
The downloaded checkpoint is stored under `checkpoints/` and pinned to Hugging Face revision `fd571b47c1c64851e9b1989792367b9babb4af63`; each run also records its SHA-256. NumPy/SciPy run on CPU; the LLM is not downloaded.

Released Gemma Scope inference weights already incorporate training-time centering and scaling. Use the raw released equations from Appendix A of the Gemma Scope paper; do not subtract the decoder bias before encoding.

The runner saves all initial and subsequent nonzero float32 values, without top-k truncation. Each step file has a companion probe-ID array, because completed trajectories stop independently. JSON endpoints include termination reasons and fresh fixed-point residuals. Perturbation trajectories are saved separately. Binary artifacts and weights are locally retained but ignored by Git.

Fixed-point candidates require five consecutive steps with distance at most `1e-5 + 1e-5 * previous_code_norm`. Candidate clustering uses full-code Euclidean distance, with `1e-4` absolute and relative tolerances. These tolerances must be checked for sensitivity before claiming distinct attractors. Up to 16 representative candidates receive small active-amplitude and new-coordinate perturbations; empirical returns do not constitute a mathematical proof of attraction. Zero endpoints have only new-coordinate perturbations.

Initial run `runs/pilot-002`: 32 randomly selected features, seed 42, three strengths (1.25, 2 and 4 times the feature threshold), plus zero baseline. All 97 trajectories converge to zero, and three small nonnegative perturbations of zero return. This establishes collapse for these synthetic seeds, not a dictionary-wide or real-data conclusion. `pilot-001` is an incomplete run whose final JSON serialization failed; use `pilot-002` as the successful replacement.

Follow-up `runs/pilot-003`: 128 randomly selected features with strengths 1.25, 4 and 16 times their isolated self-activation boundary. Compute that boundary as `(threshold - round_trip_bias) / diagonal_encoder_decoder_gain`; all gains in this checkpoint are positive. This accounts for biases that threshold-only initialization misses. These strengths remain synthetic and are not empirical activation quantiles.

Measured outcomes: 251 of 384 feature probes reach zero, while 133 hit the growth limit (code norm greater than one million times the larger of initial norm and one). The additional zero baseline stays zero. No nonzero fixed-point candidates or period-two candidates were found. At multipliers 1.25, 4 and 16, collapse/growth counts are respectively 128/0, 117/11 and 6/122. Three small nonnegative perturbations of zero return. This only tests a few local perturbations and does not estimate basin volume. Full report: `runs/pilot-003/RESULTS.md`.

Next priority: test synthetic decoder-similar background subsets and matched single/pair additions, then validate promising findings with real-context seeds. Do not assume that longer iteration yields useful semantic prototypes. The collapse/growth pattern makes amplitude and context controls essential.

Trajectory inspection completed: 66/133 growing paths initially shrink; their last-step direction cosine has median 0.9925 despite a median 16.07-fold norm increase. Features 7488 and 1468 dominate 30 and 25 growing endpoints respectively, but overall endpoint directions are heterogeneous. Growing states become dense (median 4,286 active features). Twenty collapsing paths nevertheless recruit additional features. Feature 120 provides a concrete delayed-growth example; feature 11341 provides a gradual-collapse example. Details and limitations are in `runs/pilot-003/RESULTS.md`, with reusable analysis in `scripts/analyze_trajectories.py`. Next tests should isolate whether early recruited features actually drive these outcomes; repeated dominance alone does not establish semantic structure.

Scope: initial feature trajectories and empirical fixed-point checks only. No pairwise, contextual or causal-LLM results yet. Do not turn collapse into a semantic claim.

## What trajectories must add over existing structure measurements

Decoder cosine measures writing-direction similarity; co-activation measures usage association; hierarchical analyses examine coverage and reconstruction; attribution graphs examine downstream computation. Our primary target is whether an unchanged SAE preserves or rewrites a feature combination.

The essential additional baseline is directed decoder-to-encoder coupling. With affine decoding and a linear encoder followed by a gate, a round trip is `gate(M * z + bias)`, where `M[j,i]` is encoder j dotted with decoder i. Within a fixed gate region the mapping is affine. Multi-step pair effects must therefore be interpreted in terms of changing gates, their propagation, and context, rather than advertised as unexplained nonlinear semantics.

Compare decoder cosine, co-activation, directed coupling, one-step interaction and full trajectories on held-out predictions of representation ambiguity or unexpected downstream intervention effects. Long trajectories are useful only if they add information beyond the cheaper baselines. PolySAE already explicitly models feature interactions through learned polynomial decoding; our distinction is diagnostic analysis of an existing unchanged SAE, not being the first interaction study.

## Expected interpretability value and validation

The synthetic experiment diagnoses whether the SAE preserves, replaces or suppresses features in combinations. It does not directly establish feature meanings or language-model computation. In particular, A redistributing into B and C may indicate alternative decompositions, decoder redundancy or threshold effects; it does not by itself establish a shared concept.

Measure replacement, reinforcement/suppression, pair interaction beyond the separate additions, and sensitivity across neighbor subsets and amplitudes. A reproducible group could prioritize joint inspection of activating examples. Context-sensitive responses could flag feature edits whose representations need closer validation. Delayed recruitment or switching between groups is useful only if later steps improve on first-step measurements; growing dense trajectories alone are not an interpretability result.

The concrete next-stage test is whether trajectory-based groups or metrics predict shared activating examples, representation ambiguity, or downstream intervention effects better than decoder cosine, directed decoder-to-encoder coupling and one-step responses. Evaluate incremental prediction on held-out examples or interventions, with co-activation baselines once real activations are available. A null result is informative: if static or one-step baselines explain the effects, longer trajectories have not added demonstrated value.

Current status: these are hypotheses and planned tests. Synthetic backgrounds are controlled probes, not established natural feature combinations. Real activations and model interventions are needed before claiming practical interpretability gains.

## Complete bibliography of papers referenced in this conversation

This list includes papers linked or named in our written discussion and this plan, including adjacent work. It does not imply every paper was read in full or every technique is part of the implementation.

### Attractors and autoencoder theory

1. Fumero, Moschella, Rodolà and Locatello. **Navigating the Latent Space Dynamics of Neural Models**. ICLR 2026. [Paper](https://arxiv.org/abs/2505.22785). Main motivation; SAE extension explicitly proposed as future work.
2. Radhakrishnan, Belkin and Uhler. **Overparameterized Neural Networks Implement Associative Memory**. 2020. [Paper](https://arxiv.org/abs/1909.12362). Training examples as attractors.
3. Alain and Bengio. **What Regularized Auto-Encoders Learn from the Data-Generating Distribution**. JMLR 2014. [Paper](https://arxiv.org/abs/1211.4246). Reconstruction residuals and density scores under specified assumptions.

### Rodolà's related representation and autoencoder work

4. **LIMP: Learning Latent Shape Representations with Metric Preservation Priors**. ECCV 2020. [Paper](https://arxiv.org/abs/2003.12283). Shape geometry and latent interpolation.
5. **Relative Representations Enable Zero-Shot Latent Space Communication**. ICLR 2023. [Paper](https://arxiv.org/abs/2209.15430). Anchor-relative representations and model stitching.
6. **Latent Functional Maps: A Spectral Framework for Representation Alignment**. NeurIPS 2024. [Paper](https://arxiv.org/abs/2406.14183). Comparing and aligning latent spaces.
7. **How Neural Losses Shape VAE Latents**. 2026 preprint. [Paper](https://arxiv.org/abs/2606.00635). Training losses, information rate and posterior geometry.
8. **SAGE: Semantic Audio Generative Encoder**. 2026 preprint. [Paper](https://arxiv.org/abs/2609.32755). Semantically structured audio autoencoding.

### SAE architectures and released checkpoints

9. **Scaling and Evaluating Sparse Autoencoders**. 2024. [Paper](https://arxiv.org/abs/2406.04093). TopK training and SAE evaluation; cited by the attractor paper's SAE discussion.
10. **Improving Dictionary Learning with Gated Sparse Autoencoders**. 2024. [Paper](https://arxiv.org/abs/2404.16014). Separating feature selection and magnitude; shrinkage motivation.
11. **Jumping Ahead: Improving Reconstruction Fidelity with JumpReLU Sparse Autoencoders**. 2024. [Paper](https://arxiv.org/abs/2407.14435). Architecture used by the selected checkpoint.
12. **Gemma Scope: Open Sparse Autoencoders Everywhere All At Once on Gemma 2**. 2024. [Paper](https://arxiv.org/abs/2408.05147). Checkpoint suite and inference reparameterization; [technical PDF](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf).

### Structure, matching, hierarchies and interactions

13. **The Geometry of Concepts: Sparse Autoencoder Feature Structure**. 2024, revised 2025. [Paper](https://arxiv.org/abs/2410.19750). Decoder geometry and functional co-occurrence communities.
14. **Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders**. 2024. [Paper](https://arxiv.org/abs/2410.06981); [author code](https://github.com/wlg1/univ_feat_geom). Activation-based matching and subspace comparisons.
15. **Semantic Optimal Transport for Sparse Autoencoder Feature Matching and Circuit Compression**. 2026 preprint. [Paper](https://arxiv.org/abs/2605.28567). Distribution-based feature matching.
16. **A Is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders**. NeurIPS 2025. [Paper](https://arxiv.org/abs/2409.14507). Hierarchical features and unreliable feature activation.
17. **Tree SAE: Learning Hierarchical Feature Structures in Sparse Autoencoders**. 2026 preprint. [Paper](https://arxiv.org/abs/2605.07922). Hierarchy using activation and reconstruction constraints.
18. **Circuit Tracing: Revealing Computational Graphs in Language Models**. 2025. [Research article](https://transformer-circuits.pub/2025/attribution-graphs/methods.html). Attribution graphs and computational connections; distinct from SAE round trips.
19. **PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding**. ICML 2026. [Proceedings](https://proceedings.mlr.press/v306/koromilas26a.html); [preprint](https://arxiv.org/abs/2602.01322). Explicit pairwise and higher-order decoder interactions; important related work for novelty claims.
20. **Beyond I'm Sorry, I Can't: Dissecting Large Language Model Refusal**. 2025. [Paper](https://arxiv.org/abs/2509.09708). Refusal-related SAE features and interactions in instruction-tuned models, including Gemma-2-2B-IT; not verified feature IDs for our checkpoint.
21. **SAEs Are Highly Dataset Dependent: A Case Study on the Refusal Direction**. [Research post](https://www.alignmentforum.org/posts/rtp6n7Z23uJpEH7od/saes-are-highly-dataset-dependent-a-case-study-on-the). Referenced in the conversation; a research post rather than a conference paper. Discusses refusal-direction representation in GemmaScope SAEs.

### Software and checkpoint references (not papers)

- [SAELens](https://decoderesearch.github.io/SAELens/) and [supported checkpoints](https://decoderesearch.github.io/SAELens/sae_table/).
- [Selected Gemma Scope checkpoint](https://huggingface.co/google/gemma-scope-2b-pt-res/tree/main/layer_12/width_16k/average_l0_82).
- [GPT-2 Small fallback SAEs](https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted).
