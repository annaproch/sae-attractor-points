# Steering pilot: early trajectory analysis

All 45 generations and trajectories completed. All trajectories stopped at the growth limit; none converged. Analysis uses steps 0–5, which exist for every probe, avoiding comparisons biased by early stopping. Step 0 is the SAE encoding of the steered last prompt token; subsequent steps are encode(decode(z)), not model generation steps. Original SAE thresholds were used.

## Behavioral review

These are provisional qualitative observations from saved responses, not independent judge labels. The original labels.csv is unchanged. Obvious collapse means repetitive loops, symbol/gibberish output, or disintegrating code; factual errors and reaching the 96-token cap alone do not qualify.

- 23 obvious collapse cases: all 20 contrastive/random 4× and 8× cases; random 2× on bird-family, seasons, and coding prompts (probe IDs 13, 31, 40).
- 20 cases without clear collapse: all baselines, all 1×, all contrastive 2×. This does not mean correct or successful; contrastive 2× includes factual confusion.
- Two uncertain cases: random 2× on the first and third bird prompts (IDs 4, 22); strange, confused descriptions without equally clear repetition. Excluded from the exploratory discrimination calculation.
- The candidate target-effect example is probe 12 (bird-family prompt, contrastive 2×): it explicitly calls caracaras owls, but also offers contradictory classification. It is not established clean steering. Other 2× target responses do not clearly achieve the target within the saved text.

## Early dynamics

Medians over five prompts. L0 is the number of nonzero features. Growth is code norm at step 5 divided by its own initial code norm. Jaccard is intersection/union of active feature sets against the same prompt's baseline at the same step.

| Condition | L0 step 0 → 1 → 3 → 5 | Growth at step 5 | Baseline Jaccard step 0 → 5 |
|---|---|---:|---|
| Baseline | 47 → 43 → 43 → 57 | 1.39× | 1 → 1 |
| Contrastive 1× | 52 → 46 → 48 → 73 | 1.56× | .686 → .453 |
| Random 1× | 57 → 49 → 49 → 73 | 1.56× | .677 → .491 |
| Contrastive 2× | 72 → 58 → 49 → 54 | 1.44× | .452 → .194 |
| Random 2× | 107 → 79 → 82 → 325 | 3.27× | .333 → .127 |
| Contrastive 4× | 223 → 147 → 319 → 2689 | 15.73× | .135 → .010 |
| Random 4× | 578 → 384 → 3937 → 8334 | 3969.89× | .058 → .002 |
| Contrastive 8× | 937 → 1131 → 6407 → 6115 | 52372× | .027 → .003 |
| Random 8× | 2429 → 1270 → 9666 → 7802 | 100487× | .013 → .002 |

The first iteration often prunes features, even in cases that later explode. Lower feature count after one step is therefore not evidence of recovery. Contrastive 2× can remain sparse while moving away from baseline feature identities. This is compatible with a changed representation, but does not by itself establish useful steering.

Random and contrastive 1× are almost indistinguishable on these aggregate dynamics. At equal larger strengths, the random direction is usually more disruptive. Thus direction matters beyond intervention norm, although there is only one random direction in this pilot.

## Concrete trajectories and identities

Same bird-family prompt:

- Baseline (probe 9): L0 43 → 38 → 41 → 49 at steps 0,1,3,5. Feature 34 starts strongest (1505), while 280 grows from 554 to 1356 and becomes strongest at step 5; 456, 266, and 377 also rise. Norm growth at step 5: 1.27×.
- Candidate owl response, contrastive 2× (probe 12): L0 85 → 80 → 124 → 571. Feature 712 rises 590 → 809 → 1452 by step 3; feature 34 reaches 6919 at step 5, with 280,319,534,4074 also strong. Norm growth: 6.11×.
- Repetitive contrastive 4× (probe 14): L0 273 → 278 → 1590 → 7754. Initially led by 34,534,712; by step 5 the leaders are 25,16,624,9,285, with feature 25 at 298680. Norm growth: 602×.
- Repetitive random 2× (probe 13): L0 133 → 79 → 82 → 325. Norm growth: 3.27×, LESS than the candidate owl response. This directly defeats a universal ordering of behavioral quality by growth speed.

Coding prompt:

- Baseline (probe 36): L0 47 → 43 → 62 → 167; norm growth 2.98×.
- Contrastive 2× (probe 39): L0 68 → 58 → 44 → 48; norm growth 1.44×. Features 280,107,34 lead at step 5. This steered trajectory is more stable over this window than baseline, despite starting denser. Both code outputs are truncated, so functional correctness is untested.
- Repetitive contrastive 4× (probe 41): L0 223 → 140 → 163 → 466; norm growth 3.27×. Feature 48 reaches 5059 and dominates, followed by 225 and 50. Its growth is only slightly above the coherent-looking baseline despite obvious repetition in the response.

Across prompts at step 5, feature 883 is in the top ten for every baseline and every contrastive 1× example. Feature 48 appears in the top ten for four of five contrastive 4× cases. At 8× both directions share features 381,9,113 in the top ten for every prompt. These are recurring numerical amplification patterns; no semantic interpretation of these IDs has been established, and they are not unique markers of degeneration.

## Do trajectories add predictive value?

Not demonstrated. On the 43 provisionally clear cases, retrospective AUROC is:

| Measurement (higher score predicts collapse) | AUROC |
|---|---:|
| Initial reconstruction error | 1.000 |
| Initial L0 | 1.000 |
| First-step support turnover, 1 − Jaccard(previous,current) | 1.000 |
| Norm growth at step 1 | .909 |
| Norm growth at step 3 | .970 |
| Norm growth at step 5 | .978 |
| Negative baseline support Jaccard at step 0 | .996 |
| Negative baseline support Jaccard at step 5 | .974 |
| Negative steps to growth limit | .961 |

Initial L0 ranges: 42–86 without clear collapse, 90–2849 with clear collapse; uncertain cases are 106–107. Initial reconstruction error: 3.37–6.11% without clear collapse, 7.89–93.08% with clear collapse; uncertain cases are 7.34–7.58%. The separation is retrospective and must not be presented as a validated threshold. No held-out evaluation, classifier fitting, or independent behavioral grading was performed. Samples share prompts and directions, most failures are at very large strengths, and judgments are provisional. AUROC here describes this sample only. It does not establish generalization or incremental information conditional on initial metrics.

## Implication

We have useful evidence about obvious degradation and several informative counterexamples to simple trajectory hypotheses. Divergence itself is uninformative because every baseline also diverges. Early feature turnover and growth characterize the intervention, but have not outperformed cheaper initial measurements. Studying reliable successful steering requires more clean target-effect cases; this pilot mostly contrasts weak effects with severe collapse. A future validation should use independent response labels, more prompts/directions, and held-out prompts, comparing trajectory features against initial error/L0 and intervention strength.

Artifacts: early_metrics.csv contains 270 aligned rows (45 probes × six steps), including feature-support overlap, cosine, norm growth, turnover, provisional collapse categories and initial error. statistics.txt contains aggregate statistics and concrete top-feature amplitudes. analyze.py reproduces both from saved states without loading the language model or SAE.
