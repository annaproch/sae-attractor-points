# Comparing SAE trajectories for honest and deceptive responses

## Question

Can short SAE encode–decode trajectories reveal differences between honest and deceptive responses that are harder to detect from the initial activations alone?

This is an exploratory hypothesis. 

## Starting dataset: Apollo

Start with the scenarios from [Apollo Research's Detecting Strategic Deception Using Linear Probes (2025)](https://github.com/ApolloResearch/deception-detection). Its roleplaying and insider-trading settings provide a relevant starting point and an existing linear-probe baseline. Different scenario families also allow a test of generalization.

An alternative is [MASK (2025)](https://arxiv.org/abs/2503.03750), which compares a model's elicited beliefs with its statements under pressure, helping distinguish dishonesty from factual mistakes.

## Hypothesis

Honest and deceptive responses may initially activate overlapping SAE features but differ in which feature groups persist, disappear, or recruit other features over a few round trips. These dynamics might help detect deception across unfamiliar scenarios or nominate feature groups for causal investigation. let's analyse these trajectories.

## Original proposed pilot (before model selection)

1. Check whether our current base Gemma-2-2b produces enough coherent honest and deceptive responses in a small set of Apollo scenarios before collecting a larger dataset.
2. Aim for 100–200 matched examples, keeping facts and scenarios comparable while varying the conditions that elicit honesty or deception. Label the actual responses; an instruction to lie is not evidence that the model lied. Distinguish deceptive responses from ordinary factual errors.
3. Extract natural forward-pass activations at predefined, consistent positions, such as the last prompt token and selected response positions. Analyze these positions separately. Response-token measurements detect behavior already underway; they should not be presented as advance prediction.
    4. Encode with our current layer-12 SAE and save 5–10 iterations of encode(decode(z)). Keep numerical stopping checks and report truncated trajectories.
5. Measure feature persistence, feature recruitment, gate changes, relative movement, and norm growth. Treat final convergence as secondary.
6. Compare predictive performance against probes on the original residual activations and initial SAE codes, plus simple predictors such as activation norm, initial sparsity, and one-step reconstruction error.
7. Hold out entire scenario families, keeping matched examples and variants together in the same split. Fit all feature selection and predictors on training data only.

The reason to focus on short trajectories is empirical: in our existing natural-input experiment, 83 of 100 inputs reached the growth limit, while the remaining 17 eventually approached a small set of sparse fixed points. Long trajectories may discard the distinctions we want to study.

## What would count as useful?

- **Detection:** trajectory measurements improve deception detection on unseen scenarios over the initial-activation and one-step baselines.
- **Interpretability:** recurring feature groups have understandable relationships to the behavior and survive checks for topic and wording confounds.
- **Causal evidence:** interventions on nominated groups predictably change behavior in the actual transformer, with appropriate controls for unrelated effects and SAE reconstruction error.

Trajectory separation alone establishes neither a deception mechanism nor a useful steering direction. Stop or revise the approach if its apparent gains disappear under the baselines and held-out evaluation.

## Model choice

The original local SAE targets pretrained Gemma 2 2B. For the implemented pilot,
we instead generated responses with Gemma 3 4B IT and used its matching Gemma
Scope 2 instruction-model SAE. We did not feed another model's reference
completions into Gemma or reuse the incompatible Gemma 2 SAE.

## Status

Generation, API judging, last-prompt-token activation extraction, and SAE
trajectories are complete for the first 30-scenario pilot. Behavioral comparisons
and predictive baselines remain outstanding. See [DECEPTION_PILOT.md](DECEPTION_PILOT.md)
for runnable commands and [the results record](runs/deception-trajectories-001/RESULTS.md)
for the completed steps and findings.

The judge labeled 19 responses deceptive, 3 honest, and 8 ambiguous. Excluding
ambiguous examples left 22 activations. All 22 SAE trajectories hit the norm-growth
limit after 11–14 iterations; none reached a fixed-point or period-2 candidate.
All 298 code states, including initial and terminal states, were saved and verified.
This establishes a working pipeline, not deception detection or a class difference.
