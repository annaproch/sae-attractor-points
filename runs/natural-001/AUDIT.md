# Convergence and implementation audit

The absence of convergence for most probes prompted these checks. No model
weights or experiment code were changed.

- Installed Transformers 4.57.6 Gemma2Model records the hidden state before
  each block; hidden_states[13] is therefore the output of zero-indexed block
  12. Gemma Scope residual SAEs target the post-MLP residual stream.
- Gemma Scope Appendix A states that released weights already absorb the
  training normalization and decoder-bias subtraction. Our raw encode/decode
  equations match its equations 10–11.
- An independent PyTorch evaluation of all 100 stored input activations and
  first round trips agreed with NumPy: zero active-support mismatches, relative
  L2 differences 1.59e-7 (encoding) and 1.61e-7 (first round trip).
- Replaying original probe 0 through its already-recorded 15 steps in float64
  gave endpoint norm 584,115,275.5, versus 584,116,864 in float32. Growth factor
  was about 9.23 million. This example's growth is not a float32-only artifact.

## Direct check of the contraction assumption

At each starting code, the active output support S is taken from its first
round trip. With row-vector conventions, the full local Jacobian is the
decoder-to-encoder product with inactive output columns zeroed. Its nonzero
eigenvalues are those of W_dec[S] @ W_enc[:, S]. The spectral norm of this
principal submatrix lower-bounds that of the full Jacobian. Calculations used
float64 weights and the saved float32 gate supports.

Across all 100 initial codes:

- Spectral-norm lower bound: min 1.35203, median 1.57340, max 2.67613.
- Spectral radius: min 1.30951, median 1.51540, max 2.57496.

Thus every tested starting point fails the strict ambient local contraction
condition used in the paper. This does not prove every trajectory grows:
the Jacobian changes along the path, and 17 paths eventually reach stable
regions. It also does not characterize contraction restricted to a data manifold.

Theorem A.6 assumes contraction in a suitable region, not merely a natural
starting sample. Its smoothness assumptions do not apply across JumpReLU
thresholds, and sparsity alone does not bound feedback gain below one.
Single-step reconstruction training does not ensure repeated reconstructions
stay near the natural activation distribution.

Sources:
- https://arxiv.org/html/2505.22785v4#A1.SS1
- https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf (Appendix A)

These checks find no implementation error explaining the growth, but they are
not a complete external-library or end-to-end activation-extraction validation.
