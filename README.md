# spectral-analysis

# Spectral Interpretability Pilot: Attention Graph Topology Across Model Behaviors

Pilot study testing whether a transformer's internal attention structure differs
systematically depending on the *actual outcome* of its response, across three
behaviors: reasoning correctness, hallucination, and sycophancy.

## Overview

This reproduces the baseline spectral graph analysis method from **Valentin
Noël, "Geometry of Reason: Spectral Signatures of Valid Mathematical
Reasoning" (arXiv:2601.00791, 2026)**, applied to a small instruction-tuned
model, to check whether the method's core diagnostics show layer-wise trends
across three separate behavior comparisons — not just mathematical reasoning
validity as in the original paper.

**This is a pilot, not the full research project.** It only tests for
aggregate, layer-wise trends. It does not do head-level analysis, does not
build a classifier, and does not make causal claims. It is a prerequisite
step toward a larger planned research effort.

## Research question

For a fixed behavior (reasoning, hallucination, or sycophancy), do
layer-wise spectral trajectories of the model's attention graph differ
depending on the model's actual response outcome (not the prompt type)?

## Model

`Qwen/Qwen2.5-0.5B-Instruct` — smallest instruction-tuned model that also
appears directly in the original paper's own evaluation set (Table 1),
enabling a rough sanity check against published numbers. Global (non-sliding-
window) attention, 24 layers, 14 heads.

## Method

Faithful reproduction of the paper's default baseline pipeline:

1. **Attention extraction**: `attn_implementation="eager"` (required —
   SDPA/FlashAttention backends silently drop attention weights). Full
   sequence (prompt + generated response) analyzed via a second, teacher-
   forced forward pass over the model's own exact generated token IDs.
2. **Per-head symmetrization**: `W^(l,h) = 0.5 * (A^(l,h) + A^(l,h)ᵀ)`
3. **Head aggregation**: attention-mass-weighted average across heads into
   one graph per layer (numerically equivalent to a uniform average for this
   model's standard dense softmax attention, confirmed empirically).
4. **Graph Laplacian**: combinatorial Laplacian `L = D - W` (paper's default;
   not the normalized variant).
5. **Spectral diagnostics**, computed per layer from the Laplacian
   eigendecomposition and hidden-state graph signals:
   - **Dirichlet energy** — total variation of hidden states w.r.t. graph structure
   - **HFER** (High-Frequency Energy Ratio) — proportion of signal energy in
     high-frequency spectral components (cutoff = median eigenvalue index)
   - **Spectral entropy** — how spread out signal energy is across spectral modes
   - **Fiedler value** — second-smallest Laplacian eigenvalue (algebraic connectivity)
   - **Smoothness** — normalized inverse of Dirichlet energy

## Pipeline steps (Kaggle notebook, 9 checkpoints)

1. **Environment** — GPU/internet check, package install, working directory setup
2. **Model** — load Qwen2.5-0.5B-Instruct, verify chat template and attention support
3. **Full-sequence replay** — teacher-forced second forward pass, validate attention/hidden-state shapes
4. **Graph construction** — symmetrize, aggregate, build Laplacian, validate invariants
5. **Spectral diagnostics** — compute all five metrics per layer, validate on one example
6. **Dataset generation** — pull prompts (GSM8K, TruthfulQA, Anthropic model-written-evals
   sycophancy set), generate model responses
7. **Labeling & outcome audit** — assign ground-truth response outcomes, confirm all
   six outcome groups are represented
8. **Metric computation** — run the full pipeline over every accepted, labeled example
9. **Plots & confound diagnostics** — trajectory comparisons per behavior, length/attention-sink checks

## Datasets & labeling methods

| Behavior | Source | Labeling method |
|---|---|---|
| Reasoning | GSM8K (Cobbe et al. 2021) | Automatic — exact numeric match against ground truth |
| Hallucination | TruthfulQA (Lin et al. 2021) | **Manual review** — a lexical-overlap heuristic was tried and rejected (see below) |
| Sycophancy | Anthropic model-written-evals (Perez et al. 2022) | Automatic — extracted model's chosen letter (A/B), compared to dataset's known sycophantic answer |

**Labeling notes worth knowing:**
- Hallucination labeling was attempted automatically via lexical word-overlap
  against TruthfulQA's reference correct/incorrect answer lists. This was
  rejected after testing on real data: it confidently mislabeled a genuinely
  fabricated, novel hallucination as "non-hallucinated," because the model
  invented a claim that didn't lexically resemble any answer the dataset's
  authors had anticipated. Closed-set lexical comparison structurally cannot
  catch novel fabrications — switched to full manual labeling instead.
- Sycophancy's source data had to be **randomly sampled**, not taken in file
  order — an early version accidentally sampled a contiguous block of the
  source file where the sycophantic answer was always "(A)," making the
  sycophantic/non-sycophantic split indistinguishable from simple position
  bias. Caught and fixed before the final run.

## Final sample sizes

| Behavior | Group A | n | Group B | n |
|---|---|---|---|---|
| Reasoning | correct | 13 | incorrect | 37 |
| Hallucination | non_hallucinated | 17 | hallucinated | 33 |
| Sycophancy | non_sycophantic | 20 | sycophantic | 30 |

150 total examples × 24 layers = 3,600 metric rows.

**These are pilot-scale, not publication-scale, samples.** Effect sizes at
this n are noisy and should be treated as candidates for follow-up, not
confirmed findings.

## Findings

For each behavior, effect size (Cohen's d) and Mann-Whitney significance were
computed at every layer × metric combination (120 comparisons per behavior).
Top effects were then checked against a known confound: response length
(the combinatorial Laplacian is not length-invariant — Dirichlet energy in
particular scales with sequence length).

### Reasoning (correct vs. incorrect) — real signal, but overstated by raw numbers

- Largest raw effect: `dirichlet_energy`, layer 16, **d = -1.35**, p = 0.0006
- Incorrect responses ran ~36 tokens longer on average (341 vs. 305 tokens);
  `dirichlet_energy` correlates at **r = 0.91** with sequence length in this
  group — most of the raw effect is explainable by length alone.
- After controlling for length: **d drops to ≈ -0.60**. Real, moderate
  signal remains, but roughly half the raw number.
- **Takeaway**: any follow-up on reasoning should control for response
  length as a covariate, not report raw effect sizes.

### Hallucination (non-hallucinated vs. hallucinated) — most trustworthy result

- Largest effect: **Fiedler value** (algebraic connectivity), layer 11,
  **d = 0.90**, p = 0.007. Non-hallucinated responses show more connected
  attention graphs.
- Length-controlling **strengthened** this effect (d = 0.90 → 0.97) — length
  was, if anything, suppressing the signal, not inflating it.
- Raw between-group difference at this layer is small in absolute terms
  (~2.3%), but tight within-group consistency makes it a real, moderate-to-
  large effect by Cohen's d.

### Sycophancy (non-sycophantic vs. sycophantic) — signal holds, concentrated late

- Largest effect: **spectral entropy**, layer 23 (near-final layer),
  **d = 0.91**, p = 0.012. Sycophantic responses show more concentrated
  (lower-entropy) spectral energy.
- Also strengthened under length-controlling (d = 0.91 → 1.09).
- Notably later in the network than hallucination's layer-11 effect — worth
  investigating whether different behaviors have spectrally distinct depths
  in the full study.

### Important caveats

- **No multiple-comparison correction applied.** 120 tests per behavior
  (24 layers × 5 metrics) at uncorrected p < 0.05 will surface several
  "significant" hits by chance alone. None of the above are confirmed
  findings — they are pilot-stage candidates for follow-up.
- **Trajectory plots visually understate these effects.** Between-group
  differences at any single layer are small (2–14%) relative to the huge
  layer-to-layer swings that dominate the full 24-layer plot's visual scale.
  A zoomed single-layer comparison (boxplot) makes the effect visible where
  the full trajectory plot does not.
- Sample sizes remain small; nothing here is a stable, replicated result.

## Recommended next steps

1. Apply Benjamini-Hochberg correction across all 360 tests (3 behaviors ×
   120 comparisons each) to identify which effects survive a stricter bar.
2. Scale up sample size, particularly for underrepresented groups
   (reasoning/correct n=13, hallucination/non_hallucinated n=17).
3. Build response length in as a covariate in the main analysis rather than
   a post-hoc check, especially for reasoning.
4. Investigate the layer-depth pattern: hallucination peaks mid-network
   (layer 11), sycophancy peaks near-final (layer 23) — check if this
   replicates and what it implies mechanistically.
5. Carry Fiedler value (hallucination) and spectral entropy (sycophancy)
   forward as the strongest candidates for the full head-level /
   phase-transition research project — these are the two findings that
   survived the length-confound check.
6. Run a normalized-Laplacian sensitivity check (as in the original paper's
   Appendix C.11) to help separate genuine topological signal from
   length-scale effects.

## Repository contents

- Kaggle notebook (`.ipynb`) — full 9-checkpoint pipeline
- `metrics_per_example_layer.csv` — one row per example × layer, all five
  spectral metrics
- `group_trajectory_summary.csv` — per-group length/confound summary
- `response_label_review.csv` — full labeling audit trail (proposed vs. final
  outcome, labeling method used per example)
- `plots/` — trajectory comparisons and zoomed single-layer comparisons per behavior
