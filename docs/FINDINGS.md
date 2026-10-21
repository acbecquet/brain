# Findings so far

All numbers: mean ± 95% CI over 3 seeds, quick presets, CPU. Raw per-seed data is in the
`results/` files cited under each table. These are small-scale results; the `full` presets on the
GPU (5 seeds, more data) should be run before treating any of them as settled.

## Experiment 0 — learning quantum time evolution

Sources: `results/exp0_wave/20261021-050818.*` (five models) and the follow-up run comparing
`splitstep` and `splitstep_law` in the same directory. Both runs use identical seeds and data.

| model | params | 1-step err (in-dist) | 1-step err (OOD) | 40-step rollout err | 40-step OOD | probability leak | superposition error |
|---|---|---|---|---|---|---|---|
| linear (ignores V) | 32,768 | 0.238 | 0.386 | 28.6 | 32.4 | 5,577 | 3e-7 |
| MLP | 853,760 | 0.154 | 0.355 | 1.04 | 1.71 | 0.52 | 0.070 |
| CNN | 165,506 | 0.011 | 0.041 | 0.44 | 0.94 | 0.27 | 0.012 |
| FNO | 270,978 | 0.018 | 0.069 | 1.69 | 2.24 | 6.0 | 0.018 |
| split-step (free ω(k)) | 1,281 | 0.012 | 0.058 | 0.167 | 0.43 | 1.3e-5 | 4e-7 |
| **split-step, ω(k) a polynomial** | **1,156** | **0.0036** | **0.0099** | **0.076** | **0.21** | **1.3e-5** | **4e-7** |

Hypotheses (stated in `lab/exp0_wave/run.py` before running):

- **H1, a model that ignores V fails: supported.** One-step error is 0.24, and probability
  blows up by a factor of thousands over a rollout.
- **H2, the structured model wins every metric with ~100× fewer parameters: partly refuted,
  then fixed.** The free split-step model tied the CNN on in-distribution one-step error and
  *lost* to it on out-of-distribution potentials (0.058 vs 0.041). It won long rollouts,
  probability conservation and superposition by large margins.
- **H3, generic nets leak probability, break superposition and degrade on long rollouts:
  supported.** The CNN looks as good as the structured model at one step but is 2.7× worse at
  40 steps. **One-step error is a misleading metric** for simulators.

Why the free split-step model lost out of distribution (diagnosed after the run):

- After removing a gauge offset (a constant added to ω and subtracted from g(V) leaves the
  dynamics unchanged), the learned ω(k) matches the true law k²/2 to within 0.015 for |k| < 3.
  For |k| > 6 it is off by 30–115.
- Training data has only ~1e-6 of its power at |k| > 6. Sharp OOD potentials have ~70–100× more.
- The learned potential response g(V) ≈ V up to V ≈ 10 and is flat above. Larger V only
  occurs at the box edges, where ψ ≈ 0, so no signal ever reached it.

**The model rediscovered the Schrödinger equation exactly where the data constrained it, and
nowhere else.** Assuming instead that ω(k) is a low-order polynomial in k² (`splitstep_law`) lets
the well-sampled modes fix the poorly sampled ones. That cut OOD one-step error 6× and halved
rollout error. Lesson for the larger project: the right prior about the *form* of a law buys
generalization that neither more parameters nor more in-distribution data provides.

## Experiment B — training to say "I don't know"

Sources: `results/exp_b_abstain/quick/` (pre-registered run) and `results/exp_b_abstain/quick_v2/`.

The pretrained model has graded knowledge. Accuracy is 99.5% on facts seen 30 times, 87% on
facts seen 4 times, 36% on facts seen once and 2% (chance) on unseen facts. It is confidently
wrong on 98% of unseen facts. Arithmetic was **not** learned in 3 epochs (≈1% accuracy), so in
this run it behaves as another unknowable category; that is a limitation of this run.

| regime | quick: score k=1 | quick: score k=3 | quick: hallucination on unseen facts | quick: accuracy on known facts | v2: score k=1 | v2: abstain rate |
|---|---|---|---|---|---|---|
| none | −0.25 | −1.49 | 0.98 | 0.995 | −0.25 | 0 |
| binary | −0.32 | −1.63 | 0.98 | 0.94 | −0.39 | 0 |
| binary + threshold | 0.055 | −0.39 | 0.34 | 0.87 | **0.067** | 0.63 ± 0.01 |
| ternary (+1/0/−1) | 0.097 | −0.30 | 0.61 | 0.86 | −0.38 | 0 |
| penalized (+1/0/−3) | **0.115** | **−0.20** | 0.51 | 0.80 | −0.32 | 0.19 ± 0.72 |
| Brier confidence | −0.10 | −0.80 | 0.76 | 0.71 | −0.02 | 0.79 ± 0.31 |

Reference: always answering "I don't know" scores 0 on every k.

- **H1, binary post-training does not reduce hallucination: supported.** Unseen-fact
  hallucination stays at 98%. Post-training also cut known-fact accuracy by 5–30 points in
  *every* regime, because learning new facts overwrote old ones. That is catastrophic forgetting,
  the subject of planned Experiment E.
- **H2, penalized gets the best k=3 score: supported in the quick run** (−0.20, vs −0.30 for
  ternary). But no trained regime beat the trivial "always abstain" baseline at k=3.
- **H3, a post-hoc threshold recovers much of the benefit: supported, and it was the most
  reliable regime.** It is within noise of the best trained regime at k=1 in both versions,
  needs no retraining, and has low variance across seeds.
- **H4, abstention training over-abstains on known facts: supported.** Abstention on
  often-seen facts was 13–23%.

**Design confound found and fixed.** The quick preset revisits each post-training question ~89
times, so models partly memorize answers rather than learn when to abstain. `quick_v2` makes ~2
passes. In v2, training-based abstention became **bistable**: penalized and Brier regimes
collapse to abstaining on nearly everything in some seeds and to never abstaining in others
(abstain-rate CIs of ±0.7). Ternary never learned to abstain. This matches the reported failure
mode in which abstention kills the reward gradient
([arXiv 2608.00301](https://arxiv.org/pdf/2608.00301)).

**Takeaway:** at this scale, calibrated confidence plus a decision threshold is a more reliable
route to "I don't know" than reward-shaping alone. Next steps:

1. A KL anchor to the pretrained model during post-training (the standard RL stabilizer).
2. Calibrating the confidence head on a held-out split.
3. Rerun with arithmetic actually learned, so a rule-based category is present.
