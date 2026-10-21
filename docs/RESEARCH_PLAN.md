# Research plan: toward a universal predictive model

## Framing

Newtonian mechanics is a *limit* of quantum mechanics. An LLM is not a limit of a quantum
model in any usable sense: language emerges at a scale (brains, societies) that no one can
reach by simulating particles. What the two do share is the **objective**: given the present,
predict a distribution over what comes next.

So the working definition used here:

> A universal model = one predictive learner that builds an effective model at each scale,
> plus learned bridges between scales — not a physics model with language hidden inside it.

This maps onto the brain design already in this repo (`core/`, `attention/`, `system/`):
specialist experts per scale/domain, thalamic gating that learns which level of description a
problem needs, confidence-reporting experts, and neuromodulation that sets learning and
exploration rates.

## Evidence standards (every experiment)

- Hypotheses written in the experiment's docstring **before** running it.
- Matched data and training budget across compared models; parameter counts reported.
- ≥3 seeds for quick runs, ≥5 for full runs; results reported as mean ± 95% CI (t-distribution).
- Held-out test sets generated from disjoint seeds; an out-of-distribution test where possible.
- Every run writes `results/<experiment>/<timestamp>.json` (raw per-seed numbers, config,
  git commit) and a `.md` summary table. Claims in write-ups cite those files.
- Simple baselines included so "better" always means "better than something obvious".

## Experiments

| # | Status | Question | Built on |
|---|---|---|---|
| 0 | **built** | Which learners can learn quantum time evolution, and does built-in structure beat scale? | Split-step Fourier solver (tested against analytic results) |
| A | planned | Does one generalist, separate specialists, or the brain-style gated MoE learn text + physics + arithmetic best at equal compute? | Exp 0 simulator; MoE routing results (arXiv 2604.07030) |
| B | **built** | Does training for "I don't know" beat accuracy-only training? | Kalai et al. 2025; TruthRL; RLCR |
| C | planned | Does an agent that *chooses its experiments* discover a hidden law faster than a passive learner? | NewtonBench, DiscoverPhysics; Exp 0 simulator |
| D | planned (GPU) | Does label-free RL really teach anything, and does it depend on the model family? | TTRL, INTUITOR, Spurious Rewards, ReasonMaxxer |
| E | stretch | Can multi-timescale memory / neuromodulated learning rates prevent forgetting when domains arrive in sequence? | Nested Learning / HOPE |

### 0 — Wavefunction learners (`lab/exp0_wave`)
Predict ψ(t+Δt) from (ψ(t), V). Models: linear (ignores V), MLP, CNN, FNO, and a structured
split-step net (learns dispersion ω(k) and potential response g(V); exactly unitary).
Metrics: one-step error in- and out-of-distribution (sharp barriers/double wells never seen in
training), 10- and 40-step rollout error, probability leakage, superposition violation.

### A — Generalist vs specialists vs gated experts (planned)
Same small transformer budget, three domains tokenized into one stream: TinyStories text,
quantized trajectories from the Exp 0 simulator, modular arithmetic. Arms: one dense
generalist; three specialists (summed budget); MoE with your gating design. 2026 additions:
- routing-balance variants, since balancing scope decides whether experts specialize
  ([arXiv 2604.07030](https://arxiv.org/pdf/2604.07030));
- a JEPA-style latent-prediction auxiliary loss, which evidence says helps as an *add-on* to
  next-token loss rather than a replacement ([VL-JEPA, ICLR 2026](https://bdtechtalks.substack.com/p/metas-new-vl-jepa-model-shifts-from));
- Muon vs AdamW as a recipe check ([IMU-1, arXiv 2602.02522](https://arxiv.org/pdf/2602.02522); [Muon is Scalable, arXiv 2502.16982](https://arxiv.org/html/2502.16982v1)).

Metrics: held-out loss per domain, transfer speed to a new domain, expert-usage entropy.

### B — Abstention and calibration (`lab/exp_b_abstain`)
A toy world of facts seen many/few/once/never plus arithmetic. All regimes post-train the same
pretrained model with the exact expected reward: binary; binary + post-hoc threshold; ternary
(+1/0/−1, [TruthRL](https://arxiv.org/pdf/2509.25760)); penalized (+1/0/−k, the confidence-target
scoring proposed by [Kalai et al. 2025](https://arxiv.org/abs/2509.04664)); and a Brier-calibrated
confidence head ([RLCR](https://arxiv.org/pdf/2507.16806)). Metrics: accuracy, hallucination,
abstention, k-penalized score, calibration error, broken down by how often a fact was seen.
Known risk to watch: abstention can collapse the reward gradient
([arXiv 2608.00301](https://arxiv.org/pdf/2608.00301)).

### C — The scientific method as an agent (planned)
Hidden potential V in the Exp 0 simulator. The agent chooses initial states to "launch" and
observes the outcomes, then updates its estimate of V. Arms: random experiments; experiments
chosen where an ensemble of models disagrees most (active learning); a variant that only sees
**sampled measurement positions** (Born-rule samples, never ψ itself) — the realistic way we
learn about nature. Metric: error in recovered V vs number of experiments. Modeled on
[NewtonBench (ICLR 2026)](https://arxiv.org/abs/2510.07172), which found agents degrade sharply
with system complexity and noise, and [DiscoverPhysics](https://arxiv.org/pdf/2605.26087).
This merges two earlier ideas (the inverse problem and measurement-only learning).

### D — Label-free RL, replicated (planned, needs the GPU)
Small pretrained models (Qwen2.5 0.5B/1.5B, Llama-3.2-1B, OLMo-2-1B) on grade-school math.
Rewards: ground truth; majority vote ([TTRL](https://arxiv.org/abs/2504.16084)); self-certainty
([INTUITOR](https://arxiv.org/pdf/2505.19590)); random ([Spurious Rewards](https://proceedings.mlr.press/v306/shao26d.html));
plus the RL-free ReasonMaxxer baseline ([arXiv 2605.06241](https://arxiv.org/pdf/2605.06241)).
Track collapse over training ([Co-rewarding, ICLR 2026](https://arxiv.org/pdf/2508.00410)).
Prediction from the literature: gains depend on the model family and mostly sharpen what
pretraining already put there.

### E — Continual learning with multi-timescale memory (stretch)
Domains arrive one after another. Compare a fixed learning rate, a neuromodulated rate
(surprise raises plasticity, as in `system/neuromodulation.py`), and multi-timescale parameters
inspired by [Nested Learning / HOPE](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/).
Metric: forgetting of earlier domains vs learning speed on new ones.

### Dropped or deferred
- Two-particle and spin-chain systems: deferred until Exp 0/C show which models scale.
- A standalone Hamiltonian inverse problem and measurement-only learning: merged into C.

## Verified literature (checked October 2026)

| Claim used here | Source | Notes |
|---|---|---|
| Hallucination is partly caused by grading that rewards guessing; fix the scoring | Kalai, Nachum, Vempala, Zhang, [arXiv 2509.04664](https://arxiv.org/abs/2509.04664) (Sept 2025) | OpenAI + Georgia Tech |
| Majority-vote reward on unlabeled questions boosts math reasoning | Zuo et al., [TTRL, arXiv 2504.16084](https://arxiv.org/abs/2504.16084), NeurIPS 2025 | ~211% AIME-2024 pass@1 gain on Qwen2.5-Math-7B (v2) |
| Self-certainty as the only reward matches GRPO on math | Zhao, Kang, Feng, Levine, Song, [INTUITOR, arXiv 2505.19590](https://arxiv.org/pdf/2505.19590) | no independent replication found |
| Self-play with a code executor, zero human data | Zhao et al., [Absolute Zero, arXiv 2505.03335](https://arxiv.org/abs/2505.03335), NeurIPS 2025 | |
| Random rewards help Qwen2.5-Math but not Llama/OLMo | Shao et al., [Spurious Rewards, arXiv 2506.10947](https://proceedings.mlr.press/v306/shao26d.html), ICML 2026 | +21.4 pts MATH-500 random vs +29.1 ground truth |
| RL mostly re-weights 1–3% of tokens toward choices the base model already ranks top-5 | Akgül et al., [arXiv 2605.06241](https://arxiv.org/pdf/2605.06241) (May 2026) | preprint; claims unverified |
| Self-rewarding RL frequently collapses; mitigations proposed | [Co-rewarding, ICLR 2026](https://arxiv.org/pdf/2508.00410); [RESTRAIN](https://arxiv.org/pdf/2510.02172); [JURY-RL](https://arxiv.org/html/2604.25419) | |
| Ternary reward (+1/0/−1) reduces hallucination | [TruthRL, arXiv 2509.25760](https://arxiv.org/pdf/2509.25760) | later criticized for treating abstention like error |
| Calibration reward (Brier) on stated confidence | [RLCR, arXiv 2507.16806](https://arxiv.org/pdf/2507.16806), ICLR 2026 | |
| Abstention + clarification via verifiable RL | [Abstain-R1, arXiv 2604.17073](https://arxiv.org/pdf/2604.17073) | |
| AlphaProof: RL on formal (Lean) proofs, IMO 2024 silver level | Hubert et al., *Nature* (Nov 2025), [DeepMind](https://research.google/pubs/olympiad-level-formal-mathematical-reasoning-with-reinforcement-learning/) | 80M auto-formalized statements; test-time RL |
| AlphaEvolve: 48-multiplication 4×4 complex matmul; 23% speedup of a Gemini matmul kernel (1% training time) | [DeepMind blog, May 2025](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) | |
| One network for Atari, captioning, chat, robot arm (604 tasks) | Reed et al., [Gato, arXiv 2205.06175](https://arxiv.org/abs/2205.06175), TMLR 2022 | |
| Automated idea → experiment → paper pipeline | Lu et al., [The AI Scientist, arXiv 2408.06292](https://arxiv.org/abs/2408.06292) | |
| Agents discovering altered physical laws by experimenting | [NewtonBench, ICLR 2026](https://arxiv.org/abs/2510.07172); [DiscoverPhysics](https://arxiv.org/pdf/2605.26087) | |
| JEPA objectives help as add-ons; strongest evidence in vision/video | [VL-JEPA (ICLR 2026)](https://bdtechtalks.substack.com/p/metas-new-vl-jepa-model-shifts-from); [Causal-JEPA](https://arxiv.org/abs/2602.11389) | text-only results weaker |
| Multi-timescale "nested" optimization for continual learning | [Nested Learning / HOPE](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/), NeurIPS 2025 | company-reported results |

## Hardware

Target: Ryzen 7 9850X3D, 32 GB DDR5, Radeon RX 9070 XT (16 GB). AMD lists the RX 9070 XT as
supported by PyTorch on ROCm 7.2 for Windows and Linux
([AMD release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-WINDOWS-PYTORCH-7-2.html)).
ROCm builds expose the GPU as `torch.cuda`, so the code needs no changes. Linux is the more
mature path; one user report found 9070 XT training slower than comparable NVIDIA cards on an
older ROCm ([ROCm/TheRock#2510](https://github.com/ROCm/TheRock/issues/2510)), so we benchmark
before relying on GPU timings.
