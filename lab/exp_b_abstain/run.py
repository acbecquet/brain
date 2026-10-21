"""Experiment B: does training for "I don't know" beat training for accuracy alone?

All regimes start from the same pretrained model and post-train on the same questions.
Post-training maximizes the *exact* expected reward Σ_a p(a)·r(a) over the M answers + IDK
(no sampling noise, so differences between regimes come from the reward, not RL variance).

Regimes
  none          pretrained only
  binary        r = 1 if right else 0            (accuracy-only grading)
  binary+thr    binary model, abstain post hoc when max p < k/(k+1)   (no retraining)
  ternary       r = +1 / 0 (IDK) / -1            (TruthRL-style, arXiv 2509.25760)
  penalized     r = +1 / 0 (IDK) / -k            (Kalai et al. 2025 confidence-target scoring)
  brier         binary answer reward + Brier loss on a confidence head; abstain when conf < k/(k+1)
                (RLCR-style calibration reward, arXiv 2507.16806)

Hypotheses
  H1  binary post-training hallucinates on unseen/singleton facts at least as much as pretraining.
  H2  penalized (k=3) gets the best k=3 score; ternary sits between binary and penalized.
  H3  binary+thr recovers much of the benefit without retraining *if* the model is calibrated.
  H4  abstention-trained regimes over-abstain on some knowable questions (the price of safety).

Usage:
    python -m lab.exp_b_abstain.run --preset quick
"""
import argparse
import copy
import math

import torch
import torch.nn.functional as F

from lab.common.device import describe_device, get_device, seed_everything
from lab.common.results import save_run

from .data import CATEGORIES, build_world
from .model import QAModel

PRESETS = {
    "quick": dict(m=48, pretrain_epochs=3, post_steps=600, post_batch=256, post_lr=3e-4,
                  seeds=[0, 1, 2], k=3.0),
    # v2: the quick preset revisits each post-training question ~89 times, so models memorize
    # the answers instead of learning when to abstain. v2 makes ~2 passes, like RL on fresh prompts.
    "quick_v2": dict(m=48, pretrain_epochs=3, post_epochs=2, post_batch=32, post_lr=1e-3,
                     seeds=[0, 1, 2], k=3.0),
    "full": dict(m=64, pretrain_epochs=3, post_epochs=2, post_batch=32, post_lr=1e-3,
                 seeds=[0, 1, 2, 3, 4], k=3.0),
}
REGIMES = ["none", "binary", "binary+thr", "ternary", "penalized", "brier"]
METRICS = ["acc", "halluc", "abstain", "score_k1", "score_k3", "ece",
           "halluc_unseen", "halluc_single", "acc_many", "acc_arith", "abstain_many"]


def pretrain(model, world, epochs, device, seed):
    q = world.pretrain
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(q.y), generator=gen).split(64):
            logits, _ = model(q.x[idx].to(device))
            loss = F.cross_entropy(logits, q.y[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()


def expected_reward(probs, y, idk, wrong_penalty):
    p_right = probs.gather(1, y[:, None]).squeeze(1)
    p_idk = probs[:, idk]
    return p_right - wrong_penalty * (1 - p_right - p_idk)


def posttrain(model, world, regime, cfg, device, seed):
    q = world.posttrain
    k, batch = cfg["k"], cfg["post_batch"]
    steps = cfg.get("post_steps") or math.ceil(cfg["post_epochs"] * len(q.y) / batch)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["post_lr"], weight_decay=0.0)
    gen = torch.Generator().manual_seed(seed)
    penalty = {"binary": 0.0, "brier": 0.0, "ternary": 1.0, "penalized": k}[regime]
    for _ in range(steps):
        idx = torch.randint(0, len(q.y), (batch,), generator=gen)
        x, y = q.x[idx].to(device), q.y[idx].to(device)
        logits, conf_logit = model(x)
        probs = logits.softmax(-1)
        loss = -expected_reward(probs, y, model.idk, penalty).mean()
        if regime == "brier":
            correct = (logits[:, : model.m].argmax(-1) == y).float()
            loss = loss + (torch.sigmoid(conf_logit) - correct).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


def ece(conf, correct, bins=10):
    """Expected calibration error: average |confidence - accuracy| over confidence bins."""
    edges = torch.linspace(0, 1, bins + 1, device=conf.device)
    total = torch.zeros((), device=conf.device)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.float().mean() * (conf[mask].mean() - correct[mask].mean()).abs()
    return total.item()


@torch.no_grad()
def evaluate(model, world, regime, k, device):
    q = world.evaluation
    x, y, cat = q.x.to(device), q.y.to(device), q.category.to(device)
    logits, conf_logit = model(x)
    probs = logits.softmax(-1)
    answer_probs = probs[:, : model.m]
    guess = answer_probs.argmax(-1)
    threshold = k / (k + 1)
    if regime == "brier":
        confidence = torch.sigmoid(conf_logit)
        abstain = confidence < threshold
    elif regime == "binary+thr":
        confidence = answer_probs.max(-1).values
        abstain = confidence < threshold
    else:
        confidence = answer_probs.max(-1).values
        abstain = probs.argmax(-1) == model.idk
    right = (~abstain) & (guess == y)
    wrong = (~abstain) & (guess != y)
    out = {
        "acc": right.float().mean().item(),
        "halluc": wrong.float().mean().item(),
        "abstain": abstain.float().mean().item(),
        "ece": ece(confidence, (guess == y).float()),
    }
    out["score_k1"] = out["acc"] - out["halluc"]
    out["score_k3"] = out["acc"] - k * out["halluc"]
    by = lambda name: cat == CATEGORIES.index(name)
    out["halluc_unseen"] = wrong[by("unseen")].float().mean().item()
    out["halluc_single"] = wrong[by("single")].float().mean().item()
    out["acc_many"] = right[by("many")].float().mean().item()
    out["acc_arith"] = right[by("arith")].float().mean().item()
    out["abstain_many"] = abstain[by("many")].float().mean().item()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = PRESETS[args.preset]
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")

    per_regime: dict[str, dict[str, list[float]]] = {}
    for seed in cfg["seeds"]:
        seed_everything(seed)
        world = build_world(seed, cfg["m"])
        base = QAModel(cfg["m"]).to(device)
        pretrain(base, world, cfg["pretrain_epochs"], device, seed)
        trained = {"none": base}
        for regime in ["binary", "ternary", "penalized", "brier"]:
            seed_everything(seed)
            model = copy.deepcopy(base)
            posttrain(model, world, regime, cfg, device, seed)
            trained[regime] = model
        trained["binary+thr"] = trained["binary"]
        for regime in REGIMES:
            result = evaluate(trained[regime], world, regime, cfg["k"], device)
            for metric, value in result.items():
                per_regime.setdefault(regime, {}).setdefault(metric, []).append(value)
            print(f"seed {seed} {regime:11s} " + " ".join(f"{m}={result[m]:.3f}" for m in METRICS))

    path = save_run(f"exp_b_abstain/{args.preset}", cfg, per_regime, METRICS,
                    notes="acc/halluc/abstain are fractions of all evaluation questions. "
                          "score_k = acc − k·halluc (abstaining scores 0). ece = expected calibration error. "
                          "Per-category columns are fractions within that category. "
                          "Reference: always answering IDK scores 0 on every score_k.")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
