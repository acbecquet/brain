"""Experiment 0: which learners can learn quantum time evolution?

Hypotheses (stated before running):
  H1  A model that ignores V (linear) fails on varied potentials -> the rule must be conditioned on V.
  H2  The structured split-step model beats generic nets on every metric with ~100x fewer parameters.
  H3  Generic nets (MLP/CNN/FNO) leak probability (norm drift) and break superposition (linearity),
      and their errors grow fastest on long rollouts and out-of-distribution potentials.

Usage:
    python -m lab.exp0_wave.run --preset quick        # minutes on CPU
    python -m lab.exp0_wave.run --preset full         # GPU recommended
"""
import argparse
import math
import time

import torch

from lab.common.device import describe_device, get_device, seed_everything
from lab.common.results import save_run
from lab.physics.schrodinger import Grid

from .data import make_data
from .models import build, n_params

PRESETS = {
    "quick": dict(n=128, train_traj=512, train_len=10, test_traj=128, test_len=40,
                  steps=2000, batch=64, seeds=[0, 1, 2], train_sizes=[512]),
    "full": dict(n=128, train_traj=4096, train_len=10, test_traj=512, test_len=40,
                 steps=20000, batch=128, seeds=[0, 1, 2, 3, 4], train_sizes=[128, 512, 4096]),
}
MODELS = ["linear", "mlp", "cnn", "fno", "splitstep", "splitstep_law"]
LR = {"splitstep": 1e-2, "splitstep_law": 1e-2}
DT, SUBSTEPS, LENGTH = 0.2, 20, 20.0

METRICS = ["params", "step_err_id", "step_err_ood", "rollout10_id", "rollout40_id",
           "rollout40_ood", "norm_drift40", "linearity_err", "train_sec"]


def rel_err(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Per-sample relative L2 error ||pred - true|| / ||true||."""
    return (pred - true).abs().pow(2).sum(-1).sqrt() / true.abs().pow(2).sum(-1).sqrt()


def train(model, data, cfg, device, seed):
    psi_in, V, psi_out = (t.to(device) for t in data.pairs())
    opt = torch.optim.AdamW(model.parameters(), lr=LR.get(cfg["model"], 1e-3), weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=cfg["steps"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model.train()
    for _ in range(cfg["steps"]):
        idx = torch.randint(0, psi_in.shape[0], (cfg["batch"],), generator=gen).to(device)
        pred = model(psi_in[idx], V[idx])
        loss = rel_err(pred, psi_out[idx]).pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()


@torch.no_grad()
def rollout(model, psi0, V, steps):
    states, psi = [], psi0
    for _ in range(steps):
        psi = model(psi, V)
        states.append(psi)
    return torch.stack(states, dim=1)


@torch.no_grad()
def evaluate(model, test_id, test_ood, grid, device) -> dict[str, float]:
    model.eval()
    out = {}
    for tag, data in (("id", test_id), ("ood", test_ood)):
        traj, V = data.traj.to(device), data.V.to(device)
        psi_in, Vp, psi_out = (t.to(device) for t in data.pairs())
        out[f"step_err_{tag}"] = rel_err(model(psi_in, Vp), psi_out).mean().item()
        pred = rollout(model, traj[:, 0], V, traj.shape[1] - 1)
        errs = rel_err(pred, traj[:, 1:])  # (batch, steps)
        if tag == "id":
            out["rollout10_id"] = errs[:, 9].mean().item()
            out["rollout40_id"] = errs[:, 39].mean().item()
            norms = pred[:, 39].abs().pow(2).sum(-1) * grid.dx
            out["norm_drift40"] = (norms - 1).abs().mean().item()
        else:
            out["rollout40_ood"] = errs[:, 39].mean().item()
    # Superposition test: the true map is linear, so f(aψ1 + bψ2) should equal a f(ψ1) + b f(ψ2).
    traj, V = test_id.traj.to(device), test_id.V.to(device)
    g = torch.Generator().manual_seed(123)
    a = torch.exp(2j * math.pi * torch.rand(traj.shape[0], 1, generator=g)).to(device) * 0.6
    b = torch.exp(2j * math.pi * torch.rand(traj.shape[0], 1, generator=g)).to(device) * 0.6
    psi1, psi2 = traj[:, 0], traj.roll(1, dims=0)[:, 0]  # same V, two different states
    lhs = model(a * psi1 + b * psi2, V)
    rhs = a * model(psi1, V) + b * model(psi2, V)
    out["linearity_err"] = rel_err(lhs, rhs).mean().item()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=PRESETS, default="quick")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = dict(PRESETS[args.preset], dt=DT, substeps=SUBSTEPS, length=LENGTH, models=args.models)
    device = get_device(args.device)
    print(f"device: {describe_device(device)}")
    grid = Grid(cfg["n"], LENGTH)

    per_model: dict[str, dict[str, list[float]]] = {}
    for seed in cfg["seeds"]:
        # Test sets use seeds disjoint from training so no trajectory is shared.
        test_id = make_data(cfg["test_traj"], "smooth", grid, DT, cfg["test_len"], SUBSTEPS, 10_000 + seed)
        test_ood = make_data(cfg["test_traj"], "sharp", grid, DT, cfg["test_len"], SUBSTEPS, 20_000 + seed)
        full_train = make_data(max(cfg["train_sizes"]), "smooth", grid, DT, cfg["train_len"], SUBSTEPS, seed)
        for size in cfg["train_sizes"]:
            train_data = type(full_train)(full_train.traj[:size], full_train.V[:size])
            for name in args.models:
                seed_everything(seed)
                model = build(name, cfg["n"], DT, LENGTH).to(device)
                start = time.time()
                train(model, train_data, dict(cfg, model=name), device, seed)
                result = evaluate(model, test_id, test_ood, grid, device)
                result["train_sec"] = time.time() - start
                result["params"] = n_params(model)
                key = name if len(cfg["train_sizes"]) == 1 else f"{name}@{size}"
                for metric, value in result.items():
                    per_model.setdefault(key, {}).setdefault(metric, []).append(value)
                print(f"seed {seed} {key:16s} " + " ".join(f"{m}={result[m]:.3g}" for m in METRICS))

    path = save_run("exp0_wave", cfg, per_model, METRICS,
                    notes="Errors are relative L2. norm_drift40 = |∫|ψ|²dx − 1| after a 40-step rollout. "
                          "linearity_err = relative violation of superposition (truth = 0).")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
