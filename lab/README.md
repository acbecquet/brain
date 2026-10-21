# lab — experiments toward a universal predictive model

The plan, hypotheses and verified references are in [`docs/RESEARCH_PLAN.md`](../docs/RESEARCH_PLAN.md).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**AMD GPU (RX 9070 XT).** Install a ROCm build of PyTorch (ROCm 7.2+) instead of the default wheel:

- Linux: follow the PyTorch "Get Started" selector with *ROCm*, e.g.
  `pip install torch --index-url https://download.pytorch.org/whl/rocm7.2` (check the selector for the current version).
- Windows: install *AMD Software: PyTorch on Windows* per AMD's release notes.

Verify with:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.hip, torch.cuda.get_device_name(0))"
```

ROCm appears through `torch.cuda`, so every script picks the GPU automatically. Force a device
with `--device cpu` or the `LAB_DEVICE` environment variable.

## Run

```bash
python -m pytest tests -q                       # physics + experiment sanity checks
python -m lab.exp0_wave.run --preset quick      # wavefunction learners
python -m lab.exp_b_abstain.run --preset quick  # "I don't know" training
python -m lab.exp0_wave.run --preset full       # bigger, 5 seeds, data-size sweep (GPU)
```

Results land in `results/<experiment>/<timestamp>.json` (raw per-seed numbers + config + git
commit) and a matching `.md` table of mean ± 95% CI.

## Layout

```
lab/common/      device selection (CUDA/ROCm/CPU), seeding, stats, result files
lab/physics/     exact 1D Schrödinger solver + random states and potentials
lab/exp0_wave/   Experiment 0: which models can learn quantum time evolution
lab/exp_b_abstain/ Experiment B: abstention and calibration training
```
