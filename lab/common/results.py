"""Save experiment results as JSON + a Markdown table, so every claim has a file behind it."""
import json
import platform
import subprocess
import time
from pathlib import Path

import torch

from .stats import summarize

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def markdown_table(per_model: dict[str, dict[str, list[float]]], metrics: list[str]) -> str:
    """Render {model: {metric: [value per seed]}} as 'mean ± 95% CI' Markdown rows."""
    lines = ["| model | " + " | ".join(metrics) + " |", "|---" * (len(metrics) + 1) + "|"]
    for model, values in per_model.items():
        cells = [summarize(values[m]).fmt() if m in values else "—" for m in metrics]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save_run(
    experiment: str,
    config: dict,
    per_model: dict[str, dict[str, list[float]]],
    metrics: list[str],
    notes: str = "",
) -> Path:
    """Write results/<experiment>/<timestamp>.{json,md} and return the JSON path."""
    out_dir = RESULTS_ROOT / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    record = {
        "experiment": experiment,
        "timestamp": stamp,
        "git_commit": _git_commit(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "config": config,
        "per_model": per_model,
    }
    json_path = out_dir / f"{stamp}.json"
    json_path.write_text(json.dumps(record, indent=2))
    table = markdown_table(per_model, metrics)
    (out_dir / f"{stamp}.md").write_text(
        f"# {experiment} — {stamp}\n\nconfig: `{json.dumps(config)}`\n\n"
        f"Values are mean ± 95% CI over seeds.\n\n{table}\n\n{notes}\n"
    )
    print(table)
    return json_path
