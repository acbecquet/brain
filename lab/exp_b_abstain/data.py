"""A toy world with knowable, half-knowable and unknowable questions.

Each question is three tokens (task, a, b):
  task FACT: the answer is an arbitrary random value attached to the key (a, b),
             like a person's birthday. It can only be known by memorizing it.
  task ADD:  the answer is (a + b) mod M. It can be known by learning the rule.

Facts appear in the pretraining stream with different frequencies:
  many    seen often      -> should be known
  few     seen a few times -> partially known
  single  seen once        -> barely known (Kalai et al. 2025: "singleton" facts drive hallucination)
  unseen  never seen       -> unknowable; the only right move is to abstain

Keys are split three ways per category: pretraining knowledge / post-training questions /
evaluation questions, so evaluation always asks questions not used in post-training.
"""
from dataclasses import dataclass

import torch

FACT, ADD = 0, 1
CATEGORIES = ["many", "few", "single", "unseen", "arith"]


@dataclass
class Questions:
    x: torch.Tensor         # (N, 3) long tokens
    y: torch.Tensor         # (N,) long answers in [0, M)
    category: torch.Tensor  # (N,) long index into CATEGORIES


@dataclass
class World:
    pretrain: Questions  # a stream with repetitions
    posttrain: Questions
    evaluation: Questions
    m: int


def build_world(seed: int, m: int = 48, counts=None, fractions=None) -> World:
    """Sample a world. counts = pretraining occurrences per category."""
    counts = counts or {"many": 30, "few": 4, "single": 1, "unseen": 0}
    fractions = fractions or {"many": 0.3, "few": 0.2, "single": 0.2, "unseen": 0.3}
    gen = torch.Generator().manual_seed(seed)
    keys = torch.cartesian_prod(torch.arange(m), torch.arange(m))
    values = torch.randint(0, m, (len(keys),), generator=gen)
    perm = torch.randperm(len(keys), generator=gen)

    pre, post, ev = [], [], []

    def add(bucket, task, k, y, cat, repeat=1):
        x = torch.cat([torch.full((len(k), 1), task), k], dim=1)
        bucket.append(Questions(x.repeat(repeat, 1), y.repeat(repeat), torch.full((len(k) * repeat,), cat)))

    start = 0
    for cat_name, frac in fractions.items():
        cat = CATEGORIES.index(cat_name)
        idx = perm[start:start + int(frac * len(keys))]
        start += len(idx)
        k, y = keys[idx], values[idx]
        if counts[cat_name] > 0:
            add(pre, FACT, k, y, cat, counts[cat_name])
        # Half of each category's keys are post-training questions, half are evaluation questions.
        half = len(idx) // 2
        add(post, FACT, k[:half], y[:half], cat)
        add(ev, FACT, k[half:], y[half:], cat)

    # Arithmetic: half of all pairs are shown in pretraining (4x each); the rest must be inferred.
    cat = CATEGORIES.index("arith")
    perm = torch.randperm(len(keys), generator=gen)
    seen, unseen = perm[: len(keys) // 2], perm[len(keys) // 2:]
    ans = (keys[:, 0] + keys[:, 1]) % m
    add(pre, ADD, keys[seen], ans[seen], cat, 4)
    half = len(unseen) // 2
    add(post, ADD, keys[unseen[:half]], ans[unseen[:half]], cat)
    add(ev, ADD, keys[unseen[half:]], ans[unseen[half:]], cat)

    def cat_all(parts):
        return Questions(*(torch.cat([getattr(p, f) for p in parts]) for f in ("x", "y", "category")))

    return World(cat_all(pre), cat_all(post), cat_all(ev), m)
