"""Sanity checks for the experiment code (not the results)."""
import torch

from lab.exp0_wave.data import make_data
from lab.exp0_wave.models import build
from lab.exp_b_abstain.data import CATEGORIES, build_world
from lab.exp_b_abstain.model import QAModel
from lab.exp_b_abstain.run import expected_reward
from lab.physics.schrodinger import Grid

GRID = Grid(n=64, length=20.0)


def test_every_wave_model_maps_shapes_and_starts_near_identity():
    data = make_data(4, "smooth", GRID, dt=0.2, steps=2, substeps=10, seed=0)
    psi, V, _ = data.pairs()
    for name in ["linear", "mlp", "cnn", "fno", "splitstep"]:
        out = build(name, GRID.n, 0.2)(psi, V)
        assert out.shape == psi.shape and out.is_complex()
        assert torch.allclose(out, psi, atol=1e-5), name  # zero-initialized output layers


def test_splitstep_is_unitary_for_any_parameters():
    model = build("splitstep", GRID.n, 0.2)
    for p in model.parameters():
        torch.nn.init.normal_(p)
    data = make_data(4, "sharp", GRID, dt=0.2, steps=1, substeps=10, seed=1)
    psi, V, _ = data.pairs()
    out = model(psi, V)
    assert torch.allclose(out.abs().pow(2).sum(-1), psi.abs().pow(2).sum(-1), rtol=1e-4)


def test_abstain_world_splits_are_disjoint():
    world = build_world(0, m=16)
    key = lambda q: {tuple(r) for r in q.x.tolist()}
    pre, post, ev = key(world.pretrain), key(world.posttrain), key(world.evaluation)
    assert not post & ev
    unseen = CATEGORIES.index("unseen")
    unseen_eval = {tuple(r) for r in world.evaluation.x[world.evaluation.category == unseen].tolist()}
    assert not unseen_eval & pre


def test_expected_reward_matches_hand_computation():
    probs = torch.tensor([[0.6, 0.1, 0.3]])  # answers 0, 1 and IDK (index 2)
    y = torch.tensor([0])
    # right 0.6, wrong 0.1, idk 0.3 -> 0.6 - 3 * 0.1
    assert torch.allclose(expected_reward(probs, y, idk=2, wrong_penalty=3.0), torch.tensor([0.3]))


def test_qa_model_outputs():
    model = QAModel(m=16, d=32)
    x = torch.tensor([[0, 1, 2], [1, 3, 4]])
    logits, conf = model(x)
    assert logits.shape == (2, 17) and conf.shape == (2,)
