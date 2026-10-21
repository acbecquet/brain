"""Check the ground-truth solver against physics we know analytically."""
import math

import torch

from lab.physics.schrodinger import (
    Grid, energy, evolve, gaussian_packet, mean_position, norm,
    potentials_sharp, potentials_smooth, random_states, trajectory,
)

GRID = Grid(n=256, length=40.0)


def test_norm_is_conserved():
    gen = torch.Generator().manual_seed(0)
    psi = random_states(8, GRID, gen)
    V = potentials_smooth(8, GRID, gen)
    out = evolve(psi, V, GRID, dt=5.0, substeps=500)
    assert torch.allclose(norm(out, GRID), torch.ones(8, dtype=torch.float64), atol=1e-10)


def test_energy_is_conserved():
    gen = torch.Generator().manual_seed(1)
    psi = random_states(8, GRID, gen)
    V = potentials_smooth(8, GRID, gen)
    e0 = energy(psi, V, GRID)
    e1 = energy(evolve(psi, V, GRID, dt=2.0, substeps=400), V, GRID)
    assert torch.allclose(e0, e1, rtol=1e-4)


def test_harmonic_oscillator_coherent_state():
    """A displaced ground state in V = ω²x²/2 oscillates as <x>(t) = x0 cos(ωt)."""
    omega, x0 = 0.5, 3.0
    V = (0.5 * omega**2 * GRID.x() ** 2).unsqueeze(0)
    psi0 = gaussian_packet(GRID, [x0], [1 / math.sqrt(2 * omega)], [0.0])
    traj = trajectory(psi0, V, GRID, dt=0.5, steps=12, substeps=50)
    times = 0.5 * torch.arange(13, dtype=torch.float64)
    expected = x0 * torch.cos(omega * times)
    got = mean_position(traj[0], GRID)
    assert torch.allclose(got, expected, atol=2e-5)


def test_split_step_is_second_order():
    """Strang splitting: 4x more substeps should cut the error ~16x."""
    omega, x0 = 0.5, 3.0
    V = (0.5 * omega**2 * GRID.x() ** 2).unsqueeze(0)
    psi0 = gaussian_packet(GRID, [x0], [1 / math.sqrt(2 * omega)], [0.0])
    times = 0.5 * torch.arange(13, dtype=torch.float64)
    errs = []
    for substeps in (50, 200):
        traj = trajectory(psi0, V, GRID, dt=0.5, steps=12, substeps=substeps)
        errs.append((mean_position(traj[0], GRID) - x0 * torch.cos(omega * times)).abs().max())
    assert 12 < errs[0] / errs[1] < 20


def test_free_packet_spreads_as_predicted():
    """Free Gaussian width grows as σ(t) = σ0 sqrt(1 + (t / 2σ0²)²)."""
    sigma0, t = 1.0, 3.0
    psi0 = gaussian_packet(GRID, [0.0], [sigma0], [0.0])
    psi = evolve(psi0, torch.zeros(1, GRID.n), GRID, dt=t, substeps=1)
    x = GRID.x()
    var = ((x**2) * psi.abs() ** 2).sum(-1) * GRID.dx
    expected = sigma0**2 * (1 + (t / (2 * sigma0**2)) ** 2)
    assert abs(var.item() - expected) < 1e-6


def test_potential_families_are_finite_and_bounded():
    gen = torch.Generator().manual_seed(2)
    grid = Grid()  # the grid the experiments use
    for family in (potentials_smooth, potentials_sharp):
        V = family(64, grid, gen)
        assert torch.isfinite(V).all()
        assert V.abs().max() < 25
