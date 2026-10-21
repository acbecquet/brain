"""Ground-truth time evolution for the 1D Schrödinger equation.

Units: hbar = m = 1, so   i dψ/dt = -1/2 d²ψ/dx² + V(x) ψ.

The box is [-L/2, L/2) with periodic boundaries. The integrator is Strang
split-step Fourier: half a potential kick, a full kinetic step in momentum
space, another half kick. Each piece is a pure phase, so the method is exactly
unitary (total probability stays 1 up to float round-off).

Everything is batched over the leading dimension and runs on CPU or GPU.
"""
import math
from dataclasses import dataclass

import torch

DTYPE = torch.complex128  # ground truth in double precision


@dataclass(frozen=True)
class Grid:
    n: int = 128
    length: float = 20.0

    @property
    def dx(self) -> float:
        return self.length / self.n

    def x(self, device=None) -> torch.Tensor:
        return -self.length / 2 + self.dx * torch.arange(self.n, device=device, dtype=torch.float64)

    def k(self, device=None) -> torch.Tensor:
        """Angular wavenumbers in FFT order."""
        return 2 * math.pi * torch.fft.fftfreq(self.n, d=self.dx, device=device, dtype=torch.float64)


def norm(psi: torch.Tensor, grid: Grid) -> torch.Tensor:
    """Total probability ∫|ψ|² dx for each item in the batch."""
    return (psi.abs() ** 2).sum(-1) * grid.dx


def normalize(psi: torch.Tensor, grid: Grid) -> torch.Tensor:
    return psi / norm(psi, grid).sqrt().unsqueeze(-1)


def energy(psi: torch.Tensor, V: torch.Tensor, grid: Grid) -> torch.Tensor:
    """<ψ|H|ψ> = kinetic (computed in momentum space) + potential."""
    k = grid.k(psi.device)
    psi_k = torch.fft.fft(psi, dim=-1)
    # Parseval: sum|ψ|² dx = sum|ψ_k|² dx / n
    kinetic = 0.5 * ((psi_k.abs() ** 2) * k**2).sum(-1) * grid.dx / grid.n
    potential = (V * psi.abs() ** 2).sum(-1) * grid.dx
    return kinetic + potential


def mean_position(psi: torch.Tensor, grid: Grid) -> torch.Tensor:
    return (grid.x(psi.device) * psi.abs() ** 2).sum(-1) * grid.dx / norm(psi, grid)


def evolve(psi: torch.Tensor, V: torch.Tensor, grid: Grid, dt: float, substeps: int) -> torch.Tensor:
    """Advance ψ by total time dt using `substeps` Strang steps."""
    h = dt / substeps
    k = grid.k(psi.device)
    half_kick = torch.exp(-0.5j * h * V.to(torch.float64))
    drift = torch.exp(-0.5j * h * k**2)
    psi = psi.to(DTYPE)
    for _ in range(substeps):
        psi = half_kick * psi
        psi = torch.fft.ifft(drift * torch.fft.fft(psi, dim=-1), dim=-1)
        psi = half_kick * psi
    return psi


def trajectory(psi0: torch.Tensor, V: torch.Tensor, grid: Grid, dt: float, steps: int, substeps: int) -> torch.Tensor:
    """Return states at t = 0, dt, ..., steps*dt with shape (batch, steps+1, n)."""
    states = [psi0.to(DTYPE)]
    for _ in range(steps):
        states.append(evolve(states[-1], V, grid, dt, substeps))
    return torch.stack(states, dim=1)


# --------------------------------------------------------------------------
# Random initial states and potential families
# --------------------------------------------------------------------------

def gaussian_packet(grid: Grid, center, width, momentum, device=None) -> torch.Tensor:
    """Normalized Gaussian wavepackets; arguments broadcast over a batch."""
    x = grid.x(device)
    center, width, momentum = (torch.as_tensor(a, dtype=torch.float64, device=device).unsqueeze(-1)
                               for a in (center, width, momentum))
    psi = torch.exp(-((x - center) ** 2) / (4 * width**2) + 1j * momentum * x)
    return normalize(psi, grid)


def random_states(batch: int, grid: Grid, gen: torch.Generator, max_packets: int = 2) -> torch.Tensor:
    """Superpositions of 1..max_packets Gaussian packets with random complex weights."""
    L = grid.length
    psi = torch.zeros(batch, grid.n, dtype=DTYPE)
    n_packets = torch.randint(1, max_packets + 1, (batch,), generator=gen)
    for p in range(max_packets):
        active = (n_packets > p).to(torch.float64).unsqueeze(-1)
        center = (torch.rand(batch, generator=gen, dtype=torch.float64) - 0.5) * L / 2
        width = 0.5 + torch.rand(batch, generator=gen, dtype=torch.float64)
        momentum = (torch.rand(batch, generator=gen, dtype=torch.float64) - 0.5) * 6
        phase = torch.exp(2j * math.pi * torch.rand(batch, generator=gen, dtype=torch.float64)).unsqueeze(-1)
        psi = psi + active * phase * gaussian_packet(grid, center, width, momentum)
    return normalize(psi, grid)


def potentials_smooth(batch: int, grid: Grid, gen: torch.Generator) -> torch.Tensor:
    """Training family: harmonic trap plus up to 3 smooth Gaussian bumps or dips."""
    x = grid.x()
    L = grid.length
    omega = 0.2 + 0.4 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    V = 0.5 * omega**2 * x**2
    for _ in range(3):
        on = (torch.rand(batch, 1, generator=gen) < 0.7).to(torch.float64)
        amp = (torch.rand(batch, 1, generator=gen, dtype=torch.float64) - 0.5) * 4
        c = (torch.rand(batch, 1, generator=gen, dtype=torch.float64) - 0.5) * L / 2
        w = 0.5 + 1.5 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
        V = V + on * amp * torch.exp(-((x - c) ** 2) / (2 * w**2))
    return V


def potentials_sharp(batch: int, grid: Grid, gen: torch.Generator) -> torch.Tensor:
    """Out-of-distribution family: double wells and near-square barriers.

    These have sharp edges and shapes never seen in training, so they test whether
    a model learned the *rule* (V acts locally as a phase) or memorized shapes.
    """
    x = grid.x()
    L = grid.length
    kind = torch.randint(0, 2, (batch, 1), generator=gen)
    # double well a (x² - b²)², scaled so the walls stay within ~10 energy units in the box
    b = 1.5 + 2.0 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    a = 0.02 + 0.03 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    double_well = torch.clamp(a * (x**2 - b**2) ** 2, max=10.0)
    # smoothed square barrier on top of a weak trap
    height = 1.0 + 3.0 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    c = (torch.rand(batch, 1, generator=gen, dtype=torch.float64) - 0.5) * L / 3
    half_w = 0.3 + 0.7 * torch.rand(batch, 1, generator=gen, dtype=torch.float64)
    edge = 0.1
    barrier = height * 0.5 * (torch.tanh((x - c + half_w) / edge) - torch.tanh((x - c - half_w) / edge))
    barrier = barrier + 0.5 * 0.3**2 * x**2
    return torch.where(kind == 0, double_well, barrier)
