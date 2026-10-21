"""Candidate learners for one step of quantum time evolution.

Every model maps (ψ_t complex (B, n), V real (B, n)) -> ψ_{t+Δt} complex (B, n).
They span a spectrum from "no physics assumed" to "the equation's form is built in":

    linear      one learned complex matrix; cannot see V (control: is V needed?)
    mlp         dense network on the whole grid; no locality, no symmetry
    cnn         local convolutions with periodic padding; locality + translation symmetry
    fno         Fourier Neural Operator; global mixing in frequency space
    splitstep   learns a dispersion ω(k) and a pointwise potential response g(V), then
                applies them as phases. Exactly unitary by construction. It knows the
                *form* of the Schrödinger equation but not the kinetic law or units.
"""
import torch
from torch import nn

from .data import V_SCALE, to_channels


def _complex_out(x: torch.Tensor) -> torch.Tensor:
    """(B, 2, n) real -> (B, n) complex."""
    return torch.complex(x[:, 0], x[:, 1])


class LinearNoV(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.w_re = nn.Parameter(torch.eye(n))
        self.w_im = nn.Parameter(torch.zeros(n, n))

    def forward(self, psi, V):
        return psi @ torch.complex(self.w_re, self.w_im).T


class MLP(nn.Module):
    def __init__(self, n: int, hidden: int = 512, depth: int = 3):
        super().__init__()
        layers, d = [], 3 * n
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.GELU()]
            d = hidden
        layers.append(nn.Linear(d, 2 * n))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.n = n

    def forward(self, psi, V):
        delta = self.net(to_channels(psi, V).flatten(1)).view(-1, 2, self.n)
        return psi + _complex_out(delta)


class CNN(nn.Module):
    def __init__(self, width: int = 64, depth: int = 8, kernel: int = 5):
        super().__init__()
        conv = lambda i, o: nn.Conv1d(i, o, kernel, padding=kernel // 2, padding_mode="circular")
        self.inp = conv(3, width)
        self.blocks = nn.ModuleList(nn.Sequential(nn.GELU(), conv(width, width)) for _ in range(depth))
        self.out = nn.Conv1d(width, 2, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, psi, V):
        h = self.inp(to_channels(psi, V))
        for block in self.blocks:
            h = h + block(h)
        return psi + _complex_out(self.out(torch.nn.functional.gelu(h)))


class SpectralConv1d(nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        scale = 1 / width
        self.modes = modes
        self.w = nn.Parameter(scale * torch.randn(width, width, modes, dtype=torch.cfloat))

    def forward(self, x):
        x_ft = torch.fft.rfft(x, dim=-1)
        out = torch.zeros_like(x_ft)
        out[..., : self.modes] = torch.einsum("bim,iom->bom", x_ft[..., : self.modes], self.w)
        return torch.fft.irfft(out, n=x.shape[-1], dim=-1)


class FNO(nn.Module):
    def __init__(self, width: int = 32, modes: int = 32, depth: int = 4):
        super().__init__()
        self.lift = nn.Conv1d(3, width, 1)
        self.spectral = nn.ModuleList(SpectralConv1d(width, modes) for _ in range(depth))
        self.pointwise = nn.ModuleList(nn.Conv1d(width, width, 1) for _ in range(depth))
        self.proj = nn.Sequential(nn.Conv1d(width, 128, 1), nn.GELU(), nn.Conv1d(128, 2, 1))
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(self, psi, V):
        h = self.lift(to_channels(psi, V))
        for spec, point in zip(self.spectral, self.pointwise):
            h = torch.nn.functional.gelu(spec(h) + point(h))
        return psi + _complex_out(self.proj(h))


class SplitStep(nn.Module):
    """Learned Strang splitting: ψ' = P(V) · IFFT(D · FFT(P(V) · ψ)), repeated `substeps` times.

    D = exp(-i ω(k) h) with one free ω per Fourier mode; P = exp(-i g(V) h / 2) with g a
    small pointwise MLP. Both are pure phases, so ||ψ'|| == ||ψ|| for any parameters.
    Initialized at the identity map (ω = 0, g = 0).
    """

    def __init__(self, n: int, dt: float, substeps: int = 4, hidden: int = 32):
        super().__init__()
        self.h = dt / substeps
        self.substeps = substeps
        self.omega = nn.Parameter(torch.zeros(n))
        self.g = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, 1))
        nn.init.zeros_(self.g[-1].weight)
        nn.init.zeros_(self.g[-1].bias)

    def forward(self, psi, V):
        phase_v = self.g((V / V_SCALE).unsqueeze(-1)).squeeze(-1)
        half_kick = torch.exp(-0.5j * self.h * phase_v)
        drift = torch.exp(-1j * self.h * self.omega)
        for _ in range(self.substeps):
            psi = half_kick * psi
            psi = torch.fft.ifft(drift * torch.fft.fft(psi, dim=-1), dim=-1)
            psi = half_kick * psi
        return psi


def build(name: str, n: int, dt: float) -> nn.Module:
    return {
        "linear": lambda: LinearNoV(n),
        "mlp": lambda: MLP(n),
        "cnn": lambda: CNN(),
        "fno": lambda: FNO(),
        "splitstep": lambda: SplitStep(n, dt),
    }[name]()


def n_params(model: nn.Module) -> int:
    """Count real parameters (a complex parameter counts as two)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())
