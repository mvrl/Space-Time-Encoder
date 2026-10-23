"""Space-Time Encoder (Mickisch et al., 2025) with LayerNorm, trained on CHELSA."""
import math
from pathlib import Path

import torch
import torch.nn as nn

from .spherical_harmonics import SphericalHarmonics

LEGENDRE_POLYS = 40          # -> 40*40 = 1600 spherical-harmonic features
N_FREQS = 60                 # -> 2*60 = 120 Fourier time features
N_VARS = 11
DIM_HIDDEN = 256
N_BLOCKS = 4                 # -> ~0.97M params
WEIGHTS = Path(__file__).parent / "weights.safetensors"


class FourierTime(nn.Module):
    """sin/cos of k * 2 pi month / 12, k = 1..n_freqs."""

    def __init__(self, n_freqs: int = N_FREQS):
        super().__init__()
        self.n_freqs = n_freqs
        k = torch.arange(1, n_freqs + 1, dtype=torch.float32)
        self.register_buffer("k", k)

    def forward(self, month):
        tau = 2 * math.pi * month.float() / 12.0          # (N,)
        ang = tau[:, None] * self.k[None, :]               # (N, n_freqs)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (N, 2*n_freqs)


class ResLayerLN(nn.Module):
    """Residual block with LayerNorm after the skip connection."""

    def __init__(self, linear_size):
        super().__init__()
        self.nonlin1 = nn.ReLU(inplace=True)
        self.nonlin2 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.w1 = nn.Linear(linear_size, linear_size)
        self.w2 = nn.Linear(linear_size, linear_size)
        self.norm = nn.LayerNorm(linear_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.nonlin1(y)
        y = self.dropout1(y)
        y = self.w2(y)
        y = self.nonlin2(y)
        return self.norm(x + y)


class NormedFCNet(nn.Module):
    """Linear -> ReLU -> LayerNorm -> n_blocks x ResLayerLN -> class_emb (no bias)."""

    def __init__(self, num_inputs, num_classes, dim_hidden, n_blocks):
        super().__init__()
        self.class_emb = nn.Linear(dim_hidden, num_classes, bias=False)
        layers = [nn.Linear(num_inputs, dim_hidden), nn.ReLU(inplace=True), nn.LayerNorm(dim_hidden)]
        layers += [ResLayerLN(dim_hidden) for _ in range(n_blocks)]
        self.feats = nn.Sequential(*layers)

    def forward(self, x):
        return self.class_emb(self.feats(x))


def orthonormality_penalty(A: torch.Tensor) -> torch.Tensor:
    """||A^T A / B - I||_F^2 for a batch of embeddings A (B, K)."""
    b = A.shape[0]
    gram = (A.T @ A) / b
    eye = torch.eye(gram.shape[0], device=A.device, dtype=A.dtype)
    return ((gram - eye) ** 2).sum()


class SpaceTimeEncoder(nn.Module):
    """latlon (N, 2) in degrees, month (N,) in 1..12. encode -> embedding, forward -> 11 normalized variables."""

    def __init__(self, dim_hidden: int = DIM_HIDDEN, n_blocks: int = N_BLOCKS,
                 legendre_polys: int = LEGENDRE_POLYS, n_freqs: int = N_FREQS, n_vars: int = N_VARS):
        super().__init__()
        self.sh = SphericalHarmonics(legendre_polys=legendre_polys)
        self.time_enc = FourierTime(n_freqs=n_freqs)
        n_inputs = legendre_polys * legendre_polys + 2 * n_freqs
        self.net = NormedFCNet(num_inputs=n_inputs, num_classes=n_vars, dim_hidden=dim_hidden,
                               n_blocks=n_blocks)

    def _features(self, latlon, month):
        # float64: the degree-39 formulas overflow float32
        lonlat = latlon[:, [1, 0]].double()                 # SphericalHarmonics takes (lon, lat)
        sh = self.sh(lonlat).float()                        # (N, 1600)
        t = self.time_enc(month)                            # (N, 120)
        return torch.cat([sh, t], dim=-1)                   # (N, 1720)

    def encode(self, latlon, month):
        return self.net.feats(self._features(latlon, month))

    def forward(self, latlon, month):
        return self.net(self._features(latlon, month))

    def forward_with_embedding(self, latlon, month):
        """(embedding, prediction)."""
        feats = self.net.feats(self._features(latlon, month))
        return feats, self.net.class_emb(feats)


def from_pretrained(device: str = "cpu", path: str | None = None) -> SpaceTimeEncoder:
    """Released weights, or a checkpoint at `path` (.safetensors or .pt), in eval mode."""
    path = str(path or WEIGHTS)
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file
        sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    m = SpaceTimeEncoder()
    m.load_state_dict(sd, strict=True)
    return m.to(device).eval()
