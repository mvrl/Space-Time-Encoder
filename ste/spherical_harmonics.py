"""Spherical-harmonic encoding, adapted from SatCLIP (MIT, Copyright (c) Microsoft Corporation)."""
import torch
from torch import nn
from .spherical_harmonics_ylm import SH as SH_analytic


class SphericalHarmonics(nn.Module):
    def __init__(self, legendre_polys: int = 10):
        """legendre_polys: number of degrees, at most 40."""
        super(SphericalHarmonics, self).__init__()
        self.L, self.M = int(legendre_polys), int(legendre_polys)
        self.embedding_dim = self.L * self.M
        self.SH = SH_analytic

    def forward(self, lonlat):
        lon, lat = lonlat[:, 0], lonlat[:, 1]

        # convert degree to rad
        phi = torch.deg2rad(lon + 180)
        theta = torch.deg2rad(lat + 90)

        Y = []
        for l in range(self.L):
            for m in range(-l, l + 1):
                y = self.SH(m, l, phi, theta)
                if isinstance(y, float):
                    y = y * torch.ones_like(phi)
                Y.append(y)

        return torch.stack(Y,dim=-1)
