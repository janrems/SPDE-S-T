"""Spectral network: a learned function that is polynomial in chosen inputs.

The output is LINEAR in Chebyshev features of the "spectral" inputs, so it is a
genuine polynomial in those inputs and its derivatives there are exact and
stable (no autodiff blow-up through deep nonlinearities). This is the property
the method needs: in M1 the spectral inputs are the noise modes a, and chaos
coefficients are read off as derivatives at a = 0. On M0 we make x the spectral
input to validate the layer's derivative behavior.

The leading `dim_spectral` inputs are treated as spectral (Chebyshev-expanded);
any remaining inputs (e.g. time) are passed through linearly. Multivariate
spectral input currently uses per-coordinate features (no cross terms); M1 will
need tensor-product features for mixed multi-indices -- see TODO.
"""

import torch
import torch.nn as nn


def chebyshev_features(s, degree):
    """Chebyshev polynomials T_0..T_degree of each coordinate.

    s: [bs, k] -> [bs, k*(degree+1)] via T_0=1, T_1=s, T_{n+1}=2 s T_n - T_{n-1}.
    """
    terms = [torch.ones_like(s), s]
    for _ in range(2, degree + 1):
        terms.append(2 * s * terms[-1] - terms[-2])
    stacked = torch.stack(terms, dim=-1)  # [bs, k, degree+1]
    return stacked.reshape(s.size(0), -1)


class SpectralNet(nn.Module):
    def __init__(self, dim_in, dim_out, dim_spectral, degree=4, input_scale=1.0):
        super().__init__()
        if dim_spectral > dim_in:
            raise ValueError("dim_spectral cannot exceed dim_in")
        self.dim_spectral = dim_spectral
        self.degree = degree
        # Spectral inputs are divided by input_scale before the Chebyshev
        # recurrence. Default 1.0 suits the solver's ~unit-std standardized
        # inputs; set it to the half-range of a bounded spectral variable (e.g.
        # the parameter a near 0 in M1) to keep most mass in [-1,1]. A fixed
        # linear rescale, so the output stays polynomial in the input.
        self.input_scale = input_scale
        n_feat = dim_spectral * (degree + 1) + (dim_in - dim_spectral)
        self.linear = nn.Linear(n_feat, dim_out)

    def forward(self, x):
        s = x[:, : self.dim_spectral] / self.input_scale
        rest = x[:, self.dim_spectral :]
        phi = chebyshev_features(s, self.degree)
        return self.linear(torch.cat([phi, rest], dim=1))


def spectral_factory(dim_spectral, degree=4, input_scale=1.0):
    """Net factory (dim_in, dim_out) -> SpectralNet, spectral in the leading inputs."""

    def factory(dim_in, dim_out):
        return SpectralNet(
            dim_in, dim_out, dim_spectral=dim_spectral, degree=degree, input_scale=input_scale
        )

    return factory
