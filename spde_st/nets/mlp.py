"""Plain feedforward net, one per time step.

The M0 baseline approximator. The spectral net (polynomial in the noise
parameter a, for stable high-order derivatives) replaces this in M1; the
solver takes any net via a factory, so the swap is local.
"""

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_h=32, n_hidden=3):
        super().__init__()
        layers = [nn.Linear(dim_in, dim_h), nn.Tanh()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(dim_h, dim_h), nn.Tanh()]
        layers += [nn.Linear(dim_h, dim_out)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def mlp_factory(dim_h=32, n_hidden=3):
    """Return a net factory (dim_in, dim_out) -> MLP with fixed width/depth."""

    def factory(dim_in, dim_out):
        return MLP(dim_in, dim_out, dim_h=dim_h, n_hidden=n_hidden)

    return factory
