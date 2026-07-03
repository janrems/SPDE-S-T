"""Chaos net: MLP in (t,x), polynomial (tensor Chebyshev) in the parameter a.

Models u_theta(t,x,a) = sum_beta W_beta(t,x) * Psi_beta(a), where Psi_beta are
tensor-product Chebyshev features of a (total degree <= max_degree) and the
coefficient fields W_beta(t,x) are produced by an MLP. This is exactly the
shape the method needs:
  - polynomial in a  => high-order a-derivatives at 0 are exact and stable, so
    chaos coefficients c_alpha = (1/alpha!) d_a^alpha u(.;0) are recovered cleanly;
  - flexible (MLP) in (t,x) => the coefficient fields can be the non-polynomial
    heat-kernel shapes, which a polynomial in (t,x) would fit poorly.

Input layout matches the solver's features: [standardized x (dim_x), time (1),
parameter a (n_param)]. Outputs (y, z) stacked as dim_out columns; both are
polynomial in a (z is the gradient part the BSDE needs).
"""

import torch.nn as nn

from spde_st.nets.mlp import MLP
from spde_st.poly import multi_indices, tensor_chebyshev, tensor_monomial


class ChaosNet(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_x,
        n_param,
        max_degree=3,
        input_scale=1.0,
        dim_h=32,
        n_hidden=2,
        basis="chebyshev",
    ):
        super().__init__()
        self.dim_x = dim_x
        self.n_param = n_param
        self.max_degree = max_degree
        self.input_scale = input_scale
        self.dim_out = dim_out
        self.basis = basis  # "chebyshev" or "monomial" (monomial => c_beta = W_beta directly)
        self.betas = multi_indices(n_param, max_degree)
        self.n_feat = len(self.betas)
        # MLP maps (t,x) to the coefficient fields W for every output and feature.
        self.trunk = MLP(dim_x + 1, dim_out * self.n_feat, dim_h=dim_h, n_hidden=n_hidden)

    def forward(self, feat):
        xt = feat[:, : self.dim_x + 1]
        a = feat[:, self.dim_x + 1 :]
        W = self.trunk(xt).reshape(-1, self.dim_out, self.n_feat)
        if self.basis == "monomial":
            psi = tensor_monomial(a, self.betas)
        else:
            psi = tensor_chebyshev(a, self.betas, self.max_degree, self.input_scale)
        return (W * psi.unsqueeze(1)).sum(-1)  # [bs, dim_out]


def chaos_factory(
    dim_x, n_param, max_degree=3, input_scale=1.0, dim_h=32, n_hidden=2, basis="chebyshev"
):
    """Net factory (dim_in, dim_out) -> ChaosNet. dim_x and n_param fix the input split."""

    def factory(dim_in, dim_out):
        return ChaosNet(
            dim_in, dim_out, dim_x, n_param, max_degree, input_scale, dim_h, n_hidden, basis
        )

    return factory
