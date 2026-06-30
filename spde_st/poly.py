"""Polynomial-feature helpers shared by the spectral and chaos nets.

Chebyshev features in one variable, multi-indices for total-degree-bounded
tensor bases, and the tensor-product Chebyshev features in several variables.
"""

import torch


def chebyshev_features(s, degree):
    """Chebyshev polynomials T_0..T_degree of each coordinate.

    s: [bs, k] -> [bs, k*(degree+1)] via T_0=1, T_1=s, T_{n+1}=2 s T_n - T_{n-1}.
    """
    terms = [torch.ones_like(s), s]
    for _ in range(2, degree + 1):
        terms.append(2 * s * terms[-1] - terms[-2])
    stacked = torch.stack(terms, dim=-1)  # [bs, k, degree+1]
    return stacked.reshape(s.size(0), -1)


def multi_indices(n_vars, max_degree):
    """All multi-indices beta in N_0^n_vars with sum(beta) <= max_degree.

    Count is C(n_vars + max_degree, max_degree).
    """
    out = []

    def rec(prefix, remaining, budget):
        if remaining == 0:
            out.append(tuple(prefix))
            return
        for k in range(budget + 1):
            rec(prefix + [k], remaining - 1, budget - k)

    rec([], n_vars, max_degree)
    return out


def tensor_chebyshev(a, betas, degree, scale=1.0):
    """Tensor-product Chebyshev features.

    a: [bs, N]; betas: list of length-N multi-indices. Returns [bs, len(betas)]
    with column for beta equal to prod_i T_{beta_i}(a_i / scale). Cross terms
    (mixed beta) are included, which is what lets the net represent -- and the
    recovery test meaningfully check -- mixed high-order derivatives.
    """
    s = a / scale
    bs, n = s.shape
    table = [chebyshev_features(s[:, i : i + 1], degree).reshape(bs, degree + 1) for i in range(n)]
    feats = []
    for beta in betas:
        prod = torch.ones(bs, device=a.device)
        for i, k in enumerate(beta):
            prod = prod * table[i][:, k]
        feats.append(prod)
    return torch.stack(feats, dim=1)
