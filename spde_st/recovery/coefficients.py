"""Recover Wiener-chaos coefficients from a learned S-transform.

The chaos coefficient field is the Taylor coefficient of u at a = 0:
    c_alpha(t,x) = (1/alpha!) d_a^alpha u(t,x; 0).
We take the derivatives by autograd. This is net-agnostic: for the chaos net
(polynomial in a) the derivatives are exact and stable; for a plain MLP they are
the noisy autodiff derivatives -- the contrast the method is about.
"""

from math import factorial

import torch


def taylor_coefficient(u_of_a, n_param, alpha, batch_size):
    """(1/alpha!) d_a^alpha u(a=0), per sample. u_of_a: [bs,N] leaf -> [bs] or [bs,1]."""
    a = torch.zeros(batch_size, n_param, requires_grad=True)
    cur = u_of_a(a).reshape(batch_size)
    for var, order in enumerate(alpha):
        for _ in range(order):
            g = torch.autograd.grad(cur.sum(), a, create_graph=True, allow_unused=True)[0]
            # a derivative beyond the polynomial degree is structurally zero -> None
            cur = torch.zeros(batch_size) if g is None else g[:, var]
    denom = 1
    for o in alpha:
        denom *= factorial(o)
    return (cur / denom).detach()


def chaos_coefficients(solver, n, x, max_order):
    """All coefficient fields {alpha: c_alpha(x)} up to total order max_order at step n."""
    from spde_st.poly import multi_indices

    alphas = multi_indices(solver.n_param, max_order)

    def u_of_a(a):
        y, _ = solver._yz(solver.nets[n], x, n, a)
        return y

    return {alpha: taylor_coefficient(u_of_a, solver.n_param, alpha, x.size(0)) for alpha in alphas}
