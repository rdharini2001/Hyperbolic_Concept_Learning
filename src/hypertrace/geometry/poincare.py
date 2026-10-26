"""Functional Poincaré-ball operations with curvature ``-c`` (c > 0).

All functions are pure and differentiable. The ball has radius ``1 / sqrt(c)``.
Entailment-cone formulas follow Ganea, Bécigneul & Hofmann (ICML 2018),
"Hyperbolic Entailment Cones for Learning Hierarchical Embeddings"; they are
written for the unit ball, so points are rescaled by ``sqrt(c)`` first.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Numerical guards. float32 artanh/arcosh lose precision very close to the
# boundary, so points are kept at least BOUNDARY_EPS (in unit-ball units) inside.
BOUNDARY_EPS = 1e-5
MIN_NORM = 1e-15


def _sqrt_c(c: Tensor | float, ref: Tensor) -> Tensor:
    return torch.as_tensor(c, dtype=ref.dtype, device=ref.device).clamp_min(MIN_NORM).sqrt()


def _artanh(x: Tensor) -> Tensor:
    x = x.clamp(-1 + BOUNDARY_EPS, 1 - BOUNDARY_EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def project(x: Tensor, c: Tensor | float) -> Tensor:
    """Pull points that drifted outside (or onto) the boundary back inside the ball."""
    sc = _sqrt_c(c, x)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    max_norm = (1 - BOUNDARY_EPS) / sc
    return torch.where(norm > max_norm, x / norm * max_norm, x)


def expmap0(v: Tensor, c: Tensor | float) -> Tensor:
    """Exponential map at the origin: tangent vector -> point on the ball."""
    sc = _sqrt_c(c, v)
    norm = v.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    return project(torch.tanh(sc * norm) * v / (sc * norm), c)


def logmap0(x: Tensor, c: Tensor | float) -> Tensor:
    """Logarithmic map at the origin: point on the ball -> tangent vector."""
    sc = _sqrt_c(c, x)
    norm = x.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    return _artanh(sc * norm) * x / (sc * norm)


def mobius_add(x: Tensor, y: Tensor, c: Tensor | float) -> Tensor:
    c = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    xy = (x * y).sum(-1, keepdim=True)
    x2 = (x * x).sum(-1, keepdim=True)
    y2 = (y * y).sum(-1, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    den = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / den.clamp_min(MIN_NORM)


def dist(x: Tensor, y: Tensor, c: Tensor | float) -> Tensor:
    """Geodesic distance between broadcastable point sets (last dim = features), Mobius-addition form.

    Used as an independent cross-check. The model uses ``pairwise_dist`` (arcosh form), which stays
    accurate closer to the boundary; this form saturates where artanh's argument hits 1 - BOUNDARY_EPS.
    """
    sc = _sqrt_c(c, x)
    return 2.0 / sc * _artanh(sc * mobius_add(-x, y, c).norm(dim=-1))


def pairwise_dist(x: Tensor, y: Tensor, c: Tensor | float) -> Tensor:
    """Distances between every row of ``x`` (N, D) and every row of ``y`` (M, D) -> (N, M).

    Uses the arcosh closed form. ||x - y||^2 is computed from explicit differences
    (not ||x||^2 + ||y||^2 - 2<x, y>): near the boundary the denominator is tiny,
    and the cancellation error of the expanded form gets amplified into a large
    distance error (e.g. d(x, x) ~ 0.15 in float32).
    """
    cc = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    sc = _sqrt_c(c, x)
    x2 = (x * x).sum(-1, keepdim=True)  # (N, 1)
    y2 = (y * y).sum(-1, keepdim=True).T  # (1, M)
    sq = torch.cdist(x, y, compute_mode="donot_use_mm_for_euclid_dist").pow(2)
    denom = ((1 - cc * x2) * (1 - cc * y2)).clamp_min(MIN_NORM)
    arg = 1 + 2 * cc * sq / denom
    # arcosh(z) = log(z + sqrt(z^2 - 1)); clamp keeps the gradient finite at z = 1.
    arg = arg.clamp_min(1 + 1e-7)
    return torch.log(arg + torch.sqrt(arg * arg - 1)) / sc


def dist0(x: Tensor, c: Tensor | float) -> Tensor:
    """Distance from the origin; used as the 'depth' / specificity of a concept."""
    sc = _sqrt_c(c, x)
    return 2.0 / sc * _artanh(sc * x.norm(dim=-1))


def half_aperture(x: Tensor, c: Tensor | float, k: float = 0.1) -> Tensor:
    """Half-aperture psi(x) of the entailment cone rooted at ``x``.

    Points closer to the origin get wider cones, so general concepts can contain
    many specific ones. ``k`` sets the cone width (Ganea et al. use 0.1).
    """
    sc = _sqrt_c(c, x)
    n = (sc * x).norm(dim=-1).clamp_min(MIN_NORM)
    return torch.asin((k * (1 - n * n) / n).clamp(-1 + 1e-6, 1 - 1e-6))


def oxy_angle(x: Tensor, y: Tensor, c: Tensor | float) -> Tensor:
    """Exterior angle Xi(x, y): angle at ``x`` between the geodesic to ``y`` and the ray from the origin.

    ``y`` lies inside the cone of ``x`` iff ``oxy_angle(x, y) <= half_aperture(x)``.
    """
    sc = _sqrt_c(c, x)
    x, y = sc * x, sc * y
    xy = (x * y).sum(-1)
    x2 = (x * x).sum(-1)
    y2 = (y * y).sum(-1)
    num = xy * (1 + x2) - x2 * (1 + y2)
    den = x2.sqrt() * (x - y).norm(dim=-1) * torch.sqrt((1 + x2 * y2 - 2 * xy).clamp_min(MIN_NORM))
    return torch.acos((num / den.clamp_min(MIN_NORM)).clamp(-1 + 1e-6, 1 - 1e-6))


# ---------------------------------------------------------------- Riemannian optimisation
# Used when parameters live ON the ball (free concept points), see hypertrace/optim.py.
# Formulas follow Ganea et al. (NeurIPS 2018) and Bécigneul & Ganea (ICLR 2019); the
# parallel transport via gyration is the one used by geoopt (Kochurov et al., 2020).


def conformal_factor(x: Tensor, c: Tensor | float) -> Tensor:
    """lambda_x = 2 / (1 - c ||x||^2); the metric at x is lambda_x^2 times the Euclidean one."""
    cc = torch.as_tensor(c, dtype=x.dtype, device=x.device)
    return 2.0 / (1 - cc * (x * x).sum(-1, keepdim=True)).clamp_min(MIN_NORM)


def egrad2rgrad(x: Tensor, grad: Tensor, c: Tensor | float) -> Tensor:
    """Riemannian gradient from the Euclidean one: grad / lambda_x^2."""
    return grad / conformal_factor(x, c).pow(2)


def expmap(x: Tensor, u: Tensor, c: Tensor | float) -> Tensor:
    """Exponential map at x: x (+) tanh(sqrt(c) lambda_x ||u|| / 2) u / (sqrt(c) ||u||)."""
    sc = _sqrt_c(c, x)
    un = u.norm(dim=-1, keepdim=True).clamp_min(MIN_NORM)
    second = torch.tanh(sc * conformal_factor(x, c) * un / 2) * u / (sc * un)
    return project(mobius_add(x, second, c), c)


def gyration(u: Tensor, v: Tensor, w: Tensor, c: Tensor | float) -> Tensor:
    """gyr[u, v] w, the rotation that makes Möbius addition associative."""
    cc = torch.as_tensor(c, dtype=u.dtype, device=u.device)
    u2, v2 = (u * u).sum(-1, keepdim=True), (v * v).sum(-1, keepdim=True)
    uv, uw, vw = (u * v).sum(-1, keepdim=True), (u * w).sum(-1, keepdim=True), (v * w).sum(-1, keepdim=True)
    a = -cc**2 * uw * v2 + cc * vw + 2 * cc**2 * uv * vw
    b = -cc**2 * vw * u2 - cc * uw
    d = 1 + 2 * cc * uv + cc**2 * u2 * v2
    return w + 2 * (a * u + b * v) / d.clamp_min(MIN_NORM)


def transp(x: Tensor, y: Tensor, v: Tensor, c: Tensor | float) -> Tensor:
    """Parallel transport of tangent vector v from x to y (preserves the Riemannian norm)."""
    return gyration(y, -x, v, c) * conformal_factor(x, c) / conformal_factor(y, c)
