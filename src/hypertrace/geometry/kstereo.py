"""κ-stereographic model: one closed-form geometry for every constant curvature κ.

    κ < 0   Poincaré ball of radius 1/sqrt(-κ)       (κ = -c reproduces geometry/poincare.py)
    κ = 0   Euclidean space                           (all formulas reduce to the flat ones)
    κ > 0   stereographic projection of the sphere of radius 1/sqrt(κ): the origin is one
            pole, the circle ||x|| = 1/sqrt(κ) is the equator, the antipode is at infinity

Formulas (Bachmann, Bécigneul & Ganea, ICML 2020; Skopek et al., ICLR 2020; geoopt's
``Stereographic``), with tan_κ(u) = tan(sqrt(κ) u)/sqrt(κ) for κ > 0, tanh(sqrt(-κ) u)/sqrt(-κ)
for κ < 0 and u at κ = 0:

    x ⊕ y     = ((1 - 2κ<x,y> - κ|y|²) x + (1 + κ|x|²) y) / (1 - 2κ<x,y> + κ²|x|²|y|²)
    d(x, y)   = 2 tan_κ⁻¹( |(-x) ⊕ y| )
    exp_0(u)  = tan_κ(|u|) u / |u|,    log_0(x) = tan_κ⁻¹(|x|) x / |x|
    λ_x       = 2 / (1 + κ|x|²)                      (conformal factor)
    exp_x(u)  = x ⊕ tan_κ(λ_x |u| / 2) u / |u|
    P_{x→y}v  = gyr[y, -x] v · λ_x / λ_y              (parallel transport)

Near κ = 0 the trigonometric forms are replaced by their Taylor series (to |κ|² terms), so a
learned curvature can pass through 0 with finite values and gradients. The branch is chosen
from the current value of κ (a scalar), so each call uses one exact formula.

Entailment cones use the conformal generalisation of Ganea et al. (2018):
half-aperture ψ(x) = asin(K (1 + κ|x|²) / |x|), which is the hyperbolic formula for κ = -1 and
the Euclidean one for κ = 0. The exterior angle at x towards y is the angle between x (the
outward radial direction) and log_x(y), whose direction is (-x) ⊕ y; the model is conformal,
so this ambient angle is the Riemannian one. On the sphere (κ > 0) cones are only formal: the
sphere has no nested-cone structure, and product factors on the sphere skip the cone loss by
default (``cones_on_sphere: false``).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

EPS = 1e-15
SMALL_K = 1e-3          # |κ| below this: Taylor series
BALL_EPS = 1e-5         # κ < 0: stay this far (relatively) inside the ball
SPHERE_MAX_ANGLE = 0.95  # κ > 0: keep points within 0.95 π / sqrt(κ) of the origin (antipode at infinity)


def _k(k: Tensor | float, ref: Tensor) -> Tensor:
    return torch.as_tensor(k, dtype=ref.dtype, device=ref.device)


def _kval(k: Tensor | float) -> float:
    return float(k.detach()) if torch.is_tensor(k) else float(k)


def tan_k(u: Tensor, k: Tensor | float) -> Tensor:
    k_t, kv = _k(k, u), _kval(k)
    if abs(kv) < SMALL_K:
        return u + k_t * u**3 / 3 + 2 * k_t**2 * u**5 / 15
    sk = k_t.abs().sqrt()
    return torch.tan(sk * u) / sk if kv > 0 else torch.tanh(sk * u) / sk


def artan_k(u: Tensor, k: Tensor | float) -> Tensor:
    k_t, kv = _k(k, u), _kval(k)
    if abs(kv) < SMALL_K:
        return u - k_t * u**3 / 3 + k_t**2 * u**5 / 5
    sk = k_t.abs().sqrt()
    if kv > 0:
        return torch.atan(sk * u) / sk
    z = (sk * u).clamp(max=1 - 1e-7)
    return 0.5 * (torch.log1p(z) - torch.log1p(-z)) / sk


def _norm(x: Tensor) -> Tensor:
    return x.norm(dim=-1, keepdim=True).clamp_min(EPS)


def project(x: Tensor, k: Tensor | float) -> Tensor:
    """κ < 0: back inside the ball. κ > 0: away from the antipode (at infinity). κ = 0: unchanged."""
    kv = _kval(k)
    if abs(kv) < SMALL_K:
        return x
    n = _norm(x)
    if kv < 0:
        max_norm = (1 - BALL_EPS) / math.sqrt(-kv)
    else:
        max_norm = math.tan(SPHERE_MAX_ANGLE * math.pi / 2) / math.sqrt(kv)
    return torch.where(n > max_norm, x / n * max_norm, x)


def clip_tangent0(u: Tensor, k: Tensor | float) -> Tensor:
    """κ > 0: cap |u| so exp_0 stays short of the antipode (tan blows up at |u| = π / (2 sqrt κ))."""
    kv = _kval(k)
    if kv < SMALL_K:
        return u
    cap = SPHERE_MAX_ANGLE * math.pi / (2 * math.sqrt(kv))
    n = _norm(u)
    return u * torch.clamp(cap / n, max=1.0)


def mobius_add(x: Tensor, y: Tensor, k: Tensor | float) -> Tensor:
    k = _k(k, x)
    x2, y2 = (x * x).sum(-1, keepdim=True), (y * y).sum(-1, keepdim=True)
    xy = (x * y).sum(-1, keepdim=True)
    num = (1 - 2 * k * xy - k * y2) * x + (1 + k * x2) * y
    den = 1 - 2 * k * xy + k**2 * x2 * y2
    return num / den.clamp_min(EPS)  # den = (1 - κ<x,y>)² + κ²(|x|²|y|² - <x,y>²) >= 0 for every κ


def dist(x: Tensor, y: Tensor, k: Tensor | float) -> Tensor:
    return 2 * artan_k(mobius_add(-x, y, k).norm(dim=-1), k)


def pairwise_dist(x: Tensor, y: Tensor, k: Tensor | float) -> Tensor:
    """(N, D) x (M, D) -> (N, M)."""
    return dist(x.unsqueeze(1), y.unsqueeze(0), k)


def dist0(x: Tensor, k: Tensor | float) -> Tensor:
    return 2 * artan_k(x.norm(dim=-1), k)


def expmap0(u: Tensor, k: Tensor | float) -> Tensor:
    u = clip_tangent0(u, k)
    n = _norm(u)
    return project(tan_k(n, k) * u / n, k)


def logmap0(x: Tensor, k: Tensor | float) -> Tensor:
    n = _norm(x)
    return artan_k(n, k) * x / n


def conformal_factor(x: Tensor, k: Tensor | float) -> Tensor:
    k = _k(k, x)
    return 2 / (1 + k * (x * x).sum(-1, keepdim=True)).clamp_min(EPS)


def egrad2rgrad(x: Tensor, grad: Tensor, k: Tensor | float) -> Tensor:
    return grad / conformal_factor(x, k).pow(2)


def expmap(x: Tensor, u: Tensor, k: Tensor | float) -> Tensor:
    n = _norm(u)
    second = tan_k(conformal_factor(x, k) * n / 2, k) * u / n
    return project(mobius_add(x, second, k), k)


def gyration(a: Tensor, b: Tensor, u: Tensor, k: Tensor | float) -> Tensor:
    """gyr[a, b] u in closed form (geoopt's stereographic formula)."""
    k = _k(k, a)
    a2, b2 = (a * a).sum(-1, keepdim=True), (b * b).sum(-1, keepdim=True)
    ab = (a * b).sum(-1, keepdim=True)
    au, bu = (a * u).sum(-1, keepdim=True), (b * u).sum(-1, keepdim=True)
    k2 = k**2
    A = -k2 * au * b2 - k * bu + 2 * k2 * ab * bu
    B = -k2 * bu * a2 + k * au
    D = 1 - 2 * k * ab + k2 * a2 * b2
    return u + 2 * (A * a + B * b) / D.clamp_min(EPS)


def transp(x: Tensor, y: Tensor, v: Tensor, k: Tensor | float) -> Tensor:
    return gyration(y, -x, v, k) * conformal_factor(x, k) / conformal_factor(y, k)


def half_aperture(x: Tensor, k: Tensor | float, cone_k: float = 0.1, min_norm: float = 1e-6) -> Tensor:
    kk = _k(k, x)
    n2 = (x * x).sum(-1)
    n = n2.sqrt().clamp_min(min_norm)
    return torch.asin((cone_k * (1 + kk * n2) / n).clamp(-1 + 1e-6, 1 - 1e-6))


def oxy_angle(x: Tensor, y: Tensor, k: Tensor | float, min_norm: float = 1e-6) -> Tensor:
    """Exterior angle at x towards y: angle between x and the direction of log_x(y), i.e. (-x) ⊕ y."""
    d = mobius_add(-x, y, k)
    cos = (d * x).sum(-1) / (d.norm(dim=-1).clamp_min(min_norm) * x.norm(dim=-1).clamp_min(min_norm))
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))
