"""AdamW for ordinary weights + Riemannian Adam for parameters that live in the concept space.

Free concept points are points *on* the Poincaré ball, so a Euclidean gradient step
would leave the manifold and ignore its metric. Riemannian Adam (Bécigneul & Ganea,
ICLR 2019) instead, per point x with Euclidean gradient g:

    r   = g / lambda_x^2                         Riemannian gradient
    m   = b1 * transported m + (1 - b1) * r      first moment (a tangent vector at x)
    v   = b2 * v + (1 - b2) * lambda_x^2 r^2     second moment (per coordinate, or summed per point)
    x'  = exp_x( -lr * m_hat / (sqrt(v_hat) + eps) )
    m  <- P_{x -> x'}(m)                         parallel transport to the new point

with the exact exponential map and parallel transport of the ball
(``geometry/poincare.py``). The step therefore follows geodesics and stays inside
the ball. (geoopt's RiemannianAdam implements the same algorithm with a first-order
retraction x + u instead of exp_x; the tests check that we match it when it is given
the exact exponential map.)

Second moment, ``second_moment``:

``coordinate`` (default) lambda_x^2 r_i^2 per coordinate. The ball's metric is conformal
               (lambda_x^2 times the identity), so the coordinate axes are orthogonal
               directions of the tangent space and a diagonal preconditioner is well
               defined. For the Euclidean twin (rgrad = g, exp_x(u) = x + u, identity
               transport) this is EXACTLY Adam, so both arms run the same algorithm, and a
               step moves a point by about lr * sqrt(D) in Riemannian length in both.
``point``      lambda_x^2 ||r||^2, one scalar per point, as in Bécigneul & Ganea's
               Riemannian Adam for product manifolds and in geoopt. Steps are then about
               lr in Riemannian length, sqrt(D) times shorter than Adam's, so the Poincaré
               arm would also get a smaller effective learning rate than its Euclidean twin.
               On synthetic data this left the concept points almost at their initial
               positions (kept as an option and cross-checked against geoopt in the tests).

Ordinary weights (projector matrices, scales, biases, temperature) are updated with
standard decoupled-weight-decay AdamW, the same formula as ``torch.optim.AdamW``.
Groups whose ``manifold`` flag is set use the Riemannian update with the geometry given
to the constructor (read at every step, so a learned curvature is respected). ``geometry``
may also be a list (product models: one geometry per factor); a group then names its
geometry by ``geometry_index`` (an int, so ``state_dict`` stays plain data).
"""

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer

from hypertrace.geometry import ConceptGeometry


class RiemannianAdamW(Optimizer):
    def __init__(self, params, geometry: ConceptGeometry | list[ConceptGeometry], lr: float = 1e-3,
                 betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 second_moment: str = "coordinate") -> None:
        if second_moment not in ("coordinate", "point"):
            raise ValueError(f"second_moment must be 'coordinate' or 'point', got {second_moment!r}")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, manifold=False,
                                      geometry_index=0))
        # not stored in param_groups, so state_dict stays plain data
        self.geometries = list(geometry) if isinstance(geometry, list | tuple) else [geometry]
        self.per_point = second_moment == "point"

    @property
    def geometry(self) -> ConceptGeometry:
        return self.geometries[0]

    def _inner(self, g, p, r):
        q = g.component_inner(p, r)
        return q.sum(-1, keepdim=True) if self.per_point else q

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, (b1, b2), eps, wd = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                geom = self.geometries[group.get("geometry_index", 0)]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = (torch.zeros_like(self._inner(geom, p, p))
                                           if group["manifold"] else torch.zeros_like(p))
                state["step"] += 1
                t = state["step"]
                m, v = state["exp_avg"], state["exp_avg_sq"]
                bc1, bc2 = 1 - b1**t, 1 - b2**t
                if group["manifold"]:
                    if wd:
                        raise ValueError("weight decay is not defined for points on the manifold")
                    g = geom
                    r = g.egrad2rgrad(p, p.grad)
                    m.mul_(b1).add_(r, alpha=1 - b1)
                    v.mul_(b2).add_(self._inner(g, p, r), alpha=1 - b2)
                    direction = (m / bc1) / ((v / bc2).sqrt() + eps)
                    new = g.expmap(p, -lr * direction)
                    m.copy_(g.transp(p, new, m))
                    p.copy_(new)
                else:
                    if wd:
                        p.mul_(1 - lr * wd)
                    m.mul_(b1).add_(p.grad, alpha=1 - b1)
                    v.mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                    denom = (v.sqrt() / math.sqrt(bc2)).add_(eps)
                    p.addcdiv_(m, denom, value=-lr / bc1)
        return loss
