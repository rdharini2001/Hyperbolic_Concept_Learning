"""Concept-space geometries behind one interface.

The Poincaré and Euclidean spaces expose the same methods, so the main
comparison in the proposal (arm B, flat concept space, vs. arm C, Poincaré
concept space) is a one-line config switch. Everything upstream (projector,
losses, metrics) is shared.

Parameterisation: projectors output *tangent vectors* at the origin, and
``to_space`` maps them onto the manifold (``expmap0`` for Poincaré, identity for
Euclidean). Those weights stay Euclidean and plain AdamW is exact for them (as in
MERU). Free concept points, by contrast, are parameters that live on the manifold;
they are updated with Riemannian Adam (``hypertrace/optim.py``), which uses the
``egrad2rgrad`` / ``expmap`` / ``transp`` / ``component_inner`` methods below. For
the Euclidean twin the same code is exactly Adam.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor, nn

from hypertrace.geometry import kstereo as ks
from hypertrace.geometry import poincare as pm


class ConceptGeometry(nn.Module, ABC):
    """Abstract concept-space geometry."""

    name: str

    @abstractmethod
    def to_space(self, v: Tensor) -> Tensor:
        """Map tangent vectors at the origin to points in the space."""

    @abstractmethod
    def to_tangent(self, x: Tensor) -> Tensor:
        """Inverse of ``to_space``: points -> tangent vectors at the origin (log_0)."""

    @abstractmethod
    def pairwise_dist(self, x: Tensor, y: Tensor) -> Tensor:
        """(N, D) x (M, D) -> (N, M) distances."""

    @abstractmethod
    def depth(self, x: Tensor) -> Tensor:
        """Distance from the origin; larger means more specific."""

    @abstractmethod
    def half_aperture(self, x: Tensor) -> Tensor:
        """Half-aperture of the entailment cone at ``x``."""

    @abstractmethod
    def oxy_angle(self, x: Tensor, y: Tensor) -> Tensor:
        """Exterior angle at ``x`` towards ``y`` (elementwise over leading dims)."""

    # --- Riemannian optimisation of parameters that live in the space
    @abstractmethod
    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        """Euclidean gradient -> Riemannian gradient at x."""

    @abstractmethod
    def component_inner(self, x: Tensor, u: Tensor) -> Tensor:
        """Per-coordinate contribution to <u, u>_x (shape of u). Summing over the last dim gives
        the Riemannian squared norm. Riemannian Adam uses either (see hypertrace/optim.py)."""

    @abstractmethod
    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        """Move from x along tangent vector u (exact exponential map)."""

    @abstractmethod
    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        """Parallel transport of v from x to y."""

    def entailment_violation(self, parent: Tensor, child: Tensor) -> Tensor:
        """Zero when ``child`` sits inside the cone of ``parent``; positive angle otherwise."""
        return (self.oxy_angle(parent, child) - self.half_aperture(parent)).clamp_min(0)


class PoincareGeometry(ConceptGeometry):
    """Poincaré ball with curvature ``-c``; ``c`` can be fixed or learned."""

    name = "poincare"

    def __init__(
        self,
        curvature: float = 1.0,
        learn_curvature: bool = False,
        curvature_bounds: tuple[float, float] = (0.1, 10.0),
        cone_k: float = 0.1,
    ) -> None:
        super().__init__()
        if curvature <= 0:
            raise ValueError("curvature must be > 0 (the ball has curvature -c)")
        log_c = torch.tensor(math.log(curvature))
        if learn_curvature:
            self.log_c = nn.Parameter(log_c)
        else:
            self.register_buffer("log_c", log_c)
        self.bounds = (math.log(curvature_bounds[0]), math.log(curvature_bounds[1]))
        self.cone_k = cone_k

    @property
    def c(self) -> Tensor:
        return self.log_c.clamp(*self.bounds).exp()

    def to_space(self, v: Tensor) -> Tensor:
        return pm.expmap0(v, self.c)

    def to_tangent(self, x: Tensor) -> Tensor:
        return pm.logmap0(x, self.c)

    def pairwise_dist(self, x: Tensor, y: Tensor) -> Tensor:
        return pm.pairwise_dist(x, y, self.c)

    def depth(self, x: Tensor) -> Tensor:
        return pm.dist0(x, self.c)

    def half_aperture(self, x: Tensor) -> Tensor:
        return pm.half_aperture(x, self.c, self.cone_k)

    def oxy_angle(self, x: Tensor, y: Tensor) -> Tensor:
        return pm.oxy_angle(x, y, self.c)

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return pm.egrad2rgrad(x, grad, self.c)

    def component_inner(self, x: Tensor, u: Tensor) -> Tensor:
        return pm.conformal_factor(x, self.c).pow(2) * u * u

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return pm.expmap(x, u, self.c)

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        return pm.transp(x, y, v, self.c)


class EuclideanGeometry(ConceptGeometry):
    """Flat baseline. Uses Euclidean entailment cones (Ganea et al., 2018, Sec. 3)
    so the hierarchy loss has the same form in both arms."""

    name = "euclidean"

    def __init__(self, cone_k: float = 0.1, min_norm: float = 1e-6) -> None:
        super().__init__()
        self.cone_k = cone_k
        self.min_norm = min_norm

    def to_space(self, v: Tensor) -> Tensor:
        return v

    def to_tangent(self, x: Tensor) -> Tensor:
        return x

    def pairwise_dist(self, x: Tensor, y: Tensor) -> Tensor:
        return torch.cdist(x, y)

    def depth(self, x: Tensor) -> Tensor:
        return x.norm(dim=-1)

    def half_aperture(self, x: Tensor) -> Tensor:
        n = x.norm(dim=-1).clamp_min(self.min_norm)
        return torch.asin((self.cone_k / n).clamp(-1 + 1e-6, 1 - 1e-6))

    def oxy_angle(self, x: Tensor, y: Tensor) -> Tensor:
        d = y - x
        cos = (d * x).sum(-1) / (
            d.norm(dim=-1).clamp_min(self.min_norm) * x.norm(dim=-1).clamp_min(self.min_norm)
        )
        return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6))

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return grad

    def component_inner(self, x: Tensor, u: Tensor) -> Tensor:
        return u * u

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return x + u

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        return v


class KStereographicGeometry(ConceptGeometry):
    """Constant curvature κ of either sign (geometry/kstereo.py): κ < 0 ball, κ = 0 flat, κ > 0 sphere.

    ``name`` is ``sphere`` (fixed κ > 0, default +1) or ``kstereo`` (typically a learned κ starting at 0,
    clamped to ``curvature_bounds``; it can change sign during training).
    """

    def __init__(self, curvature: float = 0.0, learn_curvature: bool = False,
                 curvature_bounds: tuple[float, float] = (-4.0, 4.0), cone_k: float = 0.1,
                 name: str = "kstereo") -> None:
        super().__init__()
        self.name = name
        kappa = torch.tensor(float(curvature))
        if learn_curvature:
            self.kappa_raw = nn.Parameter(kappa)
        else:
            self.register_buffer("kappa_raw", kappa)
        self.bounds = (float(curvature_bounds[0]), float(curvature_bounds[1]))
        self.cone_k = cone_k

    @property
    def k(self) -> Tensor:
        return self.kappa_raw.clamp(*self.bounds)

    def to_space(self, v: Tensor) -> Tensor:
        return ks.expmap0(v, self.k)

    def to_tangent(self, x: Tensor) -> Tensor:
        return ks.logmap0(x, self.k)

    def pairwise_dist(self, x: Tensor, y: Tensor) -> Tensor:
        return ks.pairwise_dist(x, y, self.k)

    def depth(self, x: Tensor) -> Tensor:
        return ks.dist0(x, self.k)

    def half_aperture(self, x: Tensor) -> Tensor:
        return ks.half_aperture(x, self.k, self.cone_k)

    def oxy_angle(self, x: Tensor, y: Tensor) -> Tensor:
        return ks.oxy_angle(x, y, self.k)

    def egrad2rgrad(self, x: Tensor, grad: Tensor) -> Tensor:
        return ks.egrad2rgrad(x, grad, self.k)

    def component_inner(self, x: Tensor, u: Tensor) -> Tensor:
        return ks.conformal_factor(x, self.k).pow(2) * u * u

    def expmap(self, x: Tensor, u: Tensor) -> Tensor:
        return ks.expmap(x, u, self.k)

    def transp(self, x: Tensor, y: Tensor, v: Tensor) -> Tensor:
        return ks.transp(x, y, v, self.k)


def curvature_of(geometry: ConceptGeometry) -> float:
    """Sectional curvature as one signed number: -c (Poincaré), 0 (Euclidean), κ (κ-stereographic)."""
    if isinstance(geometry, PoincareGeometry):
        return -float(geometry.c)
    if isinstance(geometry, KStereographicGeometry):
        return float(geometry.k.detach())
    return 0.0


def build_geometry(name: str, **kwargs) -> ConceptGeometry:
    """poincare | euclidean | sphere (κ-stereographic, fixed κ > 0) | kstereo (κ-stereographic, any κ)."""
    if name in ("sphere", "kstereo"):
        geom = KStereographicGeometry(name=name, **kwargs)
        if name == "sphere" and float(geom.k) <= 0:
            raise ValueError("geometry 'sphere' needs curvature > 0 (use kstereo for other signs)")
        return geom
    registry = {"poincare": PoincareGeometry, "euclidean": EuclideanGeometry}
    if name not in registry:
        raise ValueError(f"Unknown geometry '{name}'. Options: {sorted(registry) + ['sphere', 'kstereo', 'product']}")
    return registry[name](**kwargs)
