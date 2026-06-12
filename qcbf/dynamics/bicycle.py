"""Fixed-speed kinematic bicycle (F1TENTH) with steering disturbance.

This is the first hardware-facing reuse of the certifier (README §5 / the
"port to a new plant" path). The plant differs from the Dubins car in exactly
one place -- the heading rate -- and reuses everything else:

State    x = (px, py, psi),  psi in [-pi, pi) periodic
Control  u = delta  (front-wheel steering angle, |delta| <= delta_max)
Disturb  d  (additive steering disturbance, |d| <= d_max)

    px+  = px + dt * v * cos(psi)                         (identical to Dubins)
    py+  = py + dt * v * sin(psi)                         (identical to Dubins)
    psi+ = wrap(psi + dt * (v / L) * tan(delta + d))      (bicycle yaw rate)

Because the position update is unchanged, the oracle's bilinear-xy stencil and
the *entire* verifier (CROWN/IBP, h3 compile, lattice closure, runtime filter,
audit) apply unchanged; only the heading shift and one new interval primitive
(`tan`, monotone on the steering range) are problem-specific.

The interval routines here are part of the trusted computing base and are
covered by randomized soundness tests in ``tests/test_bicycle.py`` (mirroring
the Dubins T5-T7).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Reuse the shared, soundness-tested interval primitives from the Dubins module.
from .dubins import (TWO_PI, cos_interval, g_bounds_on_box, sin_interval,
                     wrap_angle)


@dataclass(frozen=True)
class BicycleConfig:
    """Fixed-speed kinematic bicycle; F1TENTH-scale defaults.

    ``delta_max + d_max`` must stay strictly inside (-pi/2, pi/2) so the steering
    interval never reaches the tan singularity (asserted in ``successor_boxes``).
    """
    dt: float = 0.05
    v: float = 2.0
    wheelbase: float = 0.33          # L (F1TENTH ~0.33 m)
    delta_max: float = 0.40          # front-wheel steering limit (rad)
    d_max: float = 0.10              # additive steering disturbance (rad)
    # safety geometry (single obstacle inside a circular workspace)
    obs_center: tuple[float, float] = (0.0, 0.0)
    obs_radius: float = 0.50
    world_radius: float = 2.50
    p_lo: float = -3.0
    p_hi: float = 3.0
    g_fail: float = -1.0

    @property
    def yaw_rate_max(self) -> float:
        """Nominal max yaw rate (v/L)*tan(delta_max) -- the Dubins-equivalent omega."""
        return (self.v / self.wheelbase) * np.tan(self.delta_max)

    @property
    def control_max(self) -> float:
        """Generic control-input bound (here the steering limit delta_max)."""
        return self.delta_max


# --------------------------------------------------------------------------- #
def tan_interval(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact [min, max] of tan over [lo, hi] (vectorized, hi >= lo).

    tan is strictly increasing on (-pi/2, pi/2); callers must guarantee the
    interval lies inside that branch (no pole crossing).
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return np.tan(lo), np.tan(hi)


# --------------------------------------------------------------------------- #
class BicycleModel:
    """Pointwise dynamics and safety margin (simulation / training / audit)."""

    def __init__(self, cfg: BicycleConfig):
        self.cfg = cfg
        self.ox, self.oy = cfg.obs_center

    # x: (..., 3), u=delta: (...,), d: (...,)  -> (..., 3)
    def step(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        c = self.cfg
        px, py, psi = x[..., 0], x[..., 1], x[..., 2]
        out = np.empty(np.broadcast(px, u, d).shape + (3,), dtype=x.dtype)
        out[..., 0] = px + c.dt * c.v * np.cos(psi)
        out[..., 1] = py + c.dt * c.v * np.sin(psi)
        out[..., 2] = wrap_angle(psi + c.dt * (c.v / c.wheelbase)
                                 * np.tan(u + d))
        return out

    def g(self, x: np.ndarray) -> np.ndarray:
        """Safety margin; g(x) >= 0 <=> x in K (obstacle clearance & in-world)."""
        c = self.cfg
        px, py = x[..., 0], x[..., 1]
        d_obs = (px - self.ox) ** 2 + (py - self.oy) ** 2 - c.obs_radius ** 2
        d_wld = c.world_radius ** 2 - (px ** 2 + py ** 2)
        return np.minimum(d_obs, d_wld)

    def in_domain(self, x: np.ndarray) -> np.ndarray:
        c = self.cfg
        return ((x[..., 0] >= c.p_lo) & (x[..., 0] <= c.p_hi)
                & (x[..., 1] >= c.p_lo) & (x[..., 1] <= c.p_hi))

    def heading_rate(self, u, d):
        """Yaw rate (v/L)*tan(delta + d); heading update is wrap(psi+dt*heading_rate)."""
        c = self.cfg
        return (c.v / c.wheelbase) * np.tan(u + d)


# --------------------------------------------------------------------------- #
def successor_boxes(cfg: BicycleConfig,
                    px_lo, px_hi, py_lo, py_hi, psi_lo, psi_hi,
                    u_lo: float, u_hi: float,
                    d_lo: float, d_hi: float):
    """Sound over-approximation of {f(x,u,d) : x in X, u in U, d in D}.

    Vectorized over N state boxes; u (=delta) / d bounds are scalars per call.
    Returns (box1, box2, mask2) exactly as the Dubins primitive: a primary
    successor box, a secondary box for headings that wrap the periodic seam,
    and a per-box activity mask for the secondary.

    Position is identical to Dubins (depends only on psi). The heading shift is
    dt*(v/L)*tan(delta + d); since tan is monotone on the steering branch, its
    interval is [tan(u_lo+d_lo), tan(u_hi+d_hi)].
    """
    px_lo = np.asarray(px_lo, float); px_hi = np.asarray(px_hi, float)
    py_lo = np.asarray(py_lo, float); py_hi = np.asarray(py_hi, float)
    psi_lo = np.asarray(psi_lo, float); psi_hi = np.asarray(psi_hi, float)

    # guard: the steering interval must stay strictly inside one tan branch
    # (u_lo/u_hi may be per-cell arrays on the fallback path, so reduce with all)
    assert np.all(np.abs(u_hi + d_hi) < np.pi / 2 - 1e-9) and \
        np.all(np.abs(u_lo + d_lo) < np.pi / 2 - 1e-9), \
        "steering interval reaches the tan singularity; reduce delta_max + d_max"

    # position image (identical to Dubins)
    cmin, cmax = cos_interval(psi_lo, psi_hi)
    smin, smax = sin_interval(psi_lo, psi_hi)
    k = cfg.dt * cfg.v
    npx_lo = px_lo + k * cmin
    npx_hi = px_hi + k * cmax
    npy_lo = py_lo + k * smin
    npy_hi = py_hi + k * smax

    # heading image: psi + dt*(v/L)*tan([u_lo+d_lo, u_hi+d_hi])
    kpsi = cfg.dt * (cfg.v / cfg.wheelbase)
    tlo, thi = tan_interval(u_lo + d_lo, u_hi + d_hi)
    a = psi_lo + kpsi * tlo
    b = psi_hi + kpsi * thi
    full = (b - a) >= TWO_PI

    shift = np.floor((a + np.pi) / TWO_PI) * TWO_PI
    a2, b2 = a - shift, b - shift            # a2 in [-pi, pi)
    crosses = (b2 > np.pi) & ~full

    p1_lo = np.where(full, -np.pi, a2)
    p1_hi = np.where(full, np.pi, np.where(crosses, np.pi, b2))
    p2_lo = np.full_like(a2, -np.pi)
    p2_hi = np.where(crosses, b2 - TWO_PI, -np.pi)

    box1 = np.stack([npx_lo, npx_hi, npy_lo, npy_hi, p1_lo, p1_hi], axis=-1)
    box2 = np.stack([npx_lo, npx_hi, npy_lo, npy_hi, p2_lo, p2_hi], axis=-1)
    return box1, box2, crosses
