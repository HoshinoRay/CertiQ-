"""Fixed-speed Dubins car with additive angular-rate disturbance.

State   x = (px, py, psi),  psi in [-pi, pi) (periodic)
Control u in U = [-omega_max, omega_max]
Disturb d in D = [-d_max, d_max]

    px+  = px + dt * v * cos(psi)
    py+  = py + dt * v * sin(psi)
    psi+ = wrap(psi + dt * (u + d))

Safety margin (single obstacle inside a circular workspace):

    g(x) = min( ||p - o||^2 - r_obs^2 ,  R_world^2 - ||p||^2 )

This module provides BOTH the pointwise maps (simulation / training / audit)
and the *sound interval over-approximations* used by the verifier:

  * exact min/max of cos and sin over a heading interval (critical points),
  * exact min/max of g over a position box (closed form, no relaxation),
  * a sound successor-box map  X x U x D -> boxes  with wrap-splitting.

The interval routines are part of the trusted computing base of the
certificate and are covered by randomized soundness tests.
"""
from __future__ import annotations

import numpy as np

from ..config import DubinsConfig

TWO_PI = 2.0 * np.pi


def wrap_angle(psi: np.ndarray | float) -> np.ndarray | float:
    """Wrap to [-pi, pi)."""
    return (np.asarray(psi) + np.pi) % TWO_PI - np.pi


# --------------------------------------------------------------------------- #
# Pointwise dynamics and safety margin
# --------------------------------------------------------------------------- #
class DubinsModel:
    def __init__(self, cfg: DubinsConfig):
        self.cfg = cfg
        self.ox, self.oy = cfg.obs_center

    # x: (..., 3), u: (...,), d: (...,)  -> (..., 3)
    def step(self, x: np.ndarray, u: np.ndarray, d: np.ndarray) -> np.ndarray:
        c = self.cfg
        px, py, psi = x[..., 0], x[..., 1], x[..., 2]
        out = np.empty(np.broadcast(px, u, d).shape + (3,), dtype=x.dtype)
        out[..., 0] = px + c.dt * c.v * np.cos(psi)
        out[..., 1] = py + c.dt * c.v * np.sin(psi)
        out[..., 2] = wrap_angle(psi + c.dt * (u + d))
        return out

    def g(self, x: np.ndarray) -> np.ndarray:
        """Safety margin; g(x) >= 0 <=> x in K."""
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
        """Yaw rate psi_dot; the heading update is wrap(psi + dt*heading_rate).

        Generic plant hook used by the grid oracle (Dubins: additive u + d).
        """
        return u + d


# --------------------------------------------------------------------------- #
# Interval arithmetic (verifier side, float64)
# --------------------------------------------------------------------------- #
def cos_interval(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact [min, max] of cos over [lo, hi] (vectorized, hi >= lo).

    Width >= 2*pi  -> [-1, 1].  Otherwise the extrema are attained either at
    the endpoints or at the interior critical points 2k*pi (max) and
    (2k+1)*pi (min).
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    clo, chi = np.cos(lo), np.cos(hi)
    mn = np.minimum(clo, chi)
    mx = np.maximum(clo, chi)
    # interior maximum at 2k*pi: exists iff ceil(lo/2pi) <= floor(hi/2pi)
    has_max = np.ceil(lo / TWO_PI) <= np.floor(hi / TWO_PI)
    # interior minimum at (2k+1)*pi: shift by pi
    has_min = np.ceil((lo - np.pi) / TWO_PI) <= np.floor((hi - np.pi) / TWO_PI)
    mx = np.where(has_max, 1.0, mx)
    mn = np.where(has_min, -1.0, mn)
    full = (hi - lo) >= TWO_PI
    return np.where(full, -1.0, mn), np.where(full, 1.0, mx)


def sin_interval(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Exact [min, max] of sin over [lo, hi]:  sin(t) = cos(t - pi/2)."""
    return cos_interval(np.asarray(lo) - np.pi / 2.0, np.asarray(hi) - np.pi / 2.0)


def g_bounds_on_box(cfg: DubinsConfig,
                    px_lo, px_hi, py_lo, py_hi) -> tuple[np.ndarray, np.ndarray]:
    """Sound [lo, hi] enclosure of g = min(g_obs, g_world) over a position box.

    Each quadratic's per-axis extrema are attained at the box boundary or at the
    projection of the circle centre, so g_obs and g_world are each enclosed
    EXACTLY.  Because g = min(g_obs, g_world):

      * the LOWER bound is exact:  min_box min(a,b) = min(min_box a, min_box b);
      * the UPPER bound is a SOUND over-approximation, not necessarily tight:
        max_box min(a,b) <= min(max_box a, max_box b)  (equality only when the
        two maximisers coincide).

    C1 uses only the (exact) lower bound, so this is sufficient; the upper bound
    is sound, which is all any caller may rely on.
    """
    ox, oy = cfg.obs_center
    px_lo, px_hi = np.asarray(px_lo, float), np.asarray(px_hi, float)
    py_lo, py_hi = np.asarray(py_lo, float), np.asarray(py_hi, float)

    def axis_min_sq(lo, hi, c):
        inside = (lo <= c) & (c <= hi)
        m = np.minimum(np.abs(lo - c), np.abs(hi - c))
        return np.where(inside, 0.0, m) ** 2

    def axis_max_sq(lo, hi, c):
        return np.maximum(np.abs(lo - c), np.abs(hi - c)) ** 2

    # obstacle clearance: ||p - o||^2 - r^2
    obs_min = axis_min_sq(px_lo, px_hi, ox) + axis_min_sq(py_lo, py_hi, oy) - cfg.obs_radius ** 2
    obs_max = axis_max_sq(px_lo, px_hi, ox) + axis_max_sq(py_lo, py_hi, oy) - cfg.obs_radius ** 2
    # workspace: R^2 - ||p||^2
    wld_min = cfg.world_radius ** 2 - (axis_max_sq(px_lo, px_hi, 0.0) + axis_max_sq(py_lo, py_hi, 0.0))
    wld_max = cfg.world_radius ** 2 - (axis_min_sq(px_lo, px_hi, 0.0) + axis_min_sq(py_lo, py_hi, 0.0))
    return np.minimum(obs_min, wld_min), np.minimum(obs_max, wld_max)


def successor_boxes(cfg: DubinsConfig,
                    px_lo, px_hi, py_lo, py_hi, psi_lo, psi_hi,
                    u_lo: float, u_hi: float,
                    d_lo: float, d_hi: float):
    """Sound over-approximation of {f(x,u,d) : x in X, u in U, d in D}.

    Vectorized over N state boxes (u/d bounds are scalars per call).

    Returns
    -------
    box1 : (N, 6) array  [px_lo, px_hi, py_lo, py_hi, psi_lo, psi_hi]
        primary successor box, psi already wrapped to [-pi, pi)
    box2 : (N, 6) array
        secondary box for headings that wrap across the periodic boundary
    mask2 : (N,) bool
        whether box2 is active for each input box

    The raw heading interval  psi + dt*(U + D)  is wrapped and, when it
    crosses the periodic boundary, *split into two boxes* rather than merged
    into one wide interval (dev guide Sec. 8.5 / 16.4).  If the raw interval
    covers >= 2*pi the primary box degenerates to the full circle.
    """
    px_lo = np.asarray(px_lo, float); px_hi = np.asarray(px_hi, float)
    py_lo = np.asarray(py_lo, float); py_hi = np.asarray(py_hi, float)
    psi_lo = np.asarray(psi_lo, float); psi_hi = np.asarray(psi_hi, float)

    cmin, cmax = cos_interval(psi_lo, psi_hi)
    smin, smax = sin_interval(psi_lo, psi_hi)
    k = cfg.dt * cfg.v
    npx_lo = px_lo + k * cmin
    npx_hi = px_hi + k * cmax
    npy_lo = py_lo + k * smin
    npy_hi = py_hi + k * smax

    a = psi_lo + cfg.dt * (u_lo + d_lo)
    b = psi_hi + cfg.dt * (u_hi + d_hi)
    full = (b - a) >= TWO_PI

    # wrap a into [-pi, pi); carry b with the same shift
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
