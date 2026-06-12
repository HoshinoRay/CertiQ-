"""Variable-speed F1TENTH bicycle (4-state) with an analytic braking-distance CBF.

This is the plant that *breaks the fixed-speed Gate-D wall* (see
docs/dubins_e0_results_and_ablation.md): because the car can brake to a stop,
the certified set is a thick sub-level set rather than a thin constant-V orbit.

State   x = (px, py, psi, v),  v in [0, v_max]   (v_min = 0 -> can stop)
Control u = (a, delta)   a in [a_min, a_max] (a_min < 0 = brake), |delta|<=delta_max
Disturb d = (d_a, d_delta)   bounded actuator error on accel / steer

    px+  = px + dt v cos(psi)
    py+  = py + dt v sin(psi)
    psi+ = wrap(psi + dt (v/L) tan(delta + d_delta))
    v+   = clip(v + dt (a + d_a), 0, v_max)

Safety CBF (the value the runtime filter and certificate use):

    V(px,py,v) = clearance(p) - v^2/(2 b) - margin ,
      clearance(p) = min( ||p-o|| - r_obs ,  R_world - ||p|| ) ,
      b = |a_min| - d_a_max               (GUARANTEED braking deceleration)

V >= 0  <=>  the car can brake to a full stop before the obstacle / wall even
under worst-case actuator error.  Key property (the contraction margin a
fixed-speed car lacks): under max braking the clearance lost per step (<= dt v)
exactly cancels the braking-distance recovered (~ dt v), so V is invariant to
O(dt^2); a small margin absorbs that and the box over-approximation.

V is heading-INDEPENDENT, so the certificate's V-bounds are exact in psi and
the successor V depends only on (px+, py+, v+) -- and on delta not at all.

Interval routines (trusted; covered by tests/test_bicycle_accel.py):
  * `brake_cbf_bounds`   exact min/max of V over a (px,py,v) box,
  * `braking_successor`  sound (px+,py+,v+) image under braking over a state box.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dubins import TWO_PI, cos_interval, sin_interval, wrap_angle


@dataclass(frozen=True)
class BicycleAccelConfig:
    dt: float = 0.05
    wheelbase: float = 0.33
    v_max: float = 3.0               # m/s (v_min = 0)
    a_min: float = -4.0              # m/s^2 (braking)
    a_max: float = 2.0
    delta_max: float = 0.40
    d_a_max: float = 0.5             # accel disturbance (m/s^2)
    d_delta_max: float = 0.05        # steer disturbance (rad)
    obs_center: tuple[float, float] = (0.0, 0.0)
    obs_radius: float = 0.50
    world_radius: float = 2.50
    p_lo: float = -3.0
    p_hi: float = 3.0
    cbf_margin: float = 0.10         # static safety margin in V
    g_fail: float = -1.0

    @property
    def brake_decel(self) -> float:
        """Guaranteed deceleration b = |a_min| - d_a_max (worst-case)."""
        return abs(self.a_min) - self.d_a_max


def brake_distance(cfg: "BicycleAccelConfig", v):
    """EXACT discrete braking distance from speed v: sum_k dt * v_k over the
    braking trajectory v_k = max(0, v - k dt b), b the guaranteed deceleration.

    Using the discrete (not continuous v^2/2b) distance makes the CBF decrease
    *exactly* zero per step: D(v) - D(v - dt b) = dt v, which cancels the
    clearance lost (<= dt v).  Monotone increasing in v >= 0.
    """
    v = np.asarray(v, float)
    b = cfg.brake_decel
    D = np.zeros_like(v)
    vk = np.maximum(v, 0.0)
    n = int(np.ceil(cfg.v_max / (cfg.dt * b))) + 1
    for _ in range(n):
        D = D + cfg.dt * vk
        vk = np.maximum(vk - cfg.dt * b, 0.0)
    return D


# --------------------------------------------------------------------------- #
def _imul(a_lo, a_hi, b_lo, b_hi):
    """Interval product [a_lo,a_hi]*[b_lo,b_hi] -> (min, max) (vectorized)."""
    p = np.stack([a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi], axis=0)
    return p.min(axis=0), p.max(axis=0)


def _axis_min_sq(lo, hi, c):
    inside = (lo <= c) & (c <= hi)
    m = np.minimum(np.abs(lo - c), np.abs(hi - c))
    return np.where(inside, 0.0, m) ** 2


def _axis_max_sq(lo, hi, c):
    return np.maximum(np.abs(lo - c), np.abs(hi - c)) ** 2


def dist_bounds(px_lo, px_hi, py_lo, py_hi, cx, cy):
    """Exact [min, max] of ||p - c|| over the position box."""
    smin = _axis_min_sq(px_lo, px_hi, cx) + _axis_min_sq(py_lo, py_hi, cy)
    smax = _axis_max_sq(px_lo, px_hi, cx) + _axis_max_sq(py_lo, py_hi, cy)
    return np.sqrt(smin), np.sqrt(smax)


# --------------------------------------------------------------------------- #
class BicycleAccelModel:
    def __init__(self, cfg: BicycleAccelConfig):
        self.cfg = cfg
        self.ox, self.oy = cfg.obs_center

    def step(self, x, u, d):
        c = self.cfg
        px, py, psi, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
        a, delta = u[..., 0], u[..., 1]
        da, dd = d[..., 0], d[..., 1]
        out = np.empty(np.broadcast(px, a, da).shape + (4,), dtype=float)
        out[..., 0] = px + c.dt * v * np.cos(psi)
        out[..., 1] = py + c.dt * v * np.sin(psi)
        out[..., 2] = wrap_angle(psi + c.dt * (v / c.wheelbase) * np.tan(delta + dd))
        out[..., 3] = np.clip(v + c.dt * (a + da), 0.0, c.v_max)
        return out

    def g(self, x):
        """Instantaneous collision margin (>=0 safe), squared form."""
        c = self.cfg
        px, py = x[..., 0], x[..., 1]
        d_obs = (px - self.ox) ** 2 + (py - self.oy) ** 2 - c.obs_radius ** 2
        d_wld = c.world_radius ** 2 - (px ** 2 + py ** 2)
        return np.minimum(d_obs, d_wld)

    def in_domain(self, x):
        c = self.cfg
        return ((x[..., 0] >= c.p_lo) & (x[..., 0] <= c.p_hi)
                & (x[..., 1] >= c.p_lo) & (x[..., 1] <= c.p_hi))

    def brake_cbf(self, x):
        """V(px,py,v) = clearance - v^2/(2b) - margin (pointwise)."""
        c = self.cfg
        px, py, v = x[..., 0], x[..., 1], x[..., 3]
        clr = np.minimum(np.sqrt((px - self.ox) ** 2 + (py - self.oy) ** 2) - c.obs_radius,
                         c.world_radius - np.sqrt(px ** 2 + py ** 2))
        return clr - brake_distance(c, v) - c.cbf_margin


# --------------------------------------------------------------------------- #
def brake_cbf_bounds(cfg: BicycleAccelConfig, px_lo, px_hi, py_lo, py_hi,
                     v_lo, v_hi):
    """Exact [min, max] of V over a (px,py,v) box (closed form)."""
    ox, oy = cfg.obs_center
    obs_lo, obs_hi = dist_bounds(px_lo, px_hi, py_lo, py_hi, ox, oy)
    org_lo, org_hi = dist_bounds(px_lo, px_hi, py_lo, py_hi, 0.0, 0.0)
    c_obs_lo = obs_lo - cfg.obs_radius                # ||p-o|| - r
    c_obs_hi = obs_hi - cfg.obs_radius
    c_wld_lo = cfg.world_radius - org_hi              # R - ||p||
    c_wld_hi = cfg.world_radius - org_lo
    clr_lo = np.minimum(c_obs_lo, c_wld_lo)
    clr_hi = np.minimum(c_obs_hi, c_wld_hi)
    # discrete braking distance D(v) is increasing in v >= 0
    D_lo = brake_distance(cfg, np.minimum(v_lo, v_hi))
    D_hi = brake_distance(cfg, np.maximum(v_lo, v_hi))
    V_lo = clr_lo - D_hi - cfg.cbf_margin
    V_hi = clr_hi - D_lo - cfg.cbf_margin
    return V_lo, V_hi


def braking_successor(cfg: BicycleAccelConfig, px_lo, px_hi, py_lo, py_hi,
                      psi_lo, psi_hi, v_lo, v_hi):
    """Sound (px+, py+, v+) image of the box under the BRAKING fallback
    (a = a_min, worst d_a; any delta -- irrelevant to (px+,py+,v+)).

    Returns six arrays: npx_lo, npx_hi, npy_lo, npy_hi, nv_lo, nv_hi.
    """
    dt = cfg.dt
    cmin, cmax = cos_interval(psi_lo, psi_hi)
    smin, smax = sin_interval(psi_lo, psi_hi)
    vcos_lo, vcos_hi = _imul(v_lo, v_hi, cmin, cmax)
    vsin_lo, vsin_hi = _imul(v_lo, v_hi, smin, smax)
    npx_lo = px_lo + dt * vcos_lo
    npx_hi = px_hi + dt * vcos_hi
    npy_lo = py_lo + dt * vsin_lo
    npy_hi = py_hi + dt * vsin_hi
    # v+ under braking: a = a_min, d_a in [-d_a_max, +d_a_max]
    acc_lo = cfg.a_min - cfg.d_a_max
    acc_hi = cfg.a_min + cfg.d_a_max
    nv_lo = np.clip(v_lo + dt * acc_lo, 0.0, cfg.v_max)
    nv_hi = np.clip(v_hi + dt * acc_hi, 0.0, cfg.v_max)
    return npx_lo, npx_hi, npy_lo, npy_hi, nv_lo, nv_hi


def successor_box(cfg: BicycleAccelConfig, px_lo, px_hi, py_lo, py_hi,
                  psi_lo, psi_hi, v_lo, v_hi, a_lo, a_hi, da_lo, da_hi):
    """Sound (px+, py+, v+) image of the box under a GENERAL accel command.

    Generalises ``braking_successor`` to an arbitrary commanded-accel range
    ``a in [a_lo, a_hi]`` and disturbance ``d_a in [da_lo, da_hi]`` (steering
    and its disturbance are irrelevant to (px+, py+, v+) since V is heading-
    independent).  Used by the LEARNED certificate's C2 closure to bound
    V_theta over the successor of cell x heading-cell x accel-cell x D.

    Returns six arrays: npx_lo, npx_hi, npy_lo, npy_hi, nv_lo, nv_hi.
    """
    dt = cfg.dt
    cmin, cmax = cos_interval(psi_lo, psi_hi)
    smin, smax = sin_interval(psi_lo, psi_hi)
    vcos_lo, vcos_hi = _imul(v_lo, v_hi, cmin, cmax)     # v >= 0
    vsin_lo, vsin_hi = _imul(v_lo, v_hi, smin, smax)
    npx_lo = px_lo + dt * vcos_lo
    npx_hi = px_hi + dt * vcos_hi
    npy_lo = py_lo + dt * vsin_lo
    npy_hi = py_hi + dt * vsin_hi
    nv_lo = np.clip(v_lo + dt * (a_lo + da_lo), 0.0, cfg.v_max)
    nv_hi = np.clip(v_hi + dt * (a_hi + da_hi), 0.0, cfg.v_max)
    return npx_lo, npx_hi, npy_lo, npy_hi, nv_lo, nv_hi


def heading_successor_interval(cfg: BicycleAccelConfig, psi_lo, psi_hi,
                               v_lo, v_hi, delta_cmd):
    """Sound (UN-wrapped) interval of psi+ = psi + dt*(v/L)*tan(delta_cmd + d_delta)
    over a (psi, v) box and the full steer disturbance d_delta in [-dd, dd].

    Used by the grow-from-seed engine (heading evolves under a racing/steer menu
    action, unlike the heading-free braking successor).  Returns the raw lower /
    upper heading bounds BEFORE wrapping -- the cell-index converter handles the
    periodic wrap (so a single sound interval may map to one or two index ranges).

    Soundness: tan is monotone increasing on (-pi/2, pi/2) and the argument range
    [delta_cmd - dd, delta_cmd + dd] is strictly inside it (|delta|<=0.45 here),
    so tan(arg) in [tan(delta_cmd-dd), tan(delta_cmd+dd)].  v/L >= 0 (v>=0), so the
    yaw-rate interval is the 4-corner product of [v_lo/L, v_hi/L] and the tan box.
    """
    L = cfg.wheelbase
    dd = cfg.d_delta_max
    t_lo = np.tan(delta_cmd - dd)
    t_hi = np.tan(delta_cmd + dd)
    vL_lo = np.asarray(v_lo, float) / L
    vL_hi = np.asarray(v_hi, float) / L
    rate_lo, rate_hi = _imul(vL_lo, vL_hi, t_lo, t_hi)     # (v/L)*tan, v>=0
    psi_succ_lo = np.asarray(psi_lo, float) + cfg.dt * rate_lo
    psi_succ_hi = np.asarray(psi_hi, float) + cfg.dt * rate_hi
    return psi_succ_lo, psi_succ_hi


def g_bounds_sq(cfg: BicycleAccelConfig, px_lo, px_hi, py_lo, py_hi):
    """Exact min of the squared collision margin g over a position box (for C1)."""
    ox, oy = cfg.obs_center
    obs_min = (_axis_min_sq(px_lo, px_hi, ox) + _axis_min_sq(py_lo, py_hi, oy)
               - cfg.obs_radius ** 2)
    wld_min = cfg.world_radius ** 2 - (_axis_max_sq(px_lo, px_hi, 0.0)
                                       + _axis_max_sq(py_lo, py_hi, 0.0))
    return np.minimum(obs_min, wld_min)
