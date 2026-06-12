"""Soundness suite for the 4-state brakeable bicycle interval primitives.

These are the trusted base of the braking-CBF certificate: every interval must
enclose its pointwise function. If they pass, certify_brake_sublevel is sound.

    python tests/test_bicycle_accel.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         brake_cbf_bounds, braking_successor,
                                         dist_bounds, g_bounds_sq,
                                         heading_successor_interval, successor_box)
from qcbf.dynamics.dubins import wrap_angle

OK = "  [ok ]"


def test_dist_bounds():
    rng = np.random.default_rng(0)
    bad = 0
    for _ in range(500):
        lo = rng.uniform(-3, 3, 2); hi = lo + rng.uniform(0, 1.5, 2)
        cx, cy = rng.uniform(-3, 3, 2)
        dlo, dhi = dist_bounds(lo[0], hi[0], lo[1], hi[1], cx, cy)
        px = rng.uniform(lo[0], hi[0], 3000); py = rng.uniform(lo[1], hi[1], 3000)
        dd = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
        if dd.min() < dlo - 1e-9 or dd.max() > dhi + 1e-9:
            bad += 1
    assert bad == 0, f"dist_bounds violated on {bad}"
    print(f"{OK} dist_bounds sound")


def test_cbf_bounds():
    cfg = BicycleAccelConfig(); m = BicycleAccelModel(cfg)
    rng = np.random.default_rng(1); bad = 0
    for _ in range(500):
        lo = rng.uniform(-3, 3, 2); hi = lo + rng.uniform(0, 1.0, 2)
        vlo = rng.uniform(0, cfg.v_max); vhi = min(cfg.v_max, vlo + rng.uniform(0, 1.0))
        Vlo, Vhi = brake_cbf_bounds(cfg, lo[0], hi[0], lo[1], hi[1], vlo, vhi)
        px = rng.uniform(lo[0], hi[0], 4000); py = rng.uniform(lo[1], hi[1], 4000)
        v = rng.uniform(vlo, vhi, 4000)
        x = np.stack([px, py, np.zeros_like(px), v], axis=-1)
        V = m.brake_cbf(x)
        if V.min() < Vlo - 1e-9 or V.max() > Vhi + 1e-9:
            bad += 1
    assert bad == 0, f"brake_cbf_bounds violated on {bad}"
    print(f"{OK} brake_cbf_bounds sound")


def test_braking_successor():
    cfg = BicycleAccelConfig(); m = BicycleAccelModel(cfg)
    rng = np.random.default_rng(2); N = 800
    px_lo = rng.uniform(-3, 2.5, N); px_hi = px_lo + rng.uniform(0.02, 0.3, N)
    py_lo = rng.uniform(-3, 2.5, N); py_hi = py_lo + rng.uniform(0.02, 0.3, N)
    ps_lo = rng.uniform(-np.pi, np.pi - 0.5, N); ps_hi = ps_lo + rng.uniform(0.02, 0.5, N)
    v_lo = rng.uniform(0, cfg.v_max - 0.5, N); v_hi = v_lo + rng.uniform(0.02, 0.5, N)
    nx0, nx1, ny0, ny1, nv0, nv1 = braking_successor(
        cfg, px_lo, px_hi, py_lo, py_hi, ps_lo, ps_hi, v_lo, v_hi)
    bad = 0
    for _ in range(60):
        px = px_lo + (px_hi - px_lo) * rng.uniform(0, 1, N)
        py = py_lo + (py_hi - py_lo) * rng.uniform(0, 1, N)
        ps = ps_lo + (ps_hi - ps_lo) * rng.uniform(0, 1, N)
        v = v_lo + (v_hi - v_lo) * rng.uniform(0, 1, N)
        x = np.stack([px, py, ps, v], axis=-1)
        # braking: a = a_min, any d_a, any delta/d_delta
        u = np.stack([np.full(N, cfg.a_min), rng.uniform(-cfg.delta_max, cfg.delta_max, N)], axis=-1)
        d = np.stack([rng.uniform(-cfg.d_a_max, cfg.d_a_max, N),
                      rng.uniform(-cfg.d_delta_max, cfg.d_delta_max, N)], axis=-1)
        nx = m.step(x, u, d)
        inb = ((nx[:, 0] >= nx0 - 1e-9) & (nx[:, 0] <= nx1 + 1e-9)
               & (nx[:, 1] >= ny0 - 1e-9) & (nx[:, 1] <= ny1 + 1e-9)
               & (nx[:, 3] >= nv0 - 1e-9) & (nx[:, 3] <= nv1 + 1e-9))
        bad += int((~inb).sum())
    assert bad == 0, f"braking_successor not contained for {bad}"
    print(f"{OK} braking_successor contains (px+,py+,v+) under braking")


def test_successor_box():
    cfg = BicycleAccelConfig(); m = BicycleAccelModel(cfg)
    rng = np.random.default_rng(4); N = 800
    px_lo = rng.uniform(-3, 2.5, N); px_hi = px_lo + rng.uniform(0.02, 0.3, N)
    py_lo = rng.uniform(-3, 2.5, N); py_hi = py_lo + rng.uniform(0.02, 0.3, N)
    ps_lo = rng.uniform(-np.pi, np.pi - 0.5, N); ps_hi = ps_lo + rng.uniform(0.02, 0.6, N)
    v_lo = rng.uniform(0, cfg.v_max - 0.5, N); v_hi = v_lo + rng.uniform(0.02, 0.5, N)
    a_lo = rng.uniform(cfg.a_min, cfg.a_max - 0.3, N); a_hi = a_lo + rng.uniform(0.02, 0.3, N)
    nx0, nx1, ny0, ny1, nv0, nv1 = successor_box(
        cfg, px_lo, px_hi, py_lo, py_hi, ps_lo, ps_hi, v_lo, v_hi,
        a_lo, a_hi, -cfg.d_a_max, cfg.d_a_max)
    bad = 0
    for _ in range(60):
        px = px_lo + (px_hi - px_lo) * rng.uniform(0, 1, N)
        py = py_lo + (py_hi - py_lo) * rng.uniform(0, 1, N)
        ps = ps_lo + (ps_hi - ps_lo) * rng.uniform(0, 1, N)
        v = v_lo + (v_hi - v_lo) * rng.uniform(0, 1, N)
        a = a_lo + (a_hi - a_lo) * rng.uniform(0, 1, N)
        u = np.stack([a, rng.uniform(-cfg.delta_max, cfg.delta_max, N)], axis=-1)
        d = np.stack([rng.uniform(-cfg.d_a_max, cfg.d_a_max, N),
                      rng.uniform(-cfg.d_delta_max, cfg.d_delta_max, N)], axis=-1)
        nx = m.step(x=np.stack([px, py, ps, v], axis=-1), u=u, d=d)
        inb = ((nx[:, 0] >= nx0 - 1e-9) & (nx[:, 0] <= nx1 + 1e-9)
               & (nx[:, 1] >= ny0 - 1e-9) & (nx[:, 1] <= ny1 + 1e-9)
               & (nx[:, 3] >= nv0 - 1e-9) & (nx[:, 3] <= nv1 + 1e-9))
        bad += int((~inb).sum())
    assert bad == 0, f"successor_box not contained for {bad}"
    print(f"{OK} successor_box contains (px+,py+,v+) under general accel")


def test_heading_successor_interval():
    """psi+ (un-wrapped) interval used by the grow-from-seed engine must enclose
    the true heading update over the (psi,v) box, full steer disturbance, and any
    menu steering command."""
    cfg = BicycleAccelConfig(); rng = np.random.default_rng(5); N = 800
    bad = 0
    for dcmd in (-cfg.delta_max, -0.2, 0.0, 0.2, cfg.delta_max):
        ps_lo = rng.uniform(-np.pi, np.pi, N); ps_hi = ps_lo + rng.uniform(0.02, 0.6, N)
        v_lo = rng.uniform(0, cfg.v_max - 0.3, N); v_hi = v_lo + rng.uniform(0.02, 0.4, N)
        np_lo, np_hi = heading_successor_interval(cfg, ps_lo, ps_hi, v_lo, v_hi, dcmd)
        for _ in range(40):
            ps = ps_lo + (ps_hi - ps_lo) * rng.uniform(0, 1, N)
            v = v_lo + (v_hi - v_lo) * rng.uniform(0, 1, N)
            dd = rng.uniform(-cfg.d_delta_max, cfg.d_delta_max, N)
            psi1 = ps + cfg.dt * (v / cfg.wheelbase) * np.tan(dcmd + dd)   # un-wrapped
            inb = (psi1 >= np_lo - 1e-9) & (psi1 <= np_hi + 1e-9)
            bad += int((~inb).sum())
    assert bad == 0, f"heading_successor_interval not contained for {bad}"
    print(f"{OK} heading_successor_interval contains psi+ (un-wrapped) under full D")


def test_g_bounds():
    cfg = BicycleAccelConfig(); m = BicycleAccelModel(cfg)
    rng = np.random.default_rng(3); bad = 0
    for _ in range(400):
        lo = rng.uniform(-3, 3, 2); hi = lo + rng.uniform(0, 0.8, 2)
        gmin = g_bounds_sq(cfg, lo[0], hi[0], lo[1], hi[1])
        px = rng.uniform(lo[0], hi[0], 4000); py = rng.uniform(lo[1], hi[1], 4000)
        x = np.stack([px, py, np.zeros_like(px), np.zeros_like(px)], axis=-1)
        if m.g(x).min() < gmin - 1e-9:
            bad += 1
    assert bad == 0, f"g_bounds_sq violated on {bad}"
    print(f"{OK} g_bounds_sq sound")


def main():
    print("Brakeable bicycle (4-state) interval-primitive soundness")
    test_dist_bounds(); test_cbf_bounds(); test_braking_successor()
    test_successor_box(); test_heading_successor_interval(); test_g_bounds()
    print("-" * 60)
    print("all brakeable-bicycle soundness tests passed")


if __name__ == "__main__":
    main()
