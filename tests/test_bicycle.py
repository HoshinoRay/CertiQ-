"""Soundness suite for the F1TENTH kinematic-bicycle dynamics primitives.

Mirrors the Dubins T5-T7: the interval routines that enter the certificate's
trusted computing base must *enclose* their pointwise functions. If these pass,
the bicycle plant drops straight into the existing CROWN/IBP verifier, lattice
closure, runtime filter and audit (the position image and g are shared with
Dubins; only `tan_interval` and the heading shift in `successor_boxes` are new).

    python tests/test_bicycle.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qcbf.dynamics.bicycle import (BicycleConfig, BicycleModel, successor_boxes,
                                   tan_interval)
from qcbf.dynamics.dubins import g_bounds_on_box

PASS = "  [ok ]"


def _ok(name, extra=""):
    print(f"{PASS} {name}  {extra}")


def test_tan_interval():
    rng = np.random.default_rng(0)
    lim = np.pi / 2 - 0.2
    lo = rng.uniform(-lim, lim - 0.05, 500)
    hi = lo + rng.uniform(0.0, 0.2, 500)
    hi = np.minimum(hi, lim)
    tlo, thi = tan_interval(lo, hi)
    # sample inside each interval; enclosure must hold
    t = lo[:, None] + (hi - lo)[:, None] * rng.uniform(0, 1, (500, 64))
    vals = np.tan(t)
    assert np.all(vals >= tlo[:, None] - 1e-12), "tan lower bound violated"
    assert np.all(vals <= thi[:, None] + 1e-12), "tan upper bound violated"
    _ok("tan_interval sound (monotone branch)")


def test_g_bounds():
    cfg = BicycleConfig()
    model = BicycleModel(cfg)
    rng = np.random.default_rng(1)
    bad = 0
    for _ in range(400):
        lo = rng.uniform(cfg.p_lo, cfg.p_hi, 2)
        hi = lo + rng.uniform(0.0, 0.6, 2)
        gmin, gmax = g_bounds_on_box(cfg, lo[0], hi[0], lo[1], hi[1])
        px = rng.uniform(lo[0], hi[0], 4000)
        py = rng.uniform(lo[1], hi[1], 4000)
        g = model.g(np.stack([px, py, np.zeros_like(px)], axis=-1))
        if g.min() < gmin - 1e-9 or g.max() > gmax + 1e-9:
            bad += 1
    assert bad == 0, f"g bounds violated on {bad} boxes"
    _ok("g_bounds_on_box sound for bicycle g")


def test_successor_containment():
    cfg = BicycleConfig()
    model = BicycleModel(cfg)
    rng = np.random.default_rng(2)
    N = 800
    px_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.3, N); px_hi = px_lo + rng.uniform(0.02, 0.25, N)
    py_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.3, N); py_hi = py_lo + rng.uniform(0.02, 0.25, N)
    psi_lo = rng.uniform(-np.pi, np.pi - 0.4, N); psi_hi = psi_lo + rng.uniform(0.02, 0.4, N)
    u_lo, u_hi = -0.2, 0.35          # steering sub-cell
    d_lo, d_hi = -cfg.d_max, cfg.d_max

    b1, b2, m2 = successor_boxes(cfg, px_lo, px_hi, py_lo, py_hi, psi_lo, psi_hi,
                                 u_lo, u_hi, d_lo, d_hi)

    def in_box(nx, box):
        # position inside, heading inside [psi_lo, psi_hi]
        return ((nx[:, 0] >= box[:, 0] - 1e-9) & (nx[:, 0] <= box[:, 1] + 1e-9)
                & (nx[:, 1] >= box[:, 2] - 1e-9) & (nx[:, 1] <= box[:, 3] + 1e-9)
                & (nx[:, 2] >= box[:, 4] - 1e-9) & (nx[:, 2] <= box[:, 5] + 1e-9))

    bad = 0
    K = 60
    for _ in range(K):
        px = px_lo + (px_hi - px_lo) * rng.uniform(0, 1, N)
        py = py_lo + (py_hi - py_lo) * rng.uniform(0, 1, N)
        psi = psi_lo + (psi_hi - psi_lo) * rng.uniform(0, 1, N)
        u = rng.uniform(u_lo, u_hi, N)
        d = rng.uniform(d_lo, d_hi, N)
        nx = model.step(np.stack([px, py, psi], axis=-1), u, d)
        inside = in_box(nx, b1) | (m2 & in_box(nx, b2))
        bad += int((~inside).sum())
    assert bad == 0, f"successor not contained for {bad} samples"
    _ok("successor_boxes contain sampled successors",
        f"wrap-split active on {int(m2.sum())}/{N} boxes")


def main():
    print("Bicycle (F1TENTH) dynamics soundness")
    test_tan_interval()
    test_g_bounds()
    test_successor_containment()
    print("-" * 60)
    print("all bicycle soundness tests passed")


if __name__ == "__main__":
    main()
