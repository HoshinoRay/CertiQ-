"""Soundness of the P1 direct-composition primitive (qcbf/certify/direct.py).

D1  crown_lower_affine functional lower-bounds the network everywhere in the box
     and contracts to crown_bounds()[0] exactly.
D2  crown_brake_successor_lb <= V_theta(f_brake(x, a_min, d)) for every sampled
     (x in cell, psi in heading box, d in D), AND it is >= the box-route bound
     (tighter-or-equal -- the direct-composition gain).

Run:  python -m tests.test_direct
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from qcbf.dynamics.bicycle_accel import BicycleAccelConfig, successor_box
from qcbf.nets.mlp import MLP
from qcbf.verify.bounds import (SeqNet, crown_bounds, crown_bounds_chunked,
                                crown_lower_affine, crown_upper_affine)
from qcbf.certify.direct import (crown_brake_successor_lb,
                                 crown_relational_decrease_lb)

TOL = 1e-7
rng = np.random.default_rng(7)
FAILS = 0


def check(name, ok, detail=""):
    global FAILS
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS += 1


def rand_net(seed):
    mlp = MLP([3, 32, 32, 1], seed=seed)
    r = np.random.default_rng(seed + 1)
    for i in range(len(mlp.b)):
        mlp.b[i] = r.normal(0, 0.3, size=mlp.b[i].shape)
    return SeqNet.from_mlp(mlp)


def test_affine():
    print("D1: crown_lower_affine sound + matches crown_bounds")
    for t in range(5):
        net = rand_net(100 + t)
        B = 50
        c = rng.normal(0, 1, (B, 3)); r = rng.uniform(.05, .6, (B, 3))
        lb, ub = c - r, c + r
        lo, _ = crown_bounds(net, lb, ub, True)
        A, beta = crown_lower_affine(net, lb, ub, True)
        Apos, Aneg = np.maximum(A, 0), np.minimum(A, 0)
        lo2 = (np.einsum("bmj,bj->bm", Apos, lb)
               + np.einsum("bmj,bj->bm", Aneg, ub) + beta)
        xs = lb[:, None, :] + rng.uniform(size=(B, 400, 3)) * (ub - lb)[:, None, :]
        ys = net.forward(xs.reshape(-1, 3)).reshape(B, 400, 1)
        func = np.einsum("bmj,bsj->bsm", A, xs) + beta[:, None, :]
        check(f"affine matches crown_bounds t={t}", np.allclose(lo, lo2, atol=1e-10))
        check(f"affine sound t={t}", bool((func <= ys + TOL).all()),
              f"min margin={float((ys - func).min()):+.2e}")


def test_direct_successor():
    print("D2: direct-composition brake-successor bound sound + tighter")
    cfg = BicycleAccelConfig()
    for t in range(5):
        net = rand_net(200 + t)
        M = 300
        px_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.3, M); px_hi = px_lo + rng.uniform(0.02, 0.25, M)
        py_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.3, M); py_hi = py_lo + rng.uniform(0.02, 0.25, M)
        v_lo = rng.uniform(0, cfg.v_max - 0.3, M); v_hi = v_lo + rng.uniform(0.02, 0.25, M)
        ps_lo = rng.uniform(-np.pi, np.pi - 0.5, M); ps_hi = ps_lo + rng.uniform(0.02, 0.5, M)

        lb_dir = crown_brake_successor_lb(cfg, net, px_lo, px_hi, py_lo, py_hi,
                                          ps_lo, ps_hi, v_lo, v_hi)

        # box route (what run_cert_learned uses): CROWN over the successor box
        bx0, bx1, by0, by1, bv0, bv1 = successor_box(
            cfg, px_lo, px_hi, py_lo, py_hi, ps_lo, ps_hi, v_lo, v_hi,
            cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)
        lb_box, _ = crown_bounds_chunked(net, np.column_stack([bx0, by0, bv0]),
                                         np.column_stack([bx1, by1, bv1]), True, 4096)
        lb_box = lb_box[:, 0]

        # sample the TRUE successor over cell x heading-box x D
        K = 600
        u = rng.uniform(size=(M, K, 4))
        px = px_lo[:, None] + u[..., 0] * (px_hi - px_lo)[:, None]
        py = py_lo[:, None] + u[..., 1] * (py_hi - py_lo)[:, None]
        v = v_lo[:, None] + u[..., 2] * (v_hi - v_lo)[:, None]
        ps = ps_lo[:, None] + u[..., 3] * (ps_hi - ps_lo)[:, None]
        da = rng.uniform(-cfg.d_a_max, cfg.d_a_max, size=(M, K))
        npx = px + cfg.dt * v * np.cos(ps)
        npy = py + cfg.dt * v * np.sin(ps)
        nv = np.clip(v + cfg.dt * (cfg.a_min + da), 0.0, cfg.v_max)
        Vp = net.forward(np.stack([npx, npy, nv], -1).reshape(-1, 3)).reshape(M, K)

        sound = bool((Vp >= lb_dir[:, None] - TOL).all())
        tighter = bool((lb_dir >= lb_box - TOL).all())
        gain = float(np.median(lb_dir - lb_box))
        slack = float((Vp.min(1) - lb_dir).min())
        check(f"direct bound sound t={t}", sound, f"min sample margin={slack:+.2e}")
        check(f"direct >= box route t={t}", tighter,
              f"median tightening={gain:+.4f}")


def test_upper_affine():
    print("D3: crown_upper_affine sound + matches crown_bounds[1]")
    for t in range(5):
        net = rand_net(300 + t)
        B = 50
        c = rng.normal(0, 1, (B, 3)); r = rng.uniform(.05, .6, (B, 3))
        lb, ub = c - r, c + r
        _, hi = crown_bounds(net, lb, ub, True)
        A, beta = crown_upper_affine(net, lb, ub, True)
        Apos, Aneg = np.maximum(A, 0), np.minimum(A, 0)
        hi2 = (np.einsum("bmj,bj->bm", Apos, ub)
               + np.einsum("bmj,bj->bm", Aneg, lb) + beta)
        xs = lb[:, None, :] + rng.uniform(size=(B, 400, 3)) * (ub - lb)[:, None, :]
        ys = net.forward(xs.reshape(-1, 3)).reshape(B, 400, 1)
        func = np.einsum("bmj,bsj->bsm", A, xs) + beta[:, None, :]
        check(f"upper matches crown_bounds t={t}", np.allclose(hi, hi2, atol=1e-10))
        check(f"upper sound t={t}", bool((func >= ys - TOL).all()),
              f"min margin={float((func - ys).min()):+.2e}")


def test_relational():
    print("D4: relational decrease bound sound + L_rel = lbV + G_lb tighter")
    cfg = BicycleAccelConfig()
    for t in range(5):
        net = rand_net(400 + t)
        M = 300
        px_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.3, M); px_hi = px_lo + rng.uniform(0.02, 0.25, M)
        py_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.3, M); py_hi = py_lo + rng.uniform(0.02, 0.25, M)
        v_lo = rng.uniform(0, cfg.v_max - 0.3, M); v_hi = v_lo + rng.uniform(0.02, 0.25, M)
        ps_lo = rng.uniform(-np.pi, np.pi - 0.5, M); ps_hi = ps_lo + rng.uniform(0.02, 0.5, M)

        G_lb = crown_relational_decrease_lb(cfg, net, px_lo, px_hi, py_lo, py_hi,
                                            ps_lo, ps_hi, v_lo, v_hi, rho_level=1.0)
        lbV, _ = crown_bounds_chunked(net, np.column_stack([px_lo, py_lo, v_lo]),
                                      np.column_stack([px_hi, py_hi, v_hi]), True, 4096)
        lbV = lbV[:, 0]
        L_rel = lbV + G_lb                                     # Mode-A successor lb
        L_dec = crown_brake_successor_lb(cfg, net, px_lo, px_hi, py_lo, py_hi,
                                         ps_lo, ps_hi, v_lo, v_hi)

        # sample the TRUE decrease G = V(f) - V(x) and successor V(f)
        K = 600
        u = rng.uniform(size=(M, K, 4))
        px = px_lo[:, None] + u[..., 0] * (px_hi - px_lo)[:, None]
        py = py_lo[:, None] + u[..., 1] * (py_hi - py_lo)[:, None]
        v = v_lo[:, None] + u[..., 2] * (v_hi - v_lo)[:, None]
        ps = ps_lo[:, None] + u[..., 3] * (ps_hi - ps_lo)[:, None]
        da = rng.uniform(-cfg.d_a_max, cfg.d_a_max, size=(M, K))
        npx = px + cfg.dt * v * np.cos(ps); npy = py + cfg.dt * v * np.sin(ps)
        nv = np.clip(v + cfg.dt * (cfg.a_min + da), 0.0, cfg.v_max)
        Vx = net.forward(np.stack([px, py, v], -1).reshape(-1, 3)).reshape(M, K)
        Vf = net.forward(np.stack([npx, npy, nv], -1).reshape(-1, 3)).reshape(M, K)
        Gtrue = Vf - Vx

        g_sound = bool((Gtrue >= G_lb[:, None] - TOL).all())
        l_sound = bool((Vf >= L_rel[:, None] - TOL).all())
        gain = float(np.median(L_rel - L_dec))            # object-dependent
        check(f"relational G sound t={t}", g_sound,
              f"min sample margin={float((Gtrue.min(1) - G_lb).min()):+.2e}")
        check(f"L_rel sound (<= V(f)) t={t}", l_sound,
              f"min sample margin={float((Vf.min(1) - L_rel).min()):+.2e}")
        # NOTE: tightening vs the decoupled bound is OBJECT-DEPENDENT (it needs the
        # successor/cell CROWN slopes to align, i.e. the braking-cancellation
        # structure of a distilled V_theta) -- on random nets it can be looser.
        # The real artifact-removal is measured on the learned object in
        # run_cert_rfc.py, not asserted here.  Reported only.
        print(f"        (random-net median L_rel - L_dec = {gain:+.4f}, "
              f"object-dependent, not a soundness check)")


if __name__ == "__main__":
    test_affine()
    test_direct_successor()
    test_upper_affine()
    test_relational()
    print("-" * 60)
    if FAILS:
        print(f"{FAILS} test(s) FAILED")
        sys.exit(1)
    print("all direct-composition soundness tests passed")
