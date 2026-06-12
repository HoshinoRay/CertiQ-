"""Randomized soundness tests for the certificate's trusted computing base.

Every claim the certificate rests on is exercised here with random inputs:

  T1  IBP / CROWN output bounds contain the true network output,
  T2  CROWN intermediate pre-activation bounds contain the true values,
  T3  the compiled h3 SeqNet computes EXACTLY the composition
        Q(x, clip(pi(x)), d) - gamma V(x) - eps,
  T4  the compiled policy SeqNet computes exactly clip(pi(x)),
  T5  cos/sin interval bounds are sound and tight at critical points,
  T6  exact g bounds on position boxes contain all sampled g values,
  T7  successor_boxes contains every sampled successor (incl. psi wrap),
  T8  oracle trilinear interpolation reproduces grid-node values.

Run:  python -m tests.test_soundness
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from qcbf.config import DubinsConfig, OracleConfig
from qcbf.dynamics.dubins import (DubinsModel, cos_interval, sin_interval,
                                  g_bounds_on_box, successor_boxes, wrap_angle)
from qcbf.nets.mlp import MLP, policy_forward
from qcbf.oracle.value_iteration import DubinsOracle
from qcbf.verify.bounds import (SeqNet, crown_bounds, ibp_preact_bounds,
                                crown_preact_bounds)
from qcbf.verify.compiler import compile_h3, compile_policy, as_seqnet

TOL = 1e-7
rng = np.random.default_rng(123)
FAILS = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global FAILS
    flag = "ok " if ok else "FAIL"
    print(f"  [{flag}] {name}  {detail}")
    if not ok:
        FAILS += 1


def random_net(n_in: int, n_out: int, depth: int, width: int, seed: int) -> SeqNet:
    sizes = [n_in] + [width] * depth + [n_out]
    mlp = MLP(sizes, seed=seed)
    r = np.random.default_rng(seed + 1)
    for i in range(len(mlp.b)):           # non-trivial biases
        mlp.b[i] = r.normal(0, 0.3, size=mlp.b[i].shape)
    return SeqNet.from_mlp(mlp), mlp


# ------------------------------------------------------------------ T1 / T2
def test_bound_soundness() -> None:
    print("T1/T2: IBP & CROWN soundness on random nets")
    for trial in range(6):
        n_in = int(rng.integers(2, 6))
        net, _ = random_net(n_in, int(rng.integers(1, 4)),
                            int(rng.integers(1, 4)), int(rng.integers(8, 48)),
                            seed=200 + trial)
        B = 64
        c = rng.normal(0, 1.0, size=(B, n_in))
        r = rng.uniform(0.01, 0.6, size=(B, n_in))
        lb, ub = c - r, c + r

        for tighten in (False, True):
            lo, hi = crown_bounds(net, lb, ub, tighten_intermediate=tighten)
            xs = lb[:, None, :] + rng.uniform(size=(B, 256, n_in)) * (ub - lb)[:, None, :]
            ys = net.forward(xs.reshape(-1, n_in)).reshape(B, 256, -1)
            ok = (ys >= lo[:, None, :] - TOL).all() and (ys <= hi[:, None, :] + TOL).all()
            worst = min(float((ys - lo[:, None, :]).min()),
                        float((hi[:, None, :] - ys).max()))
            check(f"crown_bounds trial={trial} tighten={tighten}", ok,
                  f"min slack={worst:+.2e}")

        # intermediate pre-activation bounds
        pre = crown_preact_bounds(net, lb, ub)
        xs = lb[:, None, :] + rng.uniform(size=(B, 128, n_in)) * (ub - lb)[:, None, :]
        h = xs.reshape(-1, n_in)
        ok = True
        for i, (W, b) in enumerate(zip(net.W, net.b)):
            z = h @ W + b
            l, u = pre[i]
            zb = z.reshape(B, 128, -1)
            ok &= bool((zb >= l[:, None, :] - TOL).all()
                       and (zb <= u[:, None, :] + TOL).all())
            if i < len(net.W) - 1:
                h = np.maximum(z, 0.0)
        check(f"crown_preact_bounds trial={trial}", ok)


# ------------------------------------------------------------------ T3 / T4
def test_compiler_exactness() -> None:
    print("T3/T4: compiled networks reproduce the exact composition")
    dyn = DubinsConfig()
    gamma, eps, om = 0.5, 5e-3, dyn.omega_max
    for trial in range(4):
        pi = MLP([3, 24, 24, 1], seed=300 + trial)
        v = MLP([3, 20, 20, 1], seed=400 + trial)
        q = MLP([5, 32, 32, 1], seed=500 + trial)
        r = np.random.default_rng(trial)
        for net in (pi, v, q):
            for i in range(len(net.b)):
                net.b[i] = r.normal(0, 0.3, size=net.b[i].shape)
            # scale pi head so the clamp actually saturates sometimes
        pi.W[-1] *= 4.0

        h3 = compile_h3(pi, q, v, gamma, eps, om)
        B = 512
        x = r.uniform(-2, 2, size=(B, 3))
        x[:, 2] = r.uniform(-np.pi, np.pi, size=B)
        d = r.uniform(-dyn.d_max, dyn.d_max, size=(B, 1))
        z = np.concatenate([x, d], axis=1)

        u = policy_forward(pi, x, om)
        exact = (q(np.concatenate([x, u, d], axis=1)).ravel()
                 - gamma * v(x).ravel() - eps)
        got = h3.forward(z).ravel()
        err = float(np.abs(got - exact).max())
        check(f"compile_h3 trial={trial}", err < 1e-9, f"max|err|={err:.2e}")

        pol = compile_policy(pi, om)
        got_u = pol.forward(x).ravel()
        err = float(np.abs(got_u - u.ravel()).max())
        sat = float(np.mean(np.abs(u) >= om - 1e-12))
        check(f"compile_policy trial={trial}", err < 1e-12,
              f"max|err|={err:.2e}  sat-frac={sat:.2f}")


# ------------------------------------------------------------------ T5
def test_trig_intervals() -> None:
    print("T5: cos/sin interval bounds")
    lo = rng.uniform(-3 * np.pi, 3 * np.pi, size=4000)
    hi = lo + rng.uniform(0, 2.2 * np.pi, size=4000)
    cmin, cmax = cos_interval(lo, hi)
    smin, smax = sin_interval(lo, hi)
    t = lo[:, None] + np.linspace(0, 1, 64)[None, :] * (hi - lo)[:, None]
    ok_c = ((np.cos(t) >= cmin[:, None] - TOL).all()
            and (np.cos(t) <= cmax[:, None] + TOL).all())
    ok_s = ((np.sin(t) >= smin[:, None] - TOL).all()
            and (np.sin(t) <= smax[:, None] + TOL).all())
    check("cos_interval sound", bool(ok_c))
    check("sin_interval sound", bool(ok_s))
    # tightness at endpoints when no critical point inside
    e_lo, e_hi = np.float64(0.3), np.float64(0.9)
    cm, cM = cos_interval(e_lo, e_hi)
    check("cos_interval tight (no critical pt)",
          abs(cm - np.cos(0.9)) < TOL and abs(cM - np.cos(0.3)) < TOL)


# ------------------------------------------------------------------ T6 / T7
def test_dynamics_intervals() -> None:
    print("T6/T7: exact g bounds + successor box containment")
    cfg = DubinsConfig()
    model = DubinsModel(cfg)
    N = 800
    px_lo = rng.uniform(-2, 1.7, N); px_hi = px_lo + rng.uniform(0, 0.4, N)
    py_lo = rng.uniform(-2, 1.7, N); py_hi = py_lo + rng.uniform(0, 0.4, N)
    ps_lo = rng.uniform(-np.pi, np.pi, N); ps_hi = ps_lo + rng.uniform(0, 0.8, N)
    ps_hi = np.minimum(ps_hi, np.pi)  # lattice cells never cross the seam

    gmin, gmax = g_bounds_on_box(cfg, px_lo, px_hi, py_lo, py_hi)
    K = 128
    sx = px_lo[:, None] + rng.uniform(size=(N, K)) * (px_hi - px_lo)[:, None]
    sy = py_lo[:, None] + rng.uniform(size=(N, K)) * (py_hi - py_lo)[:, None]
    g = model.g(np.stack([sx, sy, np.zeros_like(sx)], axis=-1))
    ok = ((g >= gmin[:, None] - TOL).all() and (g <= gmax[:, None] + TOL).all())
    check("g_bounds_on_box sound", bool(ok))
    # exactness: corners + projections must attain the bounds (sampled check)
    att = (np.abs(g.min(1) - gmin) < 1e-1).mean()
    check("g lower bound near-attained", att > 0.7, f"frac={att:.2f}")

    u_lo = rng.uniform(-1, 0.9); u_hi = u_lo + 0.1
    d_lo, d_hi = -cfg.d_max, cfg.d_max
    box1, box2, m2 = successor_boxes(cfg, px_lo, px_hi, py_lo, py_hi,
                                     ps_lo, ps_hi, u_lo, u_hi, d_lo, d_hi)
    sp = ps_lo[:, None] + rng.uniform(size=(N, K)) * (ps_hi - ps_lo)[:, None]
    su = rng.uniform(u_lo, u_hi, size=(N, K))
    sd = rng.uniform(d_lo, d_hi, size=(N, K))
    nxt = model.step(np.stack([sx, sy, sp], axis=-1), su, sd)
    npx, npy, nps = nxt[..., 0], nxt[..., 1], wrap_angle(nxt[..., 2])
    in1 = ((npx >= box1[:, None, 0] - TOL) & (npx <= box1[:, None, 1] + TOL)
           & (npy >= box1[:, None, 2] - TOL) & (npy <= box1[:, None, 3] + TOL)
           & (nps >= box1[:, None, 4] - TOL) & (nps <= box1[:, None, 5] + TOL))
    in2 = ((npx >= box2[:, None, 0] - TOL) & (npx <= box2[:, None, 1] + TOL)
           & (npy >= box2[:, None, 2] - TOL) & (npy <= box2[:, None, 3] + TOL)
           & (nps >= box2[:, None, 4] - TOL) & (nps <= box2[:, None, 5] + TOL)
           & m2[:, None])
    check("successor_boxes contain sampled successors", bool((in1 | in2).all()),
          f"wrap-split active on {int(m2.sum())}/{N} boxes")


# ------------------------------------------------------------------ T8
def test_oracle_interp() -> None:
    print("T8: oracle interpolation consistency at grid nodes")
    dyn = DubinsConfig()
    ocfg = OracleConfig(n_px=21, n_py=21, n_psi=16, n_u=5, n_d=3, max_iters=5)
    oracle = DubinsOracle(dyn, ocfg)
    g = oracle.grid
    V = rng.normal(size=(len(g.px), len(g.py), len(g.psi))).astype(np.float32)
    PX, PY, PSI = np.meshgrid(g.px, g.py, g.psi, indexing="ij")
    X = np.stack([PX, PY, PSI], axis=-1).reshape(-1, 3)
    got = oracle.interp_V(V, X).reshape(V.shape)
    err = float(np.abs(got - V).max())
    check("interp_V reproduces nodes", err < 1e-5, f"max|err|={err:.2e}")


if __name__ == "__main__":
    test_bound_soundness()
    test_compiler_exactness()
    test_trig_intervals()
    test_dynamics_intervals()
    test_oracle_interp()
    print("-" * 60)
    if FAILS:
        print(f"{FAILS} test(s) FAILED")
        sys.exit(1)
    print("all soundness tests passed")
