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


# ------------------------------------------------------------------ T9
def test_c4_successor_bounds() -> None:
    """C4 verifier helpers: the per-cell control range encloses clip(pi(x)),
    and the heading-subsplit successor lower bound under-estimates the true
    min_d V_theta(f(x,u,d)) for every in-cell (x, u in range, d in D)."""
    print("T9: _control_range / _succ_v_lower soundness (C4 tightening)")
    from qcbf.config import ExperimentConfig
    from qcbf.certify.cells import CellLattice
    from qcbf.certify.spec import _control_range, _succ_v_lower
    from qcbf.verify.compiler import compile_policy

    cfg = ExperimentConfig()
    dyn, cert = cfg.dynamics, cfg.cert
    model = DubinsModel(dyn)
    v = MLP([3, 16, 16, 1], seed=7)
    pi = MLP([3, 16, 16, 1], seed=8)
    rr = np.random.default_rng(9)
    for net in (v, pi):
        for i in range(len(net.b)):
            net.b[i] = rr.normal(0, 0.3, size=net.b[i].shape)
    pi.W[-1] *= 4.0                       # exercise the clamp
    v_net = SeqNet.from_mlp(v)
    pol = compile_policy(pi, dyn.control_max)

    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    # interior cells only, so successors stay in the position domain
    interior = ((boxes[:, 0] > dyn.p_lo + 0.3) & (boxes[:, 1] < dyn.p_hi - 0.3)
                & (boxes[:, 2] > dyn.p_lo + 0.3) & (boxes[:, 3] < dyn.p_hi - 0.3))
    cb = boxes[interior][rr.choice(int(interior.sum()), 200, replace=False)]

    u_lo, u_hi = _control_range(pol, cb, dyn, cert)
    K = 40
    sx = cb[:, 0:1] + rr.uniform(size=(len(cb), K)) * (cb[:, 1] - cb[:, 0])[:, None]
    sy = cb[:, 2:3] + rr.uniform(size=(len(cb), K)) * (cb[:, 3] - cb[:, 2])[:, None]
    sp = cb[:, 4:5] + rr.uniform(size=(len(cb), K)) * (cb[:, 5] - cb[:, 4])[:, None]
    Xs = np.stack([sx, sy, sp], -1)
    us = policy_forward(pi, Xs.reshape(-1, 3), dyn.control_max).reshape(len(cb), K)
    ok_u = bool(((us >= u_lo[:, None] - TOL) & (us <= u_hi[:, None] + TOL)).all())
    check("_control_range encloses clip(pi(x))", ok_u,
          f"mean width {np.mean(u_hi-u_lo):.3f} vs full {2*dyn.control_max:.1f}")

    lb = _succ_v_lower(v_net, cb, dyn, cert, u_lo, u_hi)
    uu = u_lo[:, None] + rr.uniform(size=(len(cb), K)) * (u_hi - u_lo)[:, None]
    dd = rr.uniform(-dyn.d_max, dyn.d_max, size=(len(cb), K))
    nxt = model.step(Xs, uu, dd)
    Vf = v(nxt.reshape(-1, 3)).reshape(len(cb), K)
    ok_lb = bool((Vf >= lb[:, None] - 1e-6).all())
    worst = float((Vf - lb[:, None]).min())
    check("_succ_v_lower is a sound lower bound of V(f)", ok_lb,
          f"min slack={worst:+.2e}")


# ------------------------------------------------------------------ T10
def test_ibp_grad() -> None:
    """The hand-coded differentiable IBP backward matches finite differences
    (used only for verifier-in-the-loop TRAINING; the verifier's own bounds in
    verify/bounds.py remain the trusted, separately-tested copy)."""
    print("T10: differentiable IBP backward vs finite differences")
    from qcbf.nets.mlp import ibp_forward, ibp_backward
    rr = np.random.default_rng(0)
    net = MLP([3, 16, 12, 1], seed=2)
    for i in range(len(net.b)):
        net.b[i] = rr.normal(0, 0.3, size=net.b[i].shape)
    B = 8
    c = rr.normal(0, 1, (B, 3)); rad = rr.uniform(0.05, 0.5, (B, 3))
    lo, hi = c - rad, c + rad
    dlb = rr.normal(0, 1, (B, 1)); dub = rr.normal(0, 1, (B, 1))

    def loss():
        lb, ub, _ = ibp_forward(net, lo, hi)
        return float(np.sum(dlb * lb + dub * ub))

    _, _, cache = ibp_forward(net, lo, hi)
    gW, gb = ibp_backward(net, cache, dlb, dub)
    eps = 1e-6; maxerr = 0.0
    for li in range(len(net.W)):
        for _ in range(15):
            a, b_ = rr.integers(net.W[li].shape[0]), rr.integers(net.W[li].shape[1])
            old = net.W[li][a, b_]
            net.W[li][a, b_] = old + eps; lp = loss()
            net.W[li][a, b_] = old - eps; lm = loss(); net.W[li][a, b_] = old
            maxerr = max(maxerr, abs((lp - lm) / (2 * eps) - gW[li][a, b_]))
    check("ibp_backward matches FD", maxerr < 1e-5, f"max|err|={maxerr:.2e}")


# ------------------------------------------------------------------ T8
def test_oracle_interp() -> None:
    print("T8: oracle interpolation consistency at grid nodes")
    dyn = DubinsConfig()
    ocfg = OracleConfig(n_px=21, n_py=21, n_psi=16, n_u=5, n_d=3, max_iters=5)
    oracle = DubinsOracle(dyn, ocfg, gamma=0.5)
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
    test_c4_successor_bounds()
    test_ibp_grad()
    test_oracle_interp()
    print("-" * 60)
    if FAILS:
        print(f"{FAILS} test(s) FAILED")
        sys.exit(1)
    print("all soundness tests passed")
