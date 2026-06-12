"""Distil the analytic braking-distance CBF into the learned trio (V_t, Q_t, pi_b).

This is the OBJECT-CONSTRUCTION half of the learned Gate-D experiment
(``run_cert_learned.py``).  Training is for **non-vacuity only**: it produces a
faithful learned approximation of the known-good analytic CBF so the verifier
has something non-trivial to certify.  The trained weights are then FROZEN and
the certificate is checked DIRECTLY on the networks with the true f -- the
distillation target is never a proof assumption (soundness firewall #1).

Targets (discrete braking distance D(v), never the continuous v^2/2b):

    V_target(px,py,v) = clearance(p) - D(v) - cbf_margin       (heading-free)
    Q_target(x,u,d)   = V_target( f(x,u,d) )                   (= V o f)
    pi_b_target(x)    = ( a_min , racing_tangent_steer(x) )    (braking witness)

The contraction margin ``m`` is the decrease margin in the co-training hinge

    relu( gamma V_t(x) - min_d Q_t(x, clip(pi_b(x)), d) + m ) ,

i.e. the trained witness over-achieves the discrete-CBF decrease by ``m``.  The
analytic CBF has exactly-zero worst-case contraction, so ``m`` is the only slack
that can absorb learned-approximation undershoot -- it is the swept knob.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         brake_distance)
from qcbf.nets.mlp import MLP, Adam
from qcbf.util.progress import Progress

# canonical action / disturbance bounds packed for clipping
GAMMA = 0.5                      # runtime decay (CertConfig/TrainConfig canon)


# --------------------------------------------------------------------------- #
def racing_steer(cfg, x):
    """State-dependent tangent steer of a CCW racing line about the obstacle."""
    px, py, psi = x[..., 0], x[..., 1], x[..., 2]
    psi_t = np.arctan2(py, px) + np.pi / 2.0
    return np.clip(2.0 * np.arctan2(np.sin(psi_t - psi), np.cos(psi_t - psi)),
                   -cfg.delta_max, cfg.delta_max)


def clip_action(cfg, y):
    """Deployed action = per-component clamp of the raw policy head (PWL)."""
    a = np.clip(y[..., 0], cfg.a_min, cfg.a_max)
    de = np.clip(y[..., 1], -cfg.delta_max, cfg.delta_max)
    return np.stack([a, de], axis=-1)


def v_target(cfg, px, py, v):
    ox, oy = cfg.obs_center
    clr = np.minimum(np.sqrt((px - ox) ** 2 + (py - oy) ** 2) - cfg.obs_radius,
                     cfg.world_radius - np.sqrt(px ** 2 + py ** 2))
    return clr - brake_distance(cfg, v) - cfg.cbf_margin


def d_probe_grid(cfg):
    """Corners + centre of the 2-D disturbance box (probes for min_d Q)."""
    da, dd = cfg.d_a_max, cfg.d_delta_max
    return np.array([[da, 0.0], [da, dd], [da, -dd], [-da, 0.0], [0.0, 0.0]])


# --------------------------------------------------------------------------- #
def _sample_states(cfg, n, rng, frac_safe=0.7):
    """States biased toward the safe-set boundary (where the cert is decided)."""
    xs = []
    while sum(len(z) for z in xs) < n:
        p = rng.uniform(cfg.p_lo, cfg.p_hi, (2 * n, 2))
        ps = rng.uniform(-np.pi, np.pi, 2 * n)
        v = rng.uniform(0.0, cfg.v_max, 2 * n)
        X = np.column_stack([p, ps, v])
        Vt = v_target(cfg, X[:, 0], X[:, 1], X[:, 3])
        keep = (Vt >= -0.6) if rng.random() < frac_safe else np.ones(len(X), bool)
        xs.append(X[keep])
    return np.concatenate(xs)[:n]


def _c2fix_states(cfg, n, rng, cex=None):
    """Boundary/core/unsafe mix for the C2-fix fine-tune (Step-0 follow-up).

    35% in-set boundary shell 0<=V_target<=0.12 (where the invariant certificate
    dies), 35% analytic safe core V_target>=0.2 (non-vacuity), 30% truly-unsafe
    g<=-0.05 (so V<=0 there).  ``cex`` (optional) injects CEGIS counterexamples
    + small jitter, replacing part of the boundary share.
    """
    def rej(cond, k):
        out, got = [], 0
        while got < max(k, 1):
            p = rng.uniform(cfg.p_lo, cfg.p_hi, (4 * k + 16, 2))
            ps = rng.uniform(-np.pi, np.pi, 4 * k + 16)
            vv = rng.uniform(0.0, cfg.v_max, 4 * k + 16)
            Xc = np.column_stack([p[:, 0], p[:, 1], ps, vv])
            m = cond(Xc)
            out.append(Xc[m]); got += int(m.sum())
        return np.concatenate(out)[:k]

    def Vt(Xc):
        return v_target(cfg, Xc[:, 0], Xc[:, 1], Xc[:, 3])

    n_band, n_core = int(0.35 * n), int(0.35 * n)
    n_uns = n - n_band - n_core
    if cex is not None and len(cex):                 # CEGIS round: jitter cex
        n_cex = min(len(cex) * 4, n_band // 2)
        jit = cex[rng.integers(0, len(cex), n_cex)].copy()
        jit[:, [0, 1]] += rng.uniform(-0.03, 0.03, (n_cex, 2))
        jit[:, 2] += rng.uniform(-0.1, 0.1, n_cex)
        jit[:, 3] = np.clip(jit[:, 3] + rng.uniform(-0.05, 0.05, n_cex), 0, cfg.v_max)
        n_band -= n_cex
    else:
        jit = np.zeros((0, 4))
    band = rej(lambda Xc: (Vt(Xc) >= 0.0) & (Vt(Xc) <= 0.12), n_band)
    core = rej(lambda Xc: Vt(Xc) >= 0.2, n_core)
    # unsafe positions: inside the obstacle disk or outside the world circle
    r_in = rng.uniform(0.0, 0.40, 3 * n_uns); a_in = rng.uniform(-np.pi, np.pi, 3 * n_uns)
    r_out = rng.uniform(2.55, 2.95, 3 * n_uns); a_out = rng.uniform(-np.pi, np.pi, 3 * n_uns)
    P = np.concatenate([np.column_stack([r_in * np.cos(a_in), r_in * np.sin(a_in)]),
                        np.column_stack([r_out * np.cos(a_out), r_out * np.sin(a_out)])])
    P = P[(P[:, 0] >= cfg.p_lo) & (P[:, 0] <= cfg.p_hi)
          & (P[:, 1] >= cfg.p_lo) & (P[:, 1] <= cfg.p_hi)]
    P = P[rng.permutation(len(P))[:n_uns]]
    uns = np.column_stack([P[:, 0], P[:, 1], rng.uniform(-np.pi, np.pi, len(P)),
                           rng.uniform(0.0, cfg.v_max, len(P))])
    X = np.concatenate([band, core, uns, jit])
    rng.shuffle(X)
    return X


def clearance(cfg, px, py):
    """Analytic obstacle/wall clearance min(||p-o||-r_obs, R_world-||p||)."""
    ox, oy = cfg.obs_center
    return np.minimum(np.sqrt((px - ox) ** 2 + (py - oy) ** 2) - cfg.obs_radius,
                      cfg.world_radius - np.sqrt(px ** 2 + py ** 2))


def train_clearance_net(cfg, seed=0, width=64, epochs=45, batch=1024, lr=1e-3,
                        lip_w=2.0, wd=1e-4, n=120_000, verbose=True):
    """STRUCTURED-V (Path B): learn C_theta(px,py) ~ clearance with a DIRECTIONAL
    Lipschitz penalty forbidding C_theta(p)-C_theta(p+) > ||p-p+|| over worst-case
    brake displacements.  Returns the MLP.  Paired with the analytic D(v), this
    preserves the 1-Lipschitz braking cancellation a black-box MLP cannot.
    """
    rng = np.random.default_rng(seed)
    C = MLP([2, width, width, 1], seed=seed)
    opt = Adam(C, lr=lr)
    P = rng.uniform(cfg.p_lo, cfg.p_hi, (n, 2))
    Y = clearance(cfg, P[:, 0], P[:, 1]).reshape(-1, 1)
    pb = Progress(epochs, "clearance C_theta") if verbose else None
    for ep in range(epochs):
        o = rng.permutation(n)
        for s in range(0, n, batch):
            idx = o[s:s + batch]; p = P[idx]; B = len(idx)
            out, zs, hs = C.forward(p, cache=True)
            gW, gb, _ = C.backward(zs, hs, 2.0 * (out - Y[idx]) / B)
            # directional Lipschitz penalty on worst-case brake displacement
            v = rng.uniform(0.0, cfg.v_max, B); ang = rng.uniform(-np.pi, np.pi, B)
            disp = cfg.dt * v
            pp = p + np.column_stack([disp * np.cos(ang), disp * np.sin(ang)])
            cp, zsp, hsp = C.forward(p, cache=True)
            cpp, zspp, hspp = C.forward(pp, cache=True)
            viol = (cp.ravel() - cpp.ravel()) - disp        # >0 violates Lipschitz
            act = (viol > 0.0).astype(float)
            gWp, gbp, _ = C.backward(zsp, hsp, (lip_w * 2.0 * viol * act / B)[:, None])
            gWpp, gbpp, _ = C.backward(zspp, hspp, (-lip_w * 2.0 * viol * act / B)[:, None])
            for i in range(len(C.W)):
                gW[i] = gW[i] + gWp[i] + gWpp[i] + wd * 2.0 * C.W[i]
                gb[i] = gb[i] + gbp[i] + gbpp[i]
            opt.step(gW, gb)
        if pb is not None:
            pb.update(ep + 1)
    if pb is not None:
        pb.done()
    return C, float(np.mean((C(P)[:, 0] - Y.ravel()) ** 2))


def make_nets(seed):
    # SMALL, low-width nets -> tighter CROWN bounds (the binding quantity: the
    # certified C3/C2 margins erode by ~the CROWN gap, so width is traded for
    # tightness).  V_target = clearance - D(v) - m is simple enough for 32-wide.
    return (MLP([3, 32, 32, 1], seed=seed),          # V_t(px,py,v)
            MLP([8, 64, 64, 1], seed=seed + 1),       # Q_t(x,u,d)
            MLP([4, 32, 32, 2], seed=seed + 2))       # pi_b(x) -> (a,delta)


# --------------------------------------------------------------------------- #
def _regress(net, X, Y, epochs, batch, lr, seed, tag, pb_desc, wd=0.0):
    """MSE regression with Adam + optional L2 weight decay (CROWN-tightness)."""
    rng = np.random.default_rng(seed)
    opt = Adam(net, lr=lr)
    n = len(X)
    Y = Y.reshape(n, -1)
    pb = Progress(epochs, pb_desc)
    mse = np.inf
    for ep in range(epochs):
        order = rng.permutation(n)
        tot = 0.0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            out, zs, hs = net.forward(X[idx], cache=True)
            err = out - Y[idx]
            tot += float(np.sum(err ** 2))
            gW, gb, _ = net.backward(zs, hs, 2.0 * err / len(idx))
            if wd:
                gW = [g + wd * 2.0 * W for g, W in zip(gW, net.W)]
            opt.step(gW, gb)
        mse = tot / n
        pb.update(ep + 1)
    pb.done()
    return mse


def _finetune_cbf(cfg, v, q, pi, X, model, gamma, margin, epochs, batch, lr,
                  pi_anchor, lip_lambda=1e-4, anchor_w=0.1, cons_w=1.0,
                  pi_anchor_w=0.5, seed=0,
                  c2_w=0.0, eta_v=0.03, qv_w=0.0, eta_q=0.015,
                  core_w=0.0, unsafe_w=0.0, mu=0.05, c2_band=0.15, vx_push=0.4,
                  anchor_target=None, pair_w=0.0, eta_pair=0.02):
    """Co-train (V,Q,pi) into a verifiable robust CBF on the brakeable plant.

    Brakeable analogue of qcbf.nets.mlp.finetune_cbf (2-D control + 2-D
    disturbance).  Base terms: decrease hinge (margin m), Q<->V o f consistency,
    V anchor to V_target, pi anchor to the braking witness, weight penalty.

    C2-FIX terms (all OFF when their weight is 0 -> bit-identical to the base
    object; turned on by distill(..., c2_fix=True)).  The Step-0 probe showed
    the witness FEASIBILITY (C3, the Q predicate) is pointwise clean while the
    true successor-value DECREASE (C2, V_theta o f) has real holes -- the
    forbidden Q ~ V o f gap.  These terms attack C2 directly, not the Q-hinge:

      * c2_w  -- DIRECT C2 hinge   relu(eta_v + gamma V(x) - min_d V(f(x,pi_b,d)))
                 backprops into V at the SUCCESSOR (reshapes V o f), not just Q.
      * qv_w  -- one-sided Q-conservatism  relu(Q(x,u,d) - V(f(x,u,d)) + eta_q):
                 drives Q(x,u,d) <= V(f) - eta_q so the runtime filter (which
                 trusts Q) never over-estimates the true successor value.
      * core_w/unsafe_w -- non-vacuity anchors so the C2 push cannot cheat by
                 shrinking {V>=0}: keep V>=mu on the analytic safe core
                 (V_target>=0.2) and V<=-mu where g<=-0.05 (truly unsafe).
    """
    n = len(X)
    rng = np.random.default_rng(seed)
    optV, optQ, optP = Adam(v, lr=lr), Adam(q, lr=lr), Adam(pi, lr=lr)
    dprobe = d_probe_grid(cfg)
    K = len(dprobe)
    Vt = v_target(cfg, X[:, 0], X[:, 1], X[:, 3])   # ground truth for core/unsafe
    # fidelity / TRUST-REGION anchor: V_target in round 0, the PREVIOUS round's
    # V_{k-1}(X) in CEGIS round k (so retraining stays near the last iterate and
    # does not chase a moving boundary -- the Path-2 stabiliser).
    atgt = Vt if anchor_target is None else np.asarray(anchor_target, float)
    gX = model.g(X)
    pb = Progress(epochs, f"cbf m={margin:.2f}")
    hist = {}
    for ep in range(epochs):
        order = rng.permutation(n)
        tot_dec = tot_cons = tot_anch = tot_c2 = tot_qv = tot_pair = 0.0
        for s in range(0, n, batch):
            idx = order[s:s + batch]
            xb = X[idx]
            B = len(idx)
            Vx, zsV, hsV = v.forward(xb[:, [0, 1, 3]], cache=True)
            Vx = Vx.ravel()
            y, zsP, hsP = pi.forward(xb, cache=True)
            u = clip_action(cfg, y)
            dV_x = np.zeros(B)                    # extra per-sample grad on V(x)
            gWv_add = [np.zeros_like(W) for W in v.W]   # successor-side V grads
            gbv_add = [np.zeros_like(b) for b in v.b]

            # ---- decrease hinge: min_d Q(x,u,d_k) vs gamma V + m ---------- #
            qvals = np.empty((B, K))
            for k in range(K):
                dk = np.tile(dprobe[k], (B, 1))
                z = np.concatenate([xb, u, dk], axis=1)
                qvals[:, k] = q(z).ravel()
            kmin = np.argmin(qvals, axis=1)
            mq = qvals[np.arange(B), kmin]
            slack = gamma * Vx - mq + margin
            active = slack > 0.0
            tot_dec += float(np.sum(np.maximum(slack, 0.0)))
            dV_dec = gamma * active.astype(float) / B
            dmin = dprobe[kmin]
            zmin = np.concatenate([xb, u, dmin], axis=1)
            _, zsQm, hsQm = q.forward(zmin, cache=True)
            dQ_dec = (-active.astype(float) / B)[:, None]
            gz = q.input_grad(zmin, np.ones((B, 1)))
            dL_du = np.where(active[:, None], -gz[:, 4:6], 0.0) / B   # u dims 4,5
            # route through the per-component clamp
            lo = np.array([cfg.a_min, -cfg.delta_max])
            hi = np.array([cfg.a_max, cfg.delta_max])
            sat_hi = (y >= hi) & (dL_du < 0)
            sat_lo = (y <= lo) & (dL_du > 0)
            dL_dy = np.where(sat_hi | sat_lo, 0.0, dL_du)
            dL_dy = dL_dy + pi_anchor_w * 2.0 * (y - pi_anchor[idx]) / B

            # ---- DIRECT C2 hinge: min_d V(f(x,pi_b,d)) >= gamma V(x)+eta_v  #
            if c2_w > 0.0:
                Vf_all = np.empty((B, K))
                S_all = np.empty((B, K, 4))
                for k in range(K):
                    dk = np.tile(dprobe[k], (B, 1))
                    Sk = model.step(xb, u, dk)
                    S_all[:, k] = Sk
                    Vf_all[:, k] = v(Sk[:, [0, 1, 3]]).ravel()
                k2 = np.argmin(Vf_all, axis=1)              # worst d for V(f)
                Vfmin = Vf_all[np.arange(B), k2]
                viol = eta_v + gamma * Vx - Vfmin
                # gate to the LIVE in-set boundary shell 0<=V_theta(x)<=c2_band
                # (V_theta drifts during training; the certificate binds on its
                # OWN current boundary, not V_target's fixed one).
                shell = (Vx >= 0.0) & (Vx <= c2_band)
                act2 = (viol > 0.0) & shell
                tot_c2 += float(np.sum(np.maximum(viol, 0.0) * shell))
                dV_x += vx_push * c2_w * gamma * act2 / B   # down-weighted V(x) push
                Smin = S_all[np.arange(B), k2]
                _, zsVf, hsVf = v.forward(Smin[:, [0, 1, 3]], cache=True)
                doutVf = (-c2_w * act2 / B)[:, None]        # -V(successor) term
                gWvf, gbvf, gSin = v.backward(zsVf, hsVf, doutVf)
                for i in range(len(v.W)):
                    gWv_add[i] += gWvf[i]; gbv_add[i] += gbvf[i]
                # route dV/d(v+) -> a=clip(pi)[0] -> pi head dim 0
                vplus = Smin[:, 3]
                notclip = (vplus > 0.0) & (vplus < cfg.v_max)
                dL_da = gSin[:, 2] * cfg.dt * notclip
                a_sat = (y[:, 0] <= cfg.a_min) | (y[:, 0] >= cfg.a_max)
                dL_dy[:, 0] += np.where(a_sat, 0.0, dL_da)

            # ---- PAIRWISE TEMPORAL REGRESSION (structured-V, Path A) ------ #
            # Match the LEARNED worst-d brake contraction to the analytic one +eta:
            #   (V_theta(f) - V_theta(x))  ~  (V*(f) - V*(x)) + eta_pair
            # directly minimising the one-step error variation eps(f)-eps(x) (the
            # binding quantity) -- a smooth pointwise target, no moving boundary,
            # no adversarial buffer.  Worst d for V(f) is d_a=+d_a_max (max v+).
            if pair_w > 0.0:
                dwn = np.array([cfg.d_a_max, 0.0])
                Sw = model.step(xb, u, np.tile(dwn, (B, 1)))
                Vfw, zsVfw, hsVfw = v.forward(Sw[:, [0, 1, 3]], cache=True)
                Vfw = Vfw.ravel()
                dstar = (v_target(cfg, Sw[:, 0], Sw[:, 1], Sw[:, 3]) - Vt[idx]
                         + eta_pair)
                resid_p = (Vfw - Vx) - dstar         # contraction error vs analytic
                tot_pair += float(np.sum(resid_p ** 2))
                gWvp, gbvp, _ = v.backward(zsVfw, hsVfw,
                                           (pair_w * 2.0 * resid_p / B)[:, None])
                for i in range(len(v.W)):
                    gWv_add[i] += gWvp[i]; gbv_add[i] += gbvp[i]
                dV_x += -pair_w * 2.0 * resid_p / B   # through V(x)

            # ---- Q <-> V o f consistency on random (u,d) ----------------- #
            ur = np.column_stack([rng.uniform(cfg.a_min, cfg.a_max, B),
                                  rng.uniform(-cfg.delta_max, cfg.delta_max, B)])
            dr = np.column_stack([rng.uniform(-cfg.d_a_max, cfg.d_a_max, B),
                                  rng.uniform(-cfg.d_delta_max, cfg.d_delta_max, B)])
            zc = np.concatenate([xb, ur, dr], axis=1)
            outQc, zsQc, hsQc = q.forward(zc, cache=True)
            nxt = model.step(xb, ur, dr)
            if qv_w > 0.0:
                Vfr, zsVr, hsVr = v.forward(nxt[:, [0, 1, 3]], cache=True)
                Vfr = Vfr.ravel()
            else:
                Vfr = v(nxt[:, [0, 1, 3]]).ravel()
            resid = outQc.ravel() - Vfr
            tot_cons += float(np.mean(resid ** 2)) * B
            dQ_cons = (cons_w * 2.0 * resid / B)[:, None]

            # ---- one-sided Q-conservatism: Q(x,u,d) <= V(f) - eta_q ------- #
            if qv_w > 0.0:
                act_qv = (resid + eta_q) > 0.0
                tot_qv += float(np.sum(np.maximum(resid + eta_q, 0.0)))
                dQ_cons = dQ_cons + (qv_w * act_qv / B)[:, None]   # push Q down
                gWvr, gbvr, _ = v.backward(                        # push V(f) up
                    zsVr, hsVr, (-qv_w * act_qv / B)[:, None])
                for i in range(len(v.W)):
                    gWv_add[i] += gWvr[i]; gbv_add[i] += gbvr[i]

            # ---- non-vacuity anchors (keep {V>=0} honest, not shrunk) ----- #
            if core_w > 0.0:
                core_m = (Vt[idx] >= 0.2) & (Vx < mu)
                dV_x += -core_w * core_m / B                # push V(x) up to mu
            if unsafe_w > 0.0:
                uns_m = (gX[idx] <= -0.05) & (Vx > -mu)
                dV_x += unsafe_w * uns_m / B                # push V(x) down to -mu

            # ---- V fidelity / trust-region anchor ------------------------ #
            dvres = Vx - atgt[idx]
            tot_anch += float(np.mean(dvres ** 2)) * B
            dV_anch = anchor_w * 2.0 * dvres / B

            gWv, gbv, _ = v.backward(zsV, hsV, (dV_dec + dV_anch + dV_x)[:, None])
            gWqm, gbqm, _ = q.backward(zsQm, hsQm, dQ_dec)
            gWqc, gbqc, _ = q.backward(zsQc, hsQc, dQ_cons)
            gWp, gbp, _ = pi.backward(zsP, hsP, dL_dy)
            for i in range(len(v.W)):
                gWv[i] = gWv[i] + gWv_add[i] + lip_lambda * 2.0 * v.W[i]
                gbv[i] = gbv[i] + gbv_add[i]
            gWq = [gWqm[i] + gWqc[i] + lip_lambda * 2.0 * q.W[i]
                   for i in range(len(q.W))]
            gbq = [gbqm[i] + gbqc[i] for i in range(len(q.b))]
            optV.step(gWv, gbv)
            optQ.step(gWq, gbq)
            optP.step(gWp, gbp)
        hist = {"decr_hinge": tot_dec / n, "cons_mse": tot_cons / n,
                "anchor_mse": tot_anch / n, "c2_hinge": tot_c2 / n,
                "qv_viol": tot_qv / n, "pair_mse": tot_pair / n}
        pb.update(ep + 1)
    pb.done()
    return hist


# --------------------------------------------------------------------------- #
def distill(cfg, seed, margin, n_samples=60_000, reg_epochs=25, cbf_epochs=25,
            batch=1024, lr=1e-3, c2_fix=False, pair_fix=False, cex=None,
            v_ref=None, verbose=True):
    """Build and train (V_t, Q_t, pi_b) for one (seed, margin).  Returns the
    three frozen MLPs plus a diagnostics dict (MSEs, pi agreement).

    ``c2_fix=True`` enables the Step-0 follow-up training (direct C2 hinge +
    one-sided Q-conservatism + non-vacuity anchors, boundary/core/unsafe mix)
    that targets the TRUE V_theta o f decrease the Q-hinge does not constrain.
    ``cex`` optionally injects CEGIS counterexamples into the boundary mix.
    """
    model = BicycleAccelModel(cfg)
    rng = np.random.default_rng(1000 + seed)
    t0 = time.time()
    if verbose:
        print(f"[distill] seed={seed} margin={margin:.2f}: sampling + regression",
              flush=True)

    # ---- regression data ------------------------------------------------- #
    Xv = _sample_states(cfg, n_samples, rng)
    Yv = v_target(cfg, Xv[:, 0], Xv[:, 1], Xv[:, 3])
    Xq = _sample_states(cfg, n_samples, rng)
    Uq = np.column_stack([rng.uniform(cfg.a_min, cfg.a_max, n_samples),
                          rng.uniform(-cfg.delta_max, cfg.delta_max, n_samples)])
    Dq = np.column_stack([rng.uniform(-cfg.d_a_max, cfg.d_a_max, n_samples),
                          rng.uniform(-cfg.d_delta_max, cfg.d_delta_max, n_samples)])
    nxt = model.step(Xq, Uq, Dq)
    Yq = v_target(cfg, nxt[:, 0], nxt[:, 1], nxt[:, 3])
    Xp = _sample_states(cfg, n_samples, rng)
    Yp = np.column_stack([np.full(n_samples, cfg.a_min), racing_steer(cfg, Xp)])

    v, q, pi = make_nets(seed)
    wd = 2e-4                       # weight decay -> low Lipschitz -> tight CROWN
    mse_v = _regress(v, Xv[:, [0, 1, 3]], Yv, reg_epochs, batch, lr, seed, "V", "reg V", wd)
    mse_q = _regress(q, np.concatenate([Xq, Uq, Dq], axis=1), Yq,
                     reg_epochs, batch, lr, seed + 1, "Q", "reg Q", wd)
    mse_p = _regress(pi, Xp, Yp, reg_epochs, batch, lr, seed + 2, "pi", "reg pi", wd)

    # ---- co-train into a verifiable robust CBF --------------------------- #
    if pair_fix:
        # Path A -- pairwise temporal regression: pin the worst-d brake
        # contraction Vth(f)-Vth(x) to the analytic Delta*+eta everywhere (smooth,
        # pointwise; no boundary hinge, no CEGIS).  This attacks eps(f)-eps(x)
        # directly -- the level-independent binding quantity Step-2 isolated.
        Xc = _c2fix_states(cfg, n_samples, rng, cex=cex)
        c2kw = dict(pair_w=1.0, eta_pair=0.02, c2_w=0.0, qv_w=0.2, eta_q=0.01,
                    core_w=0.5, unsafe_w=1.0, mu=0.05, anchor_w=0.05)
    elif c2_fix:
        Xc = _c2fix_states(cfg, n_samples, rng, cex=cex)
        # MODERATE V_target fidelity (anchor_w 0.1 -> 0.08) keeps V_theta's own
        # boundary from running away; the C2 push is gated to the LIVE shell so
        # it tracks the drifting boundary.  core/unsafe anchors keep {V>=0} valid.
        c2kw = dict(c2_w=1.0, eta_v=0.03, qv_w=0.2, eta_q=0.01,
                    core_w=0.5, unsafe_w=1.0, mu=0.05, anchor_w=0.08,
                    c2_band=0.15, vx_push=0.4)
        if v_ref is not None:                        # CEGIS round: trust region
            c2kw["anchor_target"] = v_ref.forward(Xc[:, [0, 1, 3]])[:, 0]
            c2kw["anchor_w"] = 0.15                   # stay near previous iterate
    else:
        Xc = _sample_states(cfg, n_samples, rng, frac_safe=0.85)
        c2kw = {}
    pi_anchor = np.column_stack([np.full(len(Xc), cfg.a_min), racing_steer(cfg, Xc)])
    hist = _finetune_cbf(cfg, v, q, pi, Xc, model, GAMMA, margin,
                         cbf_epochs, batch, lr, pi_anchor, lip_lambda=3e-4,
                         seed=seed, **c2kw)

    # ---- diagnostics: faithfulness of the trained object ----------------- #
    Xt = _sample_states(cfg, 40_000, np.random.default_rng(7))
    Vt = v_target(cfg, Xt[:, 0], Xt[:, 1], Xt[:, 3])
    mse_V = float(np.mean((v(Xt[:, [0, 1, 3]]).ravel() - Vt) ** 2))
    Ut = clip_action(cfg, pi(Xt))
    Dt = np.zeros((len(Xt), 2))
    Qt_pred = q(np.concatenate([Xt, Ut, Dt], axis=1)).ravel()
    nxt_t = model.step(Xt, Ut, Dt)
    Qt_tgt = v_target(cfg, nxt_t[:, 0], nxt_t[:, 1], nxt_t[:, 3])
    mse_Q = float(np.mean((Qt_pred - Qt_tgt) ** 2))
    a_agree = float(np.mean(np.abs(Ut[:, 0] - cfg.a_min) < 0.25))   # braking witness
    d_rmse = float(np.sqrt(np.mean((Ut[:, 1] - racing_steer(cfg, Xt)) ** 2)))

    diag = {"seed": seed, "margin": float(margin),
            "reg_mse_v": mse_v, "reg_mse_q": mse_q, "reg_mse_pi": mse_p,
            "mse_V_target": mse_V, "mse_Q_target": mse_Q,
            "pi_brake_agreement": a_agree, "pi_steer_rmse": d_rmse,
            "decr_hinge": hist.get("decr_hinge", float("nan")),
            "cons_mse": hist.get("cons_mse", float("nan")),
            "anchor_mse": hist.get("anchor_mse", float("nan")),
            "c2_hinge": hist.get("c2_hinge", float("nan")),
            "qv_viol": hist.get("qv_viol", float("nan")),
            "pair_mse": hist.get("pair_mse", float("nan")),
            "wall_s": round(time.time() - t0, 1)}
    if verbose:
        print(f"[distill] seed={seed} m={margin:.2f}: MSE(V)={mse_V:.4f} "
              f"MSE(Q)={mse_Q:.4f} brake-agree={a_agree:.2f} "
              f"steer-rmse={d_rmse:.3f} hinge={diag['decr_hinge']:.4f} "
              f"({diag['wall_s']:.0f}s)", flush=True)
    return v, q, pi, diag


if __name__ == "__main__":     # quick smoke
    cfg = BicycleAccelConfig()
    distill(cfg, seed=0, margin=0.10, n_samples=20_000, reg_epochs=8, cbf_epochs=8)
