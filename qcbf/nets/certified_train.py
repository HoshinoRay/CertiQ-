"""Verifier-in-the-loop (IBP) certified training of Q_theta.

This trains Q_theta to satisfy the cell-worst C4 proof condition

    lb V_theta(f(C, u, D))  -  ub Q_theta(C, u, D)  >=  margin

on the SAME sound interval bounds the verifier checks.  It shapes the network to
satisfy the Q-CBF proof at cell-worst (what Theorem A needs) -- it is NOT a
Lipschitz/flatten penalty and does not change the object's semantics.

Soundness for the deployed CROWN verifier: IBP is looser than CROWN, so
ub_IBP(Q) >= ub_CROWN(Q).  Driving ub_IBP(Q) <= lb(V f) - margin therefore
implies ub_CROWN(Q) <= lb(V f) - margin <= lb_CROWN(V f), i.e. the verifier's
C4 passes.  V_theta is frozen here, so its successor lower bound is a CONSTANT
table, precomputed once with the verifier's own CROWN routine.
"""
from __future__ import annotations

import time

import numpy as np

from qcbf.nets.mlp import Adam, ibp_forward, ibp_backward
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.dynamics.dubins import successor_boxes


def _frozen_v_succ_lower(v_seq, cb, dyn, chunk, u, dlo, dhi):
    """CROWN lower bound of  min_{d in [dlo,dhi]} V(f(cb, u, d))  (V frozen).
    -inf where any successor box leaves the position domain (then C4 is enforced
    trivially -- the runtime gate also refuses such actions)."""
    b1, b2, m2 = successor_boxes(dyn, cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3],
                                 cb[:, 4], cb[:, 5], u, u, dlo, dhi)

    def box_lb(b):
        dom = ((b[:, 0] >= dyn.p_lo - 1e-12) & (b[:, 1] <= dyn.p_hi + 1e-12)
               & (b[:, 2] >= dyn.p_lo - 1e-12) & (b[:, 3] <= dyn.p_hi + 1e-12))
        lo, hi = b[:, [0, 2, 4]], b[:, [1, 3, 5]]
        lb, _ = crown_bounds_chunked(v_seq, lo, hi, True, chunk)
        return np.where(dom, lb[:, 0], -np.inf)

    val = box_lb(b1)
    if np.any(m2):
        val = np.minimum(val, np.where(m2, box_lb(b2), np.inf))
    return val


def precompute_lbVf(v, boxes, pool, dyn, menu, d_subsplit, chunk):
    """(P, nu, nd) CROWN lower bounds of min_d V(f(cell,u,d-subbox)), V frozen."""
    v_seq = SeqNet.from_mlp(v)
    cb = boxes[pool]
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, d_subsplit + 1)
    out = np.empty((len(pool), len(menu), d_subsplit))
    for j, u in enumerate(menu):
        for k in range(d_subsplit):
            out[:, j, k] = _frozen_v_succ_lower(
                v_seq, cb, dyn, chunk, float(u), d_edges[k], d_edges[k + 1])
    return out


def train_q_certified(q, boxes, pool, lbVf, dyn, menu, d_subsplit,
                      c4_w, c4_margin, anchor_w, epochs, batch, lr,
                      v=None, model=None, seed=0, verbose=True) -> dict:
    """Push ub_IBP(Q(cell,u,d-subbox)) <= lbVf - c4_margin on the cell pool.

    `lbVf` is the precomputed (P,nu,nd) frozen-V successor lower bound.  A light
    anchor keeps Q(center,u,dmid) ~ V(f(center)) so it does not collapse far
    below the successor value (which would needlessly hurt the C3 gate).  Returns
    a small report (initial/final mean certified-C4 violation over the pool)."""
    rng = np.random.default_rng(seed)
    opt = Adam(q, lr=lr)
    cb_all = boxes[pool]
    P = len(pool)
    nd = d_subsplit
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, nd + 1)
    xc_all = np.column_stack([0.5 * (cb_all[:, 0] + cb_all[:, 1]),
                              0.5 * (cb_all[:, 2] + cb_all[:, 3]),
                              0.5 * (cb_all[:, 4] + cb_all[:, 5])])
    fin = np.isfinite(lbVf)                       # (P,nu,nd)
    t0 = time.time()
    init_viol = None
    for ep in range(epochs):
        order = rng.permutation(P)
        tot_viol, cnt = 0.0, 0
        for s in range(0, P, batch):
            sl = order[s:s + batch]
            cb = cb_all[sl]; xc = xc_all[sl]; lv = lbVf[sl]; fn = fin[sl]
            B = len(sl); cnt += B
            gW = [np.zeros_like(W) for W in q.W]
            gB = [np.zeros_like(b) for b in q.b]
            for j, u in enumerate(menu):
                for k in range(nd):
                    qlo = np.column_stack([cb[:, 0], cb[:, 2], cb[:, 4],
                                           np.full(B, u), np.full(B, d_edges[k])])
                    qhi = np.column_stack([cb[:, 1], cb[:, 3], cb[:, 5],
                                           np.full(B, u), np.full(B, d_edges[k + 1])])
                    _, ubQ, cQ = ibp_forward(q, qlo, qhi)
                    lvjk = lv[:, j, k]
                    active = fn[:, j, k] & (c4_margin - (lvjk - ubQ[:, 0]) > 0)
                    tot_viol += float(np.sum(np.where(active,
                                      c4_margin - (lvjk - ubQ[:, 0]), 0.0)))
                    d_ub = (np.where(active, c4_w, 0.0) / B)[:, None]
                    gWc, gBc = ibp_backward(q, cQ, np.zeros_like(d_ub), d_ub)
                    for i in range(len(gW)):
                        gW[i] += gWc[i]; gB[i] += gBc[i]
                    if anchor_w > 0.0 and v is not None:
                        dmid = 0.5 * (d_edges[k] + d_edges[k + 1])
                        nxt = model.step(xc, np.full(B, u), np.full(B, dmid))
                        vf = v(nxt).reshape(B, 1)
                        z = np.column_stack([xc, np.full(B, u), np.full(B, dmid)])
                        out, zs, hs = q.forward(z, cache=True)
                        gWa, gBa, _ = q.backward(zs, hs, anchor_w * 2 * (out - vf) / B)
                        for i in range(len(gW)):
                            gW[i] += gWa[i]; gB[i] += gBa[i]
            sc = 1.0 / (len(menu) * nd)
            opt.step([g * sc for g in gW], [g * sc for g in gB])
        mean_viol = tot_viol / max(cnt, 1) / (len(menu) * nd)
        if init_viol is None:
            init_viol = mean_viol
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [Qcert] epoch {ep:3d}  mean cert-C4 viol = {mean_viol:.5f}")
    return {"pool": int(P), "init_cert_c4_viol": float(init_viol),
            "final_cert_c4_viol": float(mean_viol), "wall_s": time.time() - t0}
