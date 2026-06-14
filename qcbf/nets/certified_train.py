"""Verifier-in-the-loop certified training of Q_theta (two-sided CROWN-IBP).

This trains Q_theta to satisfy, at cell-worst, BOTH Q-CBF gate conditions on the
SAME sound interval bounds the verifier checks -- it shapes the network to satisfy
the proof (what Theorem A needs), it is NOT a Lipschitz/flatten penalty and does
not change the object's semantics:

    C4:  ub Q_theta(C,u,D)        <=  lb V_theta(f(C,u,D)) - margin   (push DOWN)
    C3:  min_d lb Q_theta(C,u*,D) >=  gamma_deploy * ubV(C) + eps     (push UP)

for every menu action (C4) and the best valid menu action u* (C3).

Soundness for the deployed CROWN verifier:
  * IBP ub >= CROWN ub, so driving ub_IBP(Q) <= lb(Vf) - margin implies
    ub_CROWN(Q) <= lb_CROWN(Vf), i.e. the verifier's C4 passes;
  * IBP lb <= CROWN lb, so driving lb_IBP(Q) >= gamma*ubV + eps implies
    lb_CROWN(Q) >= gamma*ubV + eps, i.e. the verifier's C3 passes.
V_theta is frozen, so its successor lower bound is a CONSTANT table precomputed
once with the verifier's own CROWN routine.  Training is never in the TCB.

Pure IBP at the full cell width is far too loose for these net sizes (initial
cell-worst residual ~11 vs CROWN ~0.4), so a one-shot push collapses Q.  We
therefore ramp the certified box from a point (eps=0, exact) to the full cell
(eps=1, the deployed condition) over a warmup.  The one-sided C4 push alone
certifies C4 (~99%) but crushes Q below the gate and kills C3; the C3 up-term is
the counter-pressure that squeezes Q into the band [gamma*ubV+eps, lb(Vf)-margin].
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


def _eps_q_box(cb, u, k, d_edges, eps):
    """eps-scaled Q-input box [qlo, qhi] for menu action `u` and d-subbox `k`.

    Built around each cell/d-subbox CENTER and scaled by `eps` in [0, 1]:
      eps = 0 -> the degenerate (exact) center point,
      eps = 1 -> EXACTLY the full state cell x d-subbox the deployed verifier
                 checks (center +- half-width reproduces the original edges).
    `u` is a fixed menu action, so the control axis stays degenerate."""
    B = len(cb)
    cx = 0.5 * (cb[:, 0] + cb[:, 1]); hx = 0.5 * (cb[:, 1] - cb[:, 0])
    cy = 0.5 * (cb[:, 2] + cb[:, 3]); hy = 0.5 * (cb[:, 3] - cb[:, 2])
    cp = 0.5 * (cb[:, 4] + cb[:, 5]); hp = 0.5 * (cb[:, 5] - cb[:, 4])
    dc = 0.5 * (d_edges[k] + d_edges[k + 1]); dh = 0.5 * (d_edges[k + 1] - d_edges[k])
    uu = np.full(B, u)
    dlo = np.full(B, dc - eps * dh); dhi = np.full(B, dc + eps * dh)
    qlo = np.column_stack([cx - eps * hx, cy - eps * hy, cp - eps * hp, uu, dlo])
    qhi = np.column_stack([cx + eps * hx, cy + eps * hy, cp + eps * hp, uu, dhi])
    return qlo, qhi


def _cellworst_c4_viol(q, cb_all, lbVf, fin, menu, d_edges, nd, c4_margin):
    """Mean positive cell-worst C4 violation at eps=1 -- i.e. against the box the
    deployed CROWN verifier actually checks (ub_IBP >= ub_CROWN, so this is a
    sound upper bound on the verifier's own residual).  Used only for the
    honest before/after numbers in the report, not for gradients."""
    P = len(cb_all)
    tot = 0.0
    for j, u in enumerate(menu):
        for k in range(nd):
            qlo, qhi = _eps_q_box(cb_all, float(u), k, d_edges, 1.0)
            _, ubQ, _ = ibp_forward(q, qlo, qhi)
            lvjk = lbVf[:, j, k]
            active = fin[:, j, k] & (c4_margin - (lvjk - ubQ[:, 0]) > 0)
            tot += float(np.sum(np.where(active, c4_margin - (lvjk - ubQ[:, 0]), 0.0)))
    return tot / max(P, 1) / (len(menu) * nd)


def _gate_health(q, v, xc, menu, d_edges, gamma):
    """C3 collapse watch: best-menu robust gate margin at cell centers,
        max_u min_d Q(xc, u, d) - gamma * V(xc).
    A pointwise proxy (centers, d on the sub-edges) -- NOT a certificate, just a
    cheap signal that some action stays feasible (so the cert push has not
    crushed Q below the gate and broken C3).  Returns (mean, frac>=0)."""
    B = len(xc)
    Vx = v(xc).ravel()
    best = np.full(B, -np.inf)
    for u in menu:
        mind = np.full(B, np.inf)
        for d in d_edges:
            z = np.column_stack([xc, np.full(B, u), np.full(B, d)])
            mind = np.minimum(mind, q(z).ravel())
        best = np.maximum(best, mind)
    margin = best - gamma * Vx
    return float(margin.mean()), float(np.mean(margin >= 0.0))


def _menu_gate_feas(q, cb_all, gate_thresh, fin, menu, d_edges, nd):
    """Certified menu-gate feasibility at eps=1 (the deployed box): for the best
    DOMAIN-VALID menu action of each cell,

        best_u [ min_d lb_IBP Q(C, u, d) ]  vs  gate_thresh (= gamma*ubV + eps).

    IBP lb <= CROWN lb, so best_u min_d lb_IBP Q >= gate_thresh is a SOUND lower
    bound on the verifier's C3 feasibility (some menu action clears the gate at
    cell-worst).  An action is only counted if every d-subbox stays in-domain
    (else the runtime gate refuses it).  Returns (mean best-gate margin, frac
    feasible)."""
    P = len(cb_all)
    best = np.full(P, -np.inf)
    for j, u in enumerate(menu):
        validu = fin[:, j, :].all(axis=1)
        mind = np.full(P, np.inf)
        for k in range(nd):
            qlo, qhi = _eps_q_box(cb_all, float(u), k, d_edges, 1.0)
            lbQ, _, _ = ibp_forward(q, qlo, qhi)
            mind = np.minimum(mind, lbQ[:, 0])
        best = np.maximum(best, np.where(validu, mind, -np.inf))
    finite = np.isfinite(best)
    margin = best - gate_thresh
    mean_margin = float(margin[finite].mean()) if finite.any() else float("nan")
    feas = finite & (margin >= 0.0)
    return mean_margin, float(feas.mean())


def train_q_certified(q, boxes, pool, lbVf, dyn, menu, d_subsplit,
                      c4_w, c4_margin, anchor_w, epochs, batch, lr,
                      v=None, model=None, seed=0, verbose=True,
                      eps_start=0.0, eps_warmup_frac=0.5,
                      gamma_deploy=None, c3_w=0.0, gate_thresh=None) -> dict:
    """Two-sided CROWN-IBP eps-schedule training of the cell-worst Q-CBF gates.

    C4 (push DOWN the upper bound), for every menu action and d-subbox:
        ub_IBP Q(C, u, D) <= lb_CROWN V(f(C, u, D)) - c4_margin     (V frozen).
    C3 (push UP the lower bound), for the best DOMAIN-VALID menu action per cell:
        min_d lb_IBP Q(C, u*, D) >= gate_thresh = gamma_deploy * ubV(C) + eps.

    Both sides are sound for the deployed CROWN verifier: IBP ub >= CROWN ub (so
    the trained C4 implies the verifier's C4), and IBP lb <= CROWN lb (so the
    trained C3 implies the verifier's C3).  Training is never in the TCB.

    Together they SQUEEZE Q into the band [gate_thresh, lb V(f) - c4_margin]: C4
    alone (the one-sided lever) certifies C4 at ~99% but crushes Q below the gate
    and kills C3; the C3 up-term keeps at least one menu action feasible.  The
    band is non-empty for a cell iff lb V(f) - gamma*ubV >= c4_margin + eps, i.e.
    the V-side decrease margin clears the combined cell slack -- so where C3 and
    C4 still cannot both hold, the binding term is V's slack, not Q's.

    Pure IBP at full cell width is far too loose (initial cell-worst residual ~11
    vs CROWN ~0.4), so the certified box is ramped from a point (eps=0, exact) to
    the full cell (eps=1, the deployed condition) over the first `eps_warmup_frac`
    of epochs; at eps=1 `_eps_q_box` reproduces the cell x d-subbox exactly.  A
    light anchor keeps Q(center) ~ V(f(center)).  `c3_w=0` recovers the one-sided
    C4-only lever.

    Reports init/final cell-worst C4 violation and certified menu-gate (C3)
    feasibility, both at eps=1, plus a pointwise C3 gate-health proxy."""
    if c3_w > 0.0 and gate_thresh is None:
        raise ValueError("two-sided C3 training (c3_w>0) requires gate_thresh")
    rng = np.random.default_rng(seed)
    opt = Adam(q, lr=lr)
    cb_all = boxes[pool]
    P = len(pool)
    nu = len(menu)
    nd = d_subsplit
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, nd + 1)
    xc_all = np.column_stack([0.5 * (cb_all[:, 0] + cb_all[:, 1]),
                              0.5 * (cb_all[:, 2] + cb_all[:, 3]),
                              0.5 * (cb_all[:, 4] + cb_all[:, 5])])
    fin = np.isfinite(lbVf)                       # (P,nu,nd)
    warm = max(1, int(round(eps_warmup_frac * epochs)))
    sc_c4 = 1.0 / (nu * nd)                        # C4 fires on all nu*nd actions
    two_sided = c3_w > 0.0 and gate_thresh is not None   # gates the C3 GRADIENT
    report_feas = gate_thresh is not None               # gates C3 REPORTING only

    # honest before-numbers at the deployed box (eps=1)
    init_cw = _cellworst_c4_viol(q, cb_all, lbVf, fin, menu, d_edges, nd, c4_margin)
    watch = gamma_deploy is not None and v is not None
    gm_i = _gate_health(q, v, xc_all, menu, d_edges, gamma_deploy) if watch else (None, None)
    feas_i = (_menu_gate_feas(q, cb_all, gate_thresh, fin, menu, d_edges, nd)
              if report_feas else (None, None))

    t0 = time.time()
    for ep in range(epochs):
        eps = min(1.0, eps_start + (1.0 - eps_start) * ep / warm)
        order = rng.permutation(P)
        tot_c4, tot_c3, cnt = 0.0, 0.0, 0
        for s in range(0, P, batch):
            sl = order[s:s + batch]
            cb = cb_all[sl]; xc = xc_all[sl]; lv = lbVf[sl]; fn = fin[sl]
            B = len(sl); cnt += B
            gt = gate_thresh[sl] if two_sided else None
            gW = [np.zeros_like(W) for W in q.W]
            gB = [np.zeros_like(b) for b in q.b]

            # ---- pass 1: forward every (u, d-subbox), store bounds + caches ---
            LB = np.empty((B, nu, nd)); UB = np.empty((B, nu, nd))
            caches = {}
            for j, u in enumerate(menu):
                for k in range(nd):
                    qlo, qhi = _eps_q_box(cb, float(u), k, d_edges, eps)
                    lbQ, ubQ, cQ = ibp_forward(q, qlo, qhi)
                    LB[:, j, k] = lbQ[:, 0]; UB[:, j, k] = ubQ[:, 0]
                    caches[(j, k)] = cQ

            # ---- C3 target: best valid menu action's worst-d lower bound ------
            if two_sided:
                valid_u = fn.all(axis=2)                       # (B,nu)
                lb_mind = np.where(valid_u, LB.min(axis=2), -np.inf)   # (B,nu)
                u_star = lb_mind.argmax(axis=1)                # (B,)
                has_valid = valid_u.any(axis=1)
                best_lb = lb_mind[np.arange(B), u_star]
                c3_active = has_valid & (best_lb < gt)
                k_star = LB[np.arange(B), u_star, :].argmin(axis=1)    # worst d
                tot_c3 += float(np.sum(np.where(c3_active, gt - best_lb, 0.0)))

            # ---- pass 2: backward C4 (all) + C3 (selected) per (u, d) --------
            for j, u in enumerate(menu):
                for k in range(nd):
                    ubj = UB[:, j, k]; lvjk = lv[:, j, k]
                    c4_act = fn[:, j, k] & (c4_margin - (lvjk - ubj) > 0)
                    tot_c4 += float(np.sum(np.where(c4_act,
                                    c4_margin - (lvjk - ubj), 0.0)))
                    d_ub = (np.where(c4_act, c4_w, 0.0) / B)[:, None]
                    gWc, gBc = ibp_backward(q, caches[(j, k)],
                                            np.zeros_like(d_ub), d_ub)
                    for i in range(len(gW)):
                        gW[i] += gWc[i] * sc_c4; gB[i] += gBc[i] * sc_c4
                    if two_sided:
                        sel = c3_active & (u_star == j) & (k_star == k)
                        if sel.any():
                            d_lb = (np.where(sel, -c3_w, 0.0) / B)[:, None]
                            gWl, gBl = ibp_backward(q, caches[(j, k)],
                                                    d_lb, np.zeros_like(d_lb))
                            for i in range(len(gW)):
                                gW[i] += gWl[i]; gB[i] += gBl[i]
                    if anchor_w > 0.0 and v is not None:
                        dmid = 0.5 * (d_edges[k] + d_edges[k + 1])
                        nxt = model.step(xc, np.full(B, u), np.full(B, dmid))
                        vf = v(nxt).reshape(B, 1)
                        z = np.column_stack([xc, np.full(B, u), np.full(B, dmid)])
                        out, zs, hs = q.forward(z, cache=True)
                        gWa, gBa, _ = q.backward(zs, hs,
                                                 anchor_w * 2 * (out - vf) / B * sc_c4)
                        for i in range(len(gW)):
                            gW[i] += gWa[i]; gB[i] += gBa[i]
            opt.step(gW, gB)
        mc4 = tot_c4 / max(cnt, 1) / (nu * nd)
        mc3 = tot_c3 / max(cnt, 1)
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [Qcert] epoch {ep:3d}  eps={eps:.2f}  "
                  f"C4-viol={mc4:.5f}  C3-viol={mc3:.5f}")

    # honest after-numbers at the deployed box (eps=1)
    final_cw = _cellworst_c4_viol(q, cb_all, lbVf, fin, menu, d_edges, nd, c4_margin)
    gm_f = _gate_health(q, v, xc_all, menu, d_edges, gamma_deploy) if watch else (None, None)
    feas_f = (_menu_gate_feas(q, cb_all, gate_thresh, fin, menu, d_edges, nd)
              if report_feas else (None, None))
    return {"pool": int(P),
            "eps_start": float(eps_start), "eps_warmup_frac": float(eps_warmup_frac),
            "c3_w": float(c3_w),
            "init_cert_c4_viol": float(init_cw),
            "final_cert_c4_viol": float(final_cw),
            "menu_gate_feas_init": feas_i[1], "menu_gate_margin_init": feas_i[0],
            "menu_gate_feas_final": feas_f[1], "menu_gate_margin_final": feas_f[0],
            "gate_margin_mean_init": gm_i[0], "gate_frac_ge0_init": gm_i[1],
            "gate_margin_mean_final": gm_f[0], "gate_frac_ge0_final": gm_f[1],
            "wall_s": time.time() - t0}


# --------------------------------------------------------------------------- #
# Cell-worst certified training of V (the V analog of train_q_certified).
#
# V is the one piece still trained on POINTWISE hinges (train_v_cbf), so its
# CELL-WORST slack is the binding barrier: ub V leaks {V>=0} outside K (C1) and
# lb V(f) is too low for the witness band (C4).  This trains V on its own sound
# cell-worst IBP bounds, the same verifier-in-loop way:
#   C1  (push DOWN ub):  ub_IBP V(C)        <  -c1_margin     on cells with g<0
#   dec (push UP   lb):  min_d lb_IBP V(f(C,u*,D)) >= gamma*ub_IBP V(C) + dec_margin
# Soundness: IBP ub >= CROWN ub and IBP lb <= CROWN lb, so ub_IBP V<0 => the
# verifier's ub_CROWN V<0 (C1 holds), and lb_IBP V(f) >= gamma*ub_IBP V(C)+m =>
# lb_CROWN V(f) >= gamma*ub_CROWN V(C)+m (the band the witness C3/C4 needs).  A
# teacher anchor V(center)~V_HJ(center) keeps {V>=0} from collapsing.  eps ramps
# the boxes from a point to the full cell.  The successor uses the PRIMARY box b1
# only (the wrap box b2 is dropped from the gradient -- a sound training
# approximation, since the deployed verifier re-checks the true b1+b2 cell-worst).
# --------------------------------------------------------------------------- #
def _box_in_domain_local(dyn, b):
    return ((b[:, 0] >= dyn.p_lo - 1e-12) & (b[:, 1] <= dyn.p_hi + 1e-12)
            & (b[:, 2] >= dyn.p_lo - 1e-12) & (b[:, 3] <= dyn.p_hi + 1e-12))


def _eps_state_box(cb, eps):
    """eps-scaled raw state cell [px,py,psi] box around its center (for V over C)."""
    cx = 0.5 * (cb[:, 0] + cb[:, 1]); hx = 0.5 * (cb[:, 1] - cb[:, 0])
    cy = 0.5 * (cb[:, 2] + cb[:, 3]); hy = 0.5 * (cb[:, 3] - cb[:, 2])
    cp = 0.5 * (cb[:, 4] + cb[:, 5]); hp = 0.5 * (cb[:, 5] - cb[:, 4])
    lo = np.column_stack([cx - eps * hx, cy - eps * hy, cp - eps * hp])
    hi = np.column_stack([cx + eps * hx, cy + eps * hy, cp + eps * hp])
    return lo, hi


def _eps_cell_ranges(cb, eps):
    """eps-scaled raw cell edge ranges (px_lo,px_hi,py_lo,py_hi,psi_lo,psi_hi)."""
    cx = 0.5 * (cb[:, 0] + cb[:, 1]); hx = 0.5 * (cb[:, 1] - cb[:, 0])
    cy = 0.5 * (cb[:, 2] + cb[:, 3]); hy = 0.5 * (cb[:, 3] - cb[:, 2])
    cp = 0.5 * (cb[:, 4] + cb[:, 5]); hp = 0.5 * (cb[:, 5] - cb[:, 4])
    return (cx - eps * hx, cx + eps * hx, cy - eps * hy, cy + eps * hy,
            cp - eps * hp, cp + eps * hp)


def _v_succ_lb_b1(v, cb, dyn, u, dlo, dhi, eps):
    """lb_IBP V over the PRIMARY successor box of the eps-scaled cell, plus its
    cache and in-domain mask.  -inf where the successor leaves the domain."""
    pxl, pxh, pyl, pyh, pll, plh = _eps_cell_ranges(cb, eps)
    b1, _, _ = successor_boxes(dyn, pxl, pxh, pyl, pyh, pll, plh,
                               u, u, dlo, dhi)
    dom = _box_in_domain_local(dyn, b1)
    lbf, _, cache = ibp_forward(v, b1[:, [0, 2, 4]], b1[:, [1, 3, 5]])
    return np.where(dom, lbf[:, 0], -np.inf), cache, dom


def _v_cellworst_report(v, cb_all, unsafe, dyn, menu, d_edges, nd, gamma,
                        c1_margin, dec_margin):
    """Honest eps=1 before/after numbers: C1 leak fraction (ub_IBP V>=0 on cells
    with g<0) and band-open fraction (some valid action's min_d lb_IBP V(f) >=
    gamma*ub_IBP V(C) + dec_margin)."""
    lo, hi = _eps_state_box(cb_all, 1.0)
    _, ubV, _ = ibp_forward(v, lo, hi)
    ubV = ubV[:, 0]
    c1_leak = float(np.mean((ubV >= 0.0)[unsafe])) if unsafe.any() else 0.0
    P = len(cb_all)
    best = np.full(P, -np.inf)
    for u in menu:
        mind = np.full(P, np.inf)
        for k in range(nd):
            lbf, _, _ = _v_succ_lb_b1(v, cb_all, dyn, float(u),
                                      d_edges[k], d_edges[k + 1], 1.0)
            mind = np.minimum(mind, lbf)
        best = np.maximum(best, mind)
    band_open = float(np.mean(best >= gamma * ubV + dec_margin))
    return c1_leak, band_open


def train_v_certified(v, boxes, pool, dyn, menu, d_subsplit, gamma,
                      c1_w, c1_margin, dec_w, dec_margin, anchor_w,
                      epochs, batch, lr, gmin_pool, anchor_Y,
                      seed=0, verbose=True, eps_start=0.0,
                      eps_warmup_frac=0.5) -> dict:
    """Cell-worst CROWN-IBP training of V (C1 floor + decrease band).  See the
    block comment above for the conditions and soundness.  `gmin_pool` is the
    exact g lower bound per pool cell (C1 only fires where g<0); `anchor_Y` is
    V_HJ at the pool cell centers (teacher anchor against collapse)."""
    rng = np.random.default_rng(seed)
    opt = Adam(v, lr=lr)
    cb_all = boxes[pool]
    P = len(pool)
    nu = len(menu)
    nd = d_subsplit
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, nd + 1)
    xc_all = np.column_stack([0.5 * (cb_all[:, 0] + cb_all[:, 1]),
                              0.5 * (cb_all[:, 2] + cb_all[:, 3]),
                              0.5 * (cb_all[:, 4] + cb_all[:, 5])])
    unsafe_all = np.asarray(gmin_pool, float) < 0.0
    aY_all = np.asarray(anchor_Y, float).reshape(P, 1)
    warm = max(1, int(round(eps_warmup_frac * epochs)))
    # NB: unlike train_q_certified (C4 sums over all nu*nd actions), every term
    # here -- C1, the decrease on the selected action, the anchor -- fires ONCE
    # per cell, so none is averaged over nu*nd.

    c1_i, band_i = _v_cellworst_report(v, cb_all, unsafe_all, dyn, menu,
                                       d_edges, nd, gamma, c1_margin, dec_margin)
    t0 = time.time()
    for ep in range(epochs):
        eps = min(1.0, eps_start + (1.0 - eps_start) * ep / warm)
        order = rng.permutation(P)
        tot_c1, tot_dec, cnt = 0.0, 0.0, 0
        for s in range(0, P, batch):
            sl = order[s:s + batch]
            cb = cb_all[sl]; xc = xc_all[sl]; un = unsafe_all[sl]; ay = aY_all[sl]
            B = len(sl); cnt += B
            gW = [np.zeros_like(W) for W in v.W]
            gB = [np.zeros_like(b) for b in v.b]

            # ---- V over the cell C: ub (C1 + decrease threshold), cache --------
            clo, chi = _eps_state_box(cb, eps)
            _, ubVc, cC = ibp_forward(v, clo, chi)
            ubVc = ubVc[:, 0]
            c1_act = un & (ubVc + c1_margin > 0.0)
            tot_c1 += float(np.sum(np.where(c1_act, ubVc + c1_margin, 0.0)))

            # ---- decrease pass-1: lb V(f) over successors, best valid action ---
            LBf = np.full((B, nu, nd), -np.inf)
            cF = {}
            for j, u in enumerate(menu):
                for k in range(nd):
                    lbf, cache, _ = _v_succ_lb_b1(v, cb, dyn, float(u),
                                                  d_edges[k], d_edges[k + 1], eps)
                    LBf[:, j, k] = lbf
                    cF[(j, k)] = cache
            lbf_mind = LBf.min(axis=2)                     # (B,nu) min over d
            u_star = lbf_mind.argmax(axis=1)               # best decreasing action
            best = lbf_mind[np.arange(B), u_star]
            k_star = LBf[np.arange(B), u_star, :].argmin(axis=1)
            dec_act = np.isfinite(best) & (gamma * ubVc + dec_margin - best > 0.0)
            tot_dec += float(np.sum(np.where(dec_act,
                             gamma * ubVc + dec_margin - best, 0.0)))

            # ---- backward: C-box ub (C1 down + decrease-threshold down) --------
            d_ubC = (np.where(c1_act, c1_w, 0.0)
                     + np.where(dec_act, dec_w * gamma, 0.0)) / B
            gWc, gBc = ibp_backward(v, cC, np.zeros((B, 1)), d_ubC[:, None])
            for i in range(len(gW)):
                gW[i] += gWc[i]; gB[i] += gBc[i]

            # ---- backward: successor lb of the selected action (decrease up) ---
            for j in range(nu):
                for k in range(nd):
                    sel = dec_act & (u_star == j) & (k_star == k)
                    if sel.any():
                        d_lb = (np.where(sel, -dec_w, 0.0) / B)[:, None]
                        gWl, gBl = ibp_backward(v, cF[(j, k)], d_lb,
                                                np.zeros_like(d_lb))
                        for i in range(len(gW)):
                            gW[i] += gWl[i]; gB[i] += gBl[i]

            # ---- teacher anchor: V(center) ~ V_HJ(center) (against collapse) ---
            if anchor_w > 0.0:
                out, zs, hs = v.forward(xc, cache=True)
                gWa, gBa, _ = v.backward(zs, hs, anchor_w * 2 * (out - ay) / B)
                for i in range(len(gW)):
                    gW[i] += gWa[i]; gB[i] += gBa[i]
            opt.step(gW, gB)
        mc1 = tot_c1 / max(cnt, 1)
        mdec = tot_dec / max(cnt, 1) / (nu * nd)
        if verbose and (ep % max(1, epochs // 8) == 0 or ep == epochs - 1):
            print(f"  [Vcert] epoch {ep:3d}  eps={eps:.2f}  "
                  f"C1-leak-viol={mc1:.5f}  dec-viol={mdec:.5f}")

    c1_f, band_f = _v_cellworst_report(v, cb_all, unsafe_all, dyn, menu,
                                       d_edges, nd, gamma, c1_margin, dec_margin)
    return {"pool": int(P), "eps_start": float(eps_start),
            "eps_warmup_frac": float(eps_warmup_frac),
            "c1_leak_frac_init": float(c1_i), "c1_leak_frac_final": float(c1_f),
            "band_open_frac_init": float(band_i), "band_open_frac_final": float(band_f),
            "wall_s": time.time() - t0}
