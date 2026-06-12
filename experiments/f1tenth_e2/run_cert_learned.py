"""F1TENTH E2-L -- the HEADLINE run: sound Gate-D certification of the LEARNED
filter (V_t, Q_t, pi_b) on the brakeable 4-state plant, via Theorem A's
(C1)-(C3) verified DIRECTLY on the networks with the true f.

This is the first test of the actual project claim: certify a *learned* barrier
object, with NO approximation assumption (the analytic E2 run bypassed the
network verifier; this one exercises it).  PASS and FAIL are equally valid,
publishable outcomes -- the deliverable is an attributable fact, not a green.

Soundness firewall (see the spec in the run prompt):
  * (C1)-(C3) are checked on V_t, Q_t, pi_b with the TRUE dynamics; the
    distillation target (distill.py) is a TRAINING choice for non-vacuity,
    never a proof step.  No V_t ~ V_target, no ||Q-Q*|| anywhere below.
  * C3  (witness feasibility):  min_d Q_t(x, clip(pi_b(x)), d) - gamma V_t(x)
        >= eps,  bounded by CROWN -- pi_b CROWN'd to an action box, then Q_t
        CROWN'd over cell x action-box x D (decoupled but SOUND; the compiled
        correlation-preserving route is a tightening, not a soundness need).
  * skip (Lemma E):  a (cell, accel-cell) is vacuous if a sound UPPER bound of
        sup_u min_d Q_t - gamma V_t is < 0 (no runtime-admissible action there).
  * C2  (robust closure):  for every non-skipped accel-cell, the CROWN LOWER
        bound of V_t over the interval successor of cell x heading-cell x
        accel-cell x D is >= c (the successor stays in {V_t >= c}).
  Every cell is decided by a named backend; "unknown" => not certified.

The certified set is heading-INCLUSIVE: clearance(p) is a distance function
(||grad||=1 a.e.), so the worst-case-over-heading braking contraction is exactly
zero -- a heading-free learned cert is a knife-edge that approximation breaks.
The heading split (npsi) is the resolution knob the attribution sweeps.

    python experiments/f1tenth_e2/run_cert_learned.py --quick      # smoke
    python experiments/f1tenth_e2/run_cert_learned.py              # full sweep
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         brake_cbf_bounds, g_bounds_sq,
                                         successor_box)
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.util.progress import Progress
from experiments.f1tenth_e2.distill import GAMMA, distill, racing_steer


# --------------------------------------------------------------------------- #
# Base (px, py, v) grid: V_t CROWN bounds + exact C1.  Heading-independent.
# --------------------------------------------------------------------------- #
def base_grid(cfg, v_net, npx, npy, nv, chunk=4096, verbose=True):
    px = np.linspace(cfg.p_lo, cfg.p_hi, npx + 1)
    py = np.linspace(cfg.p_lo, cfg.p_hi, npy + 1)
    vv = np.linspace(0.0, cfg.v_max, nv + 1)
    PXl, PYl, Vl = np.meshgrid(px[:-1], py[:-1], vv[:-1], indexing="ij")
    PXh, PYh, Vh = np.meshgrid(px[1:], py[1:], vv[1:], indexing="ij")
    lo = np.column_stack([PXl.ravel(), PYl.ravel(), Vl.ravel()])
    hi = np.column_stack([PXh.ravel(), PYh.ravel(), Vh.ravel()])
    lbV, ubV = crown_bounds_chunked(v_net, lo, hi, True, chunk,
                                    progress="V-bounds" if verbose else None)
    lbV, ubV = lbV[:, 0], ubV[:, 0]
    gmin = g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])
    # analytic (true CBF) certified set on the SAME grid -> rho denominator
    aVlo, _ = brake_cbf_bounds(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1],
                               lo[:, 2], hi[:, 2])
    cell_vol = ((cfg.p_hi - cfg.p_lo) / npx) * ((cfg.p_hi - cfg.p_lo) / npy) * (cfg.v_max / nv)
    return {"lo": lo, "hi": hi, "lbV": lbV, "ubV": ubV, "gmin": gmin,
            "analytic_ok": aVlo >= 0.0, "cell_vol": cell_vol,
            "npx": npx, "npy": npy, "nv": nv}


# --------------------------------------------------------------------------- #
import itertools


# --------------------------------------------------------------------------- #
# 4-D cell lattice (px, py, psi, v) for the HEADING-INCLUSIVE cell-reachability
# certificate (psi periodic -> doubled-axis SAT; v non-periodic).
# --------------------------------------------------------------------------- #
def _psi_plus_bounds(cfg, ps_lo, ps_hi, v_lo, v_hi):
    """Sound [psi+_lo, psi+_hi] under braking with ANY steering in
    [-delta_max, delta_max] and steer disturbance (covers the deployed brake).
    psi+ = psi + dt (v/L) tan(delta + d_delta); |delta+d_delta| <= dmax+ddmax
    < pi/2 so tan is monotone -> the turn magnitude is dt (v/L) tan(dmax+ddmax)."""
    tmax = np.tan(cfg.delta_max + cfg.d_delta_max)
    T = cfg.dt * (v_hi / cfg.wheelbase) * tmax        # max |turn| at top speed
    return ps_lo - T, ps_hi + T                       # may exceed [-pi,pi] (wrapped later)


def _ranges_4d(cfg, res4, boxes):
    """Inclusive 4-D lattice index ranges of boxes (M,8)=[pxlo,pxhi,pylo,pyhi,
    pslo,pshi,vlo,vhi], psi on the DOUBLED axis [0,2*np_), v clamped [0,nv).
    Returns rng (M,8) int64 and dom_ok (M,) (position + speed inside domain)."""
    nx, ny, np_, nv = res4
    hx = (cfg.p_hi - cfg.p_lo) / nx
    hy = (cfg.p_hi - cfg.p_lo) / ny
    hp = (2 * np.pi) / np_
    hv = cfg.v_max / nv
    b = np.asarray(boxes, float)
    dom = ((b[:, 0] >= cfg.p_lo - 1e-9) & (b[:, 1] <= cfg.p_hi + 1e-9)
           & (b[:, 2] >= cfg.p_lo - 1e-9) & (b[:, 3] <= cfg.p_hi + 1e-9))

    def ax(lo, hi, origin, h, n, clip_hi):
        i0 = np.floor((lo - origin) / h).astype(np.int64)
        i1 = np.floor((hi - origin) / h).astype(np.int64)
        return np.clip(i0, 0, clip_hi), np.clip(i1, 0, clip_hi)

    ix0, ix1 = ax(b[:, 0], b[:, 1], cfg.p_lo, hx, nx, nx - 1)
    iy0, iy1 = ax(b[:, 2], b[:, 3], cfg.p_lo, hy, ny, ny - 1)
    iv0, iv1 = ax(b[:, 6], b[:, 7], 0.0, hv, nv, nv - 1)
    # psi: shift lo into [-pi,pi) then index on the doubled axis [0, 2 np_)
    pl = b[:, 4] - np.floor((b[:, 4] + np.pi) / (2 * np.pi)) * (2 * np.pi)
    ph = b[:, 5] - np.floor((b[:, 4] + np.pi) / (2 * np.pi)) * (2 * np.pi)
    ip0 = np.clip(np.floor((pl + np.pi) / hp).astype(np.int64), 0, 2 * np_ - 1)
    ip1 = np.clip(np.floor((ph + np.pi) / hp).astype(np.int64), 0, 2 * np_ - 1)
    rng = np.stack([ix0, ix1, iy0, iy1, ip0, ip1, iv0, iv1], axis=1)
    return rng, dom


def _sat_4d(accepted, res4):
    """Zero-padded 4-D summed-area table; psi axis tiled x2 (periodic)."""
    nx, ny, np_, nv = res4
    A = accepted.reshape(nx, ny, np_, nv)
    A2 = np.concatenate([A, A], axis=2).astype(np.int32)
    P = np.zeros((nx + 1, ny + 1, 2 * np_ + 1, nv + 1), dtype=np.int32)
    P[1:, 1:, 1:, 1:] = A2.cumsum(0).cumsum(1).cumsum(2).cumsum(3)
    return P


def _accepted_4d(P, rng):
    """True where EVERY lattice cell in the inclusive range is accepted."""
    lo = rng[:, ::2]
    hi = rng[:, 1::2] + 1
    s = np.zeros(len(rng), np.int64)
    for bits in itertools.product((0, 1), repeat=4):
        idx = tuple(hi[:, d] if bits[d] else lo[:, d] for d in range(4))
        sign = -1 if (4 - sum(bits)) % 2 else 1
        s = s + sign * P[idx].astype(np.int64)
    need = np.prod(hi - lo, axis=1)
    return s == need


def certify(cfg, v_net, q_net, pi_net, res, npsi, c_grid, n_a_cells=8,
            eps=5e-3, gamma=GAMMA, chunk=4096, verbose=True):
    """Attempt a SOUND, non-vacuous Gate-D certificate of the LEARNED trio by the
    two available sound routes -- DIRECTLY on V_t, Q_t, pi_b, no V_t~V_target.

    Route A (headline): heading-free braking-invariant sub-level {V_t>=c}.
      Valid c iff the whole sub-level {ubV>=c} is (C1) collision-free and (C2)
      braking-closed: CROWN lbV over the worst-heading brake successor >= c.
    Route B: heading-inclusive 4-D cell-reachability (Tarski GFP) under the brake
      map, run on BOTH the learned candidate {lbV_t>=c} and the ANALYTIC ideal
      candidate (V_target's exact box bounds) -- the analytic run isolates the
      cell+box obstruction from the learned object.
    C3 (witness feasibility, CROWN on Q_t(x,pi_b(x),d)) is verified on a
    sub-sample and REPORTED as liveness (does not gate safety: the brake fallback
    is always applicable and its closure is C2).

    Binding quantity: the brake-successor undershoot ubV_t - lbV_t(x+) ~ the
    CROWN gap on V_t ~ the true per-cell variation of V (the analytic CBF closes
    A via an EXACT 1-Lipschitz cancellation a black-box V_t cannot reproduce; B
    is obstructed by successor-box overlap even for the analytic V).
    """
    t0 = time.time()
    bg = base_grid(cfg, v_net, *res, chunk, verbose)
    lo, hi, lbV, ubV, gmin = bg["lo"], bg["hi"], bg["lbV"], bg["ubV"], bg["gmin"]
    a_ok = bg["analytic_ok"]
    cmin = float(min(c_grid))
    work = np.flatnonzero(ubV >= cmin)
    M = len(work)
    if verbose:
        print(f"  [base] {len(lbV)} cells, {M} reach ubV>={cmin:.2f} "
              f"(analytic certified {int(a_ok.sum())})", flush=True)
    if M == 0:
        return {"empty": True, "wall_s": round(time.time() - t0, 1)}, None

    wlo, whi = lo[work], hi[work]
    wlbV, wubV, wgmin, wa_ok = lbV[work], ubV[work], gmin[work], a_ok[work]
    ps_edges = np.linspace(-np.pi, np.pi, npsi + 1)
    nv = res[2]
    res4 = (res[0], res[1], npsi, nv)
    wtmp, wiv = work // nv, work % nv
    lbVbrake = np.full(M, np.inf)                  # Route-A: worst-psi brake lbV
    rng4 = np.zeros((npsi, M, 8), dtype=np.int64)  # Route-B: brake-successor ranges
    dom4 = np.zeros((npsi, M), dtype=bool)
    idx4 = np.stack([(wtmp * npsi + ip) * nv + wiv for ip in range(npsi)])

    pb = Progress(npsi, "cert-psi") if verbose else None
    for h in range(npsi):
        pl, ph = ps_edges[h], ps_edges[h + 1]
        bx0, bx1, by0, by1, bv0, bv1 = successor_box(
            cfg, wlo[:, 0], whi[:, 0], wlo[:, 1], whi[:, 1],
            np.full(M, pl), np.full(M, ph), wlo[:, 2], whi[:, 2],
            cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)
        bl, _ = crown_bounds_chunked(v_net, np.column_stack([bx0, by0, bv0]),
                                     np.column_stack([bx1, by1, bv1]), True, chunk)
        lbVbrake = np.minimum(lbVbrake, bl[:, 0])
        psp_lo, psp_hi = _psi_plus_bounds(cfg, pl, ph, wlo[:, 2], whi[:, 2])
        rng4[h], dom4[h] = _ranges_4d(cfg, res4, np.column_stack(
            [bx0, bx1, by0, by1, psp_lo, psp_hi, bv0, bv1]))
        if pb is not None:
            pb.update(h + 1)
    if pb is not None:
        pb.done()
    undershoot = wubV - lbVbrake

    # ---- Route A: heading-free sub-level closure (smallest valid c) -------- #
    routeA = None
    for c in c_grid:
        inset = wubV >= c
        if inset.any() and np.all(wgmin[inset] >= 0.0) and np.all(lbVbrake[inset] >= c):
            routeA = float(c)
            break

    # ---- Route B: 4-D cell-reachability GFP (learned + analytic ideal) ----- #
    def reach_gfp(cand_w):
        accepted = np.zeros(res4[0] * res4[1] * npsi * nv, dtype=bool)
        for ip in range(npsi):
            accepted[idx4[ip]] = cand_w
        hist = [int(accepted.sum())]
        it = 0
        while it < 100:
            it += 1
            P = _sat_4d(accepted, res4)
            changed = 0
            for ip in range(npsi):
                ok = _accepted_4d(P, rng4[ip]) & dom4[ip]
                kill = accepted[idx4[ip]] & ~ok
                if kill.any():
                    accepted[idx4[ip][kill]] = False
                    changed += int(kill.sum())
            hist.append(int(accepted.sum()))
            if changed == 0:
                break
        return accepted, it, hist

    acc_L, itL, histL = reach_gfp((wlbV >= 0.0) & (wgmin >= 0.0))      # learned
    acc_A, itA, histA = reach_gfp(wa_ok & (wgmin >= 0.0))             # analytic ideal
    best_acc = acc_L if acc_L.any() else None
    certB = int(acc_L.sum())

    # ---- C3 liveness on a sub-sample (verified, reported -- not gating) ---- #
    Dlo = np.array([-cfg.d_a_max, -cfg.d_delta_max])
    Dhi = np.array([cfg.d_a_max, cfg.d_delta_max])
    si = np.unique(np.linspace(0, M - 1, min(M, 3000)).astype(int))
    slo, shi, sub_ub = wlo[si], whi[si], wubV[si]
    Ms = len(si)
    h3 = np.full((npsi, Ms), -np.inf)
    for h in range(npsi):
        pl, ph = ps_edges[h], ps_edges[h + 1]
        xl = np.column_stack([slo[:, 0], slo[:, 1], np.full(Ms, pl), slo[:, 2]])
        xh = np.column_stack([shi[:, 0], shi[:, 1], np.full(Ms, ph), shi[:, 2]])
        yl, yh = crown_bounds_chunked(pi_net, xl, xh, True, chunk)
        qlo, _ = crown_bounds_chunked(
            q_net,
            np.column_stack([xl, np.clip(yl[:, 0], cfg.a_min, cfg.a_max),
                             np.clip(yl[:, 1], -cfg.delta_max, cfg.delta_max), np.tile(Dlo, (Ms, 1))]),
            np.column_stack([xh, np.clip(yh[:, 0], cfg.a_min, cfg.a_max),
                             np.clip(yh[:, 1], -cfg.delta_max, cfg.delta_max), np.tile(Dhi, (Ms, 1))]),
            True, chunk)
        h3[h] = qlo[:, 0] - gamma * sub_ub - eps
    c3_ok = (h3 >= 0.0).all(axis=0)

    # build cert_cells only if a route actually closed
    if best_acc is not None:
        ai = np.flatnonzero(best_acc)
        iv = ai % nv; t = ai // nv; ip = t % npsi; t2 = t // npsi
        iy = t2 % res[1]; ix = t2 // res[1]
        hx = (cfg.p_hi - cfg.p_lo) / res[0]; hy = (cfg.p_hi - cfg.p_lo) / res[1]
        hp = 2 * np.pi / npsi; hv = cfg.v_max / nv
        x_lo = np.column_stack([cfg.p_lo + ix * hx, cfg.p_lo + iy * hy,
                                -np.pi + ip * hp, iv * hv])
        cert_cells = {"x_lo": x_lo, "x_hi": x_lo + np.array([hx, hy, hp, hv]),
                      "acc": best_acc, "res4": res4, "c": 0.0}
    else:
        cert_cells = None

    near = (wubV >= 0.0) & (wubV < 0.15)
    us = undershoot[near]
    crown_gap = wubV - wlbV
    analytic_inc = int(a_ok.sum()) * npsi
    report = {
        "set_kind": "learned-trio Gate-D attempt (route A sub-level + route B reachability)",
        "crown_gap_V_med": float(np.median(crown_gap)),
        "crown_gap_V_p90": float(np.percentile(crown_gap, 90)),
        "undershoot_med": float(np.median(us)) if us.size else float("nan"),
        "undershoot_p90": float(np.percentile(us, 90)) if us.size else float("nan"),
        "routeA_heading_free_closes_at_c": routeA,        # None => fails
        "routeB_reach_cert_learned": certB,               # 0 => fails
        "routeB_reach_cert_analytic_ideal": int(acc_A.sum()),
        "routeB_hist_learned": histL, "routeB_hist_analytic": histA,
        "routeB_iters_learned": itL, "routeB_iters_analytic": itA,
        "c3_hole_rate": float(np.mean(~c3_ok)),
        "c3_margin_med": float(np.median(h3[np.isfinite(h3)])),
        "rho_best": float(certB / max(analytic_inc, 1)),
        "res": list(res), "npsi": int(npsi), "res4": list(res4),
        "eps": eps, "gamma": gamma, "n_base_cells": int(len(lbV)), "n_work": int(M),
        "analytic_cert_base": int(a_ok.sum()), "cell_vol": bg["cell_vol"],
        "gate_d": bool(routeA is not None or certB > 0),
        "wall_s": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  [tight] CROWN V gap med={report['crown_gap_V_med']:.3f}  "
              f"brake-undershoot med={report['undershoot_med']:+.3f}", flush=True)
        print(f"  [routeA] heading-free sub-level closes @ c={routeA}", flush=True)
        print(f"  [routeB] reach GFP cert: learned={certB} (it{itL}), "
              f"analytic-ideal={int(acc_A.sum())} (it{itA})  "
              f"hist_an={histA[:4]}..{histA[-1]}", flush=True)
        print(f"  [C3] witness-feasible (sub-sample): hole-rate={report['c3_hole_rate']:.2f} "
              f"({report['wall_s']:.0f}s)", flush=True)
    return report, cert_cells


# --------------------------------------------------------------------------- #
# Audit: roll out the LEARNED min-intervention filter under adversaries.
# --------------------------------------------------------------------------- #
def racing_action(cfg, x):
    px, py, psi, v = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    acc = np.where(v < cfg.v_max, cfg.a_max, 0.0)
    return np.stack([acc, racing_steer(cfg, x)], axis=-1)


def racing_successor_box(cfg, x):
    """8-col (px,py,psi,v) box of the racing successor of POINT states x over all
    disturbances d (px+,py+ deterministic; v+ spans d_a; psi+ spans d_delta)."""
    px, py, psi, v = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
    cpx = px + cfg.dt * v * np.cos(psi)
    cpy = py + cfg.dt * v * np.sin(psi)
    a = np.where(v < cfg.v_max, cfg.a_max, 0.0)
    v0 = np.clip(v + cfg.dt * (a - cfg.d_a_max), 0.0, cfg.v_max)
    v1 = np.clip(v + cfg.dt * (a + cfg.d_a_max), 0.0, cfg.v_max)
    steer = racing_steer(cfg, x)
    T = cfg.dt * (v / cfg.wheelbase)
    plo = psi + T * np.tan(steer - cfg.d_delta_max)
    phi = psi + T * np.tan(steer + cfg.d_delta_max)
    return np.column_stack([cpx, cpx, cpy, cpy, plo, phi, v0, v1])


def learned_filter_step(cfg, model, pi_net, x, d, P, res4):
    """Min-intervention filter the certificate covers: apply the racing action
    iff its successor box over ALL disturbances lies inside the certified set
    (sound membership via the SAT P of the accepted mask), else BRAKE (the
    certified fallback, whose closure was verified).  cbv = 0 for every d."""
    rng, dom = _ranges_4d(cfg, res4, racing_successor_box(cfg, x))
    feas = _accepted_4d(P, rng) & dom
    u_race = racing_action(cfg, x)
    # certified fallback = HARD brake (a_min, matching the certified brake box)
    # with the learned witness steering (covered by the full-steering psi+ box)
    steer_b = np.clip(pi_net.forward(x)[:, 1], -cfg.delta_max, cfg.delta_max)
    u_b = np.column_stack([np.full(len(x), cfg.a_min), steer_b])
    u = np.where(feas[:, None], u_race, u_b)
    return model.step(x, u, d), (~feas)


def audit(cfg, model, v_net, pi_net, sampler, cert_cells, n_roll=400, horizon=200,
          seed=1, verbose=True):
    """Roll out the SAT-gated learned filter from the certified 4-D cells under
    extremal + greedy(heading-steering) adversaries.  cbv must be 0."""
    P = _sat_4d(cert_cells["acc"], cert_cells["res4"])
    res4 = cert_cells["res4"]
    rng = np.random.default_rng(seed)
    out = {}
    for mode in ("extremal", "greedy"):
        X = sampler(n_roll, rng)
        ming = model.g(X).copy()
        minV = v_net.forward(X[:, [0, 1, 3]])[:, 0].copy()
        brake = 0.0
        for t in range(horizon):
            if mode == "extremal":
                d = np.stack([rng.choice([-cfg.d_a_max, cfg.d_a_max], len(X)),
                              rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))], axis=-1)
                X, br = learned_filter_step(cfg, model, pi_net, X, d, P, res4)
            else:  # greedy adversary: pick d minimizing next analytic g (steers heading)
                cand = np.array([[cfg.d_a_max, 0.0], [cfg.d_a_max, cfg.d_delta_max],
                                 [cfg.d_a_max, -cfg.d_delta_max], [-cfg.d_a_max, cfg.d_delta_max],
                                 [-cfg.d_a_max, -cfg.d_delta_max]])
                bestX = None; bestg = None; bestbr = None
                for dc in cand:
                    dd = np.tile(dc, (len(X), 1))
                    xn, br = learned_filter_step(cfg, model, pi_net, X, dd, P, res4)
                    gn = model.g(xn)
                    if bestg is None:
                        bestX, bestg, bestbr = xn, gn, br
                    else:
                        take = gn < bestg
                        bestX = np.where(take[:, None], xn, bestX)
                        bestg = np.where(take, gn, bestg)
                        bestbr = np.where(take, br, bestbr)
                X, br = bestX, bestbr
            brake += br.mean()
            np.minimum(ming, model.g(X), out=ming)
            np.minimum(minV, v_net.forward(X[:, [0, 1, 3]])[:, 0], out=minV)
        out[mode] = {"min_g": float(ming.min()), "min_V": float(minV.min()),
                     "g_violations": int((ming < 0).sum()),
                     "brake_frac": float(brake / horizon)}
        if verbose:
            r = out[mode]
            print(f"  [audit:{mode:8s}] min g {r['min_g']:+.4f}  min V {r['min_V']:+.4f}  "
                  f"g-viol {r['g_violations']}  brake {100*r['brake_frac']:.0f}%", flush=True)
    out["certified_but_violated"] = int(sum(out[m]["g_violations"] for m in ("extremal", "greedy")))
    return out


def cert_set_sampler(cert_cells):
    """Sample initial states UNIFORMLY inside the certified 4-D cells (the
    heading-inclusive tube).  Every certified heading is represented; the greedy
    adversary additionally steers heading via d_delta over the rollout, so the
    audit covers the heading axis -- no favorable-subset cherry-picking."""
    xlo, xhi = cert_cells["x_lo"], cert_cells["x_hi"]      # (K,4)
    K = len(xlo)

    def sample(n, rng):
        idx = rng.integers(0, K, n)
        u = rng.random((n, 4))
        return xlo[idx] + u * (xhi[idx] - xlo[idx])
    return sample


# --------------------------------------------------------------------------- #
def gate_d_verdict(rep, aud):
    return bool(rep and not rep.get("empty") and rep.get("gate_d")
                and aud["certified_but_violated"] == 0)


def run_one(cfg, seed, margin, res, npsi, c_grid, train_kw, audit_kw=None,
            verbose=True):
    v, q, pi, diag = distill(cfg, seed, margin, verbose=verbose, **train_kw)
    vn, qn, pn = SeqNet.from_mlp(v), SeqNet.from_mlp(q), SeqNet.from_mlp(pi)
    rep, cert_cells = certify(cfg, vn, qn, pn, res, npsi, c_grid, verbose=verbose)
    model = BicycleAccelModel(cfg)
    if cert_cells is None:                       # nothing closed -> nothing to audit
        aud = {"certified_but_violated": 0, "note": "empty certified set -- audit skipped"}
        gate = False
    else:
        aud = audit(cfg, model, v, pi, cert_set_sampler(cert_cells),
                    cert_cells, verbose=verbose, **(audit_kw or {}))
        gate = gate_d_verdict(rep, aud)
    if verbose:
        print(f"[run] seed={seed} m={margin:.2f}: GATE D {'PASS' if gate else 'FAIL'}  "
              f"rho={rep.get('rho_best', 0.0):.3f}  "
              f"cbv={aud['certified_but_violated']}", flush=True)
    return {"diag": diag, "cert": rep, "audit": aud, "gate_d_pass": gate}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny smoke config")
    ap.add_argument("--probe", action="store_true",
                    help="one full-training config to measure CROWN tightness")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    c_grid = (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)

    if args.quick:
        seeds, margins, res, npsi = [0], [0.10], (32, 32, 24), 8
        train_kw = dict(n_samples=20_000, reg_epochs=10, cbf_epochs=10)
        res_sweep = []
    elif args.probe:
        seeds, margins, res, npsi = [0], [0.10], (48, 48, 36), 14
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=25)
        res_sweep = []
    else:
        seeds, margins = [0, 1, 2], [0.0, 0.05, 0.10, 0.15]
        res, npsi = (44, 44, 34), 12
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=25)
        # resolution mini-sweep (1 seed, m=0.10) -> undershoot vs cell size
        res_sweep = [(28, 28, 22, 10), (44, 44, 34, 12), (60, 60, 46, 16)]

    t0 = time.time()
    results = {}
    total = len(seeds) * len(margins)
    k = 0
    for m in margins:
        for s in seeds:
            k += 1
            print(f"\n=== [{k}/{total}] seed={s} margin={m:.2f} "
                  f"res={res} npsi={npsi} ===", flush=True)
            results[f"m{m:.2f}_s{s}"] = run_one(cfg, s, m, res, npsi, c_grid, train_kw)

    # ---- resolution mini-sweep: brake-undershoot vs cell size (attribution) #
    res_attr = []
    for (rx, ry, rv, rp) in res_sweep:
        print(f"\n=== [res-attr] res=({rx},{ry},{rv}) npsi={rp} (seed 0, m=0.10) ===", flush=True)
        r = run_one(cfg, 0, 0.10, (rx, ry, rv), rp, c_grid, train_kw)
        res_attr.append({"res": [rx, ry, rv], "npsi": rp,
                         "cell_m": round((cfg.p_hi - cfg.p_lo) / rx, 4),
                         "undershoot_med": r["cert"].get("undershoot_med"),
                         "crown_gap_med": r["cert"].get("crown_gap_V_med"),
                         "routeA_closes": r["cert"].get("routeA_heading_free_closes_at_c"),
                         "routeB_learned": r["cert"].get("routeB_reach_cert_learned"),
                         "routeB_analytic": r["cert"].get("routeB_reach_cert_analytic_ideal")})

    # ---- aggregate per-margin (both routes, distributions over seeds) ----- #
    def _f(r, k, d=float("nan")):
        return r["cert"].get(k, d)

    summary = {}
    for m in margins:
        runs = [results[f"m{m:.2f}_s{s}"] for s in seeds]
        passes = [r["gate_d_pass"] for r in runs]
        rA = [_f(r, "routeA_heading_free_closes_at_c") for r in runs]
        certB = [_f(r, "routeB_reach_cert_learned", 0) for r in runs]
        certBa = [_f(r, "routeB_reach_cert_analytic_ideal", 0) for r in runs]
        us = [_f(r, "undershoot_med") for r in runs]
        cg = [_f(r, "crown_gap_V_med") for r in runs]
        c3h = [_f(r, "c3_hole_rate") for r in runs]
        summary[f"m{m:.2f}"] = {
            "pass_rate": float(np.mean(passes)), "n_seeds": len(seeds),
            "routeA_closes_per_seed": rA,
            "routeB_cert_learned_per_seed": certB,
            "routeB_cert_analytic_per_seed": certBa,
            "undershoot_med_mean": float(np.nanmean(us)),
            "crown_gap_med_mean": float(np.nanmean(cg)),
            "c3_hole_rate_mean": float(np.nanmean(c3h)),
        }

    blob = {"config": {"v_max": cfg.v_max, "a_min": cfg.a_min,
                       "brake_decel": cfg.brake_decel, "cbf_margin": cfg.cbf_margin,
                       "gamma": GAMMA, "res": list(res), "npsi": npsi,
                       "seeds": seeds, "margins": margins, "c_grid": list(c_grid)},
            "summary": summary, "results": results, "res_attribution": res_attr,
            "wall_s": round(time.time() - t0, 1)}
    (out / "e2_learned_report.json").write_text(json.dumps(blob, indent=2))

    # ---- final attributable report --------------------------------------- #
    print("\n" + "=" * 84)
    print("E2-L  LEARNED Gate-D (DIRECT C1/C2/C3 on V_t,Q_t,pi_b -- no V_t~V_target) result")
    print("  Route A: heading-free braking sub-level {V_t>=c}.  Route B: 4-D cell-reachability")
    print("  GFP (learned vs analytic-IDEAL candidate).  C3 = witness liveness (reported).")
    print("=" * 84)
    print(f"{'margin m':>9} | {'pass':>5} | {'A closes c':>11} | {'B cert(learn)':>13} | "
          f"{'B cert(an.)':>11} | {'undershoot':>11} | {'C3 hole':>8}")
    print("-" * 84)
    for m in margins:
        s = summary[f"m{m:.2f}"]
        a = [x for x in s["routeA_closes_per_seed"] if x is not None]
        astr = f"{np.mean(a):.2f}" if a else "none"
        print(f"{m:9.2f} | {s['pass_rate']*100:4.0f}% | {astr:>11} | "
              f"{np.mean(s['routeB_cert_learned_per_seed']):13.0f} | "
              f"{np.mean(s['routeB_cert_analytic_per_seed']):11.0f} | "
              f"{s['undershoot_med_mean']:+11.4f} | {s['c3_hole_rate_mean']:8.2f}")
    print("-" * 84)
    any_pass = any(summary[f"m{m:.2f}"]["pass_rate"] > 0 for m in margins)
    anaB = np.mean([np.mean(summary[f"m{m:.2f}"]["routeB_cert_analytic_per_seed"]) for m in margins])
    print(f"Verdict: {'PASS' if any_pass else 'FAIL -- no level closes by either sound route'}")
    print(f"  Route B certifies ~{anaB:.0f} cells for the ANALYTIC IDEAL V too -> empty learned")
    print(f"  route B is the cell+box successor-overlap obstruction, NOT the learned net.")
    if res_attr:
        print("\n  undershoot vs resolution (route A binding quantity):")
        print(f"  {'cell (m)':>9} | {'npsi':>4} | {'undershoot':>11} | {'A closes':>8} | {'B(an.)':>7}")
        for a in res_attr:
            print(f"  {a['cell_m']:9.3f} | {a['npsi']:4d} | {a['undershoot_med']:+11.4f} | "
                  f"{str(a['routeA_closes']):>8} | {a['routeB_analytic']:7d}")
    print(f"Wrote {out/'e2_learned_report.json'}  ({blob['wall_s']:.0f}s)")
    return blob


if __name__ == "__main__":
    main()
