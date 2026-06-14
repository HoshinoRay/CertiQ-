"""Recurrence (CBF-superlevel) certificate for the deployed learned artifact.

This is Theorem A in its *recurrence* form -- a single barrier function and a
single value condition, no Q-gate, no GFP.  Let

    W(x) = min( g(x), V_theta(x) )          (the safety clamp; {W>=0} subset K)

and let the deployed witness policy be  u = clip(pi_theta(x)).  For a level m,

    S_m = { x : W(x) >= m }.

The certificate condition checked here is, for every cell that can contain a
point of S_m (every "active" cell, ub_W(C) >= m):

    lb_W( f(C, [u_lo,u_hi], D) ) >= m,        (*)

a SOUND lower bound on W over the cell-worst successor box under the witness's
own control range [u_lo,u_hi] and the full disturbance D.  Because S_m is the
m-superlevel set of the SAME W, (*) says "the successor's W >= m", i.e. the
successor is back in S_m.  So:

    if (*) holds on ALL active cells  ==>  S_m is forward-invariant under the
    witness, and S_m subset {g >= m} subset K  (for m >= 0).

That is the whole proof -- no fixed-point iteration.  A viability-kernel GFP is
neither needed nor appropriate: it would discard W at the successor and round
the successor box up to whole cells, which is strictly looser.

Soundness of every step: the control range [u_lo,u_hi] is the CROWN enclosure
of clip(pi) over the cell (the deployed loop only applies clip(pi)); the
successor box uses the exact cos/sin and disturbance extrema (the trusted
successor_boxes map); lb_W = min(lb_g, lb_V) with the closed-form g bound and a
CROWN bound on V; the heading seam is split into two boxes.  None of it can
certify a false property -- looseness only shrinks the certified set.

Reporting.  The decisive distinction (this is the point of the script):

  * pass_frac(m)  = (active cells satisfying (*)) / (active cells).  This is a
    PASS RATE.  By itself it certifies NOTHING unless it is 1.0 -- a partial
    pass gives no invariant set, because a passing cell's successor may land in
    a failing cell.
  * rho_inner(m)  = Vol({lb_W >= m} cells) / Vol(Omega*).  This is the certified
    SET SIZE.  It is a valid certificate ONLY at levels m where pass_frac == 1.

We sweep m and report, for each level, both numbers, plus the smallest m at
which pass_frac reaches 1.0 (the largest COMPLETE certified S_m).  That answers
"what does the 0.64 mean": if pass_frac < 1 at m=0, then 0.64 was a pass rate,
not a certified volume, and the complete certificate is the S_{m*} at the level
where the whole superlevel set closes.
"""
from __future__ import annotations

import argparse
import time

from common import REPO, file_hash, save_json

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.certify.spec import _control_range
from qcbf.certify.volume import omega_star_volume
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import g_bounds_on_box, successor_boxes
from qcbf.nets.mlp import MLP
from qcbf.oracle.value_iteration import DubinsOracle
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.verify.compiler import compile_policy
from qcbf.verify.conditions import v_cell_bounds


def _succ_W_lower(v_net: SeqNet, cb: np.ndarray, dyn, cert,
                  u_lo: np.ndarray, u_hi: np.ndarray) -> np.ndarray:
    """Sound lower bound of  min_d W(f(C, u, d)),  W = min(g, V_theta),  over
    each state cell C, for the witness control range u in [u_lo, u_hi] and
    disturbance d in D.

    Mirrors qcbf.certify.spec._succ_v_lower but bounds W = min(g, V): the
    successor position box gets the exact closed-form g lower bound, the full
    successor box gets the CROWN V lower bound, and we take the min of the two
    (a sound lower bound of min(g, V)).  No domain guard is needed: the world
    disk lies inside the lattice domain, so any successor leaving the domain
    has g < 0 <= m and correctly fails (*).  Heading sub-split c4_psi_subsplit
    and disturbance sub-split c2_d_subsplit only tighten the enclosure.
    """
    N = len(cb)
    if N == 0:
        return np.zeros(0)
    u_lo = np.broadcast_to(np.asarray(u_lo, float), (N,))
    u_hi = np.broadcast_to(np.asarray(u_hi, float), (N,))
    npsi = max(1, cert.c4_psi_subsplit)
    nd = max(1, cert.c2_d_subsplit)
    d_edges = np.linspace(-dyn.d_max, dyn.d_max, nd + 1)
    psi_lo, psi_hi = cb[:, 4], cb[:, 5]
    worst = np.full(N, np.inf)
    for jp in range(npsi):
        pl = psi_lo + (psi_hi - psi_lo) * (jp / npsi)
        ph = psi_lo + (psi_hi - psi_lo) * ((jp + 1) / npsi)
        for kd in range(nd):
            b1, b2, m2 = successor_boxes(
                dyn, cb[:, 0], cb[:, 1], cb[:, 2], cb[:, 3], pl, ph,
                u_lo, u_hi, d_edges[kd], d_edges[kd + 1])
            lbV1, _ = crown_bounds_chunked(
                v_net, b1[:, [0, 2, 4]], b1[:, [1, 3, 5]], True, cert.chunk)
            lbg1, _ = g_bounds_on_box(dyn, b1[:, 0], b1[:, 1], b1[:, 2], b1[:, 3])
            val = np.minimum(lbV1[:, 0], lbg1)
            if np.any(m2):
                lbV2, _ = crown_bounds_chunked(
                    v_net, b2[:, [0, 2, 4]], b2[:, [1, 3, 5]], True, cert.chunk)
                lbg2, _ = g_bounds_on_box(dyn, b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3])
                lbW2 = np.minimum(lbV2[:, 0], lbg2)
                val = np.minimum(val, np.where(m2, lbW2, np.inf))
            worst = np.minimum(worst, val)
    return worst


# --------------------------------------------------------------------------- #
# T4 (Prop. T4, theory_core.md Sec. 5.9): monotone shrink-refinement of the
# certified set.  S_0 = {W >= m} cap K; iterate
#     S_{k+1} = S_k cap Pre^all_Phi(S_k),
# with the one-control witness predecessor: a cell C survives iff
#     (value)     lb_W(succ_C) >= m          -> succ subset {W >= m}   (TIGHT)
#   AND (geometric) reach(C) subset S_k       -> succ avoids removed cells.
# Any fixed point S = T_Phi(S) is robustly forward invariant under the deployed
# witness, and S subset {W >= m} subset K.  Only shrinking is sound; unknown /
# failed cells are dropped, never counted (Sec. 5.9).
# --------------------------------------------------------------------------- #
def _reach_ranges(cfg, lat, boxes, u_lo, u_hi):
    """Per-cell sound enclosure of the reachable LATTICE cells under the witness
    u in [u_lo,u_hi] and d in D (a sound SUPERSET -> stricter containment ->
    sound shrink).  Position is contiguous; the heading arc is split at the
    +-pi seam into <=2 ip ranges.  Returns the cell-range arrays + in_dom."""
    dyn = cfg.dynamics
    p_lo, p_hi = dyn.p_lo, dyn.p_hi
    hx, hy, hp = lat.hx, lat.hy, lat.hp
    nx, ny, npsi = lat.nx, lat.ny, lat.npsi
    TWO_PI = 2.0 * np.pi
    pxl, pxh, pyl, pyh = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    psil, psih = boxes[:, 4], boxes[:, 5]
    from qcbf.dynamics.dubins import cos_interval, sin_interval
    k = dyn.dt * dyn.v
    cmin, cmax = cos_interval(psil, psih)
    smin, smax = sin_interval(psil, psih)
    npx_lo, npx_hi = pxl + k * cmin, pxh + k * cmax
    npy_lo, npy_hi = pyl + k * smin, pyh + k * smax
    eps = 1e-12
    in_dom = ((npx_lo >= p_lo - eps) & (npx_hi <= p_hi + eps)
              & (npy_lo >= p_lo - eps) & (npy_hi <= p_hi + eps))
    ix0 = np.clip(np.floor((npx_lo - p_lo) / hx).astype(np.int64), 0, nx - 1)
    ix1 = np.clip(np.floor((npx_hi - p_lo) / hx).astype(np.int64), 0, nx - 1)
    iy0 = np.clip(np.floor((npy_lo - p_lo) / hy).astype(np.int64), 0, ny - 1)
    iy1 = np.clip(np.floor((npy_hi - p_lo) / hy).astype(np.int64), 0, ny - 1)
    a = psil + dyn.dt * (u_lo - dyn.d_max)
    b = psih + dyn.dt * (u_hi + dyn.d_max)
    full = (b - a) >= TWO_PI
    shift = np.floor((a + np.pi) / TWO_PI) * TWO_PI
    a2, b2 = a - shift, b - shift
    crosses = (b2 > np.pi) & ~full
    hi1 = np.where(crosses, np.pi, b2)
    ip1a = np.clip(np.floor((a2 + np.pi) / hp).astype(np.int64), 0, npsi - 1)
    ip1b = np.clip(np.floor((hi1 + np.pi) / hp).astype(np.int64), 0, npsi - 1)
    ip2b = np.clip(np.floor((b2 - TWO_PI + np.pi) / hp).astype(np.int64), 0, npsi - 1)
    ip1a = np.where(full, 0, ip1a); ip1b = np.where(full, npsi - 1, ip1b)
    crosses = crosses & ~full
    total = ((ix1 - ix0 + 1) * (iy1 - iy0 + 1)
             * ((ip1b - ip1a + 1) + np.where(crosses, ip2b + 1, 0)))
    return dict(ix0=ix0, ix1=ix1, iy0=iy0, iy1=iy1, ip1a=ip1a, ip1b=ip1b,
                ip2b=ip2b, crosses=crosses, in_dom=in_dom, total=total)


def _sat(P, x0, x1, y0, y1, z0, z1):
    x1, y1, z1 = x1 + 1, y1 + 1, z1 + 1
    return (P[x1, y1, z1] - P[x0, y1, z1] - P[x1, y0, z1] - P[x1, y1, z0]
            + P[x0, y0, z1] + P[x0, y1, z0] + P[x1, y0, z0] - P[x0, y0, z0])


def t4_refine(active, rec_ok, rr, lat, verbose=True):
    """Greatest fixed point of the realized T4 operator on the cell lattice.
    keep = active & rec_ok & (reach subset keep), iterated to convergence."""
    nx, ny, npsi = lat.nx, lat.ny, lat.npsi
    zero = np.zeros_like(rr["ip2b"])
    base = active & rec_ok & rr["in_dom"]   # value gate + domain, fixed
    keep = base.copy()
    for it in range(1, 100000):
        kept3 = keep.reshape(nx, ny, npsi).astype(np.int64)
        P = np.zeros((nx + 1, ny + 1, npsi + 1), dtype=np.int64)
        P[1:, 1:, 1:] = kept3.cumsum(0).cumsum(1).cumsum(2)
        s = _sat(P, rr["ix0"], rr["ix1"], rr["iy0"], rr["iy1"], rr["ip1a"], rr["ip1b"])
        s = s + np.where(rr["crosses"],
                         _sat(P, rr["ix0"], rr["ix1"], rr["iy0"], rr["iy1"], zero, rr["ip2b"]), 0)
        survive = base & (s == rr["total"])
        removed = int(keep.sum() - survive.sum())
        if verbose:
            print(f"    [T4] sweep {it:3d}: keep {int(survive.sum()):6d}  (-{removed})")
        if removed == 0:
            return survive, it
        keep = survive
    return keep, it


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument("--cells", type=int, default=0,
                    help="Override lattice resolution per axis (verifier lever).")
    ap.add_argument("--m-grid", default="0.0,0.02,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30",
                    help="Comma-separated levels m to sweep.")
    ap.add_argument("--t4", action="store_true",
                    help="Also run the T4 shrink-refinement (Sec. 5.9) to the "
                    "greatest certified forward-invariant subset.")
    ap.add_argument("--t4-m", default="0.0",
                    help="Comma-separated levels m for the T4 certificate "
                    "(S_0 = {W>=m} cap K); the deepest non-empty fixed point wins.")
    args = ap.parse_args()

    import dataclasses
    cfg = ExperimentConfig.load(args.config)
    if args.out_dir:
        cfg = dataclasses.replace(cfg, out_dir=args.out_dir)
    if args.cells > 0:
        cfg = dataclasses.replace(cfg, cert=dataclasses.replace(
            cfg.cert, n_cells_px=args.cells, n_cells_py=args.cells,
            n_cells_psi=args.cells))
    out = REPO / cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)

    def op(name: str):
        return REPO / cfg.out_dir / name

    t_all = time.time()
    dyn, cert = cfg.dynamics, cfg.cert
    m_grid = [float(s) for s in args.m_grid.split(",")]
    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    n = lat.n_cells

    v = MLP.load(str(op("v.npz")))
    pi = MLP.load(str(op("pi.npz")))
    v_net = SeqNet.from_mlp(v)
    pol_net = compile_policy(pi, dyn.control_max)

    # ---- W = min(g, V) cell bounds ---------------------------------------- #
    t0 = time.time()
    lbV, ubV = v_cell_bounds(v_net, boxes, np.arange(n), n, cert.chunk)
    gmin, gmax = g_bounds_on_box(dyn, boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    lbW = np.minimum(lbV, gmin)
    ubW = np.minimum(ubV, gmax)
    print(f"[recurrence-cert] cells {n}  (W bounds {time.time()-t0:.1f}s)  "
          f"max m candidate {{ubW>=0}} = {int((ubW >= 0).sum())} cells")

    # ---- successor lb_W on every candidate cell (ub_W >= 0); independent of m
    cand = np.flatnonzero(ubW >= 0.0)
    t0 = time.time()
    u_lo, u_hi = _control_range(pol_net, boxes[cand], dyn, cert)
    succ_lbW = np.full(n, -np.inf)
    succ_lbW[cand] = _succ_W_lower(v_net, boxes[cand], dyn, cert, u_lo, u_hi)
    # full-length control-range arrays (witness range over EVERY cell), for the
    # T4 geometric reach (cells outside cand are never kept, so any value is ok).
    ulo_full = np.zeros(n); uhi_full = np.zeros(n)
    ulo_full[cand], uhi_full[cand] = u_lo, u_hi
    print(f"[recurrence-cert] witness successor lb_W on {len(cand)} candidate "
          f"cells ({time.time()-t0:.1f}s)")

    # ---- Omega* reference volume ------------------------------------------ #
    oracle = DubinsOracle(dyn, cfg.oracle, gamma=cfg.train.gamma_teach)
    V_star = np.load(op("oracle.npz"))["V"]
    om = omega_star_volume(oracle, V_star, cfg)
    cell_vol = float(lat.cell_volume)

    # ---- sweep the level m ------------------------------------------------- #
    levels = []
    m_star = None
    for m in m_grid:
        active = ubW >= m                       # must all satisfy (*)
        inner = lbW >= m                         # sound interior of S_m
        rec_ok = succ_lbW >= m
        n_active = int(active.sum())
        n_pass = int((active & rec_ok).sum())
        n_fail = n_active - n_pass
        pass_frac = float(n_pass / n_active) if n_active else 0.0
        complete = (n_fail == 0) and n_active > 0
        # certified volume: inner cells, but only meaningful as a certificate
        # when complete.  Report rho_inner regardless, flagged by `complete`.
        vol_inner = int(inner.sum()) * cell_vol
        rho_inner = float(vol_inner / om["volume"]) if om["volume"] > 0 else 0.0
        # worst recurrence margin over the cells we must pass
        marg = (succ_lbW - m)[active]
        rec = {
            "m": m,
            "active_cells": n_active,
            "inner_cells": int(inner.sum()),
            "recurrence_pass": n_pass,
            "recurrence_fail": n_fail,
            "pass_frac": pass_frac,
            "complete_certificate": bool(complete),
            "rho_inner_vs_omega_star": rho_inner,
            "rho_certified": rho_inner if complete else 0.0,
            "min_recurrence_margin": float(marg.min()) if n_active else None,
        }
        levels.append(rec)
        if complete and m_star is None:
            m_star = rec
        flag = "COMPLETE" if complete else f"incomplete ({n_fail} fail)"
        print(f"  m={m:.3f}: active {n_active:6d}  pass {100*pass_frac:5.1f}%  "
              f"inner rho {rho_inner:.4f}  -> {flag}")

    # ---- failure diagnosis at m=0: verifier-slack vs genuine policy leak --- #
    active0 = ubW >= 0.0
    fail0 = active0 & (succ_lbW < 0.0)
    inner0 = lbW >= 0.0                      # truly inside S_0
    fi = fail0 & inner0                       # inner cells that fail (the real problem)
    fb = fail0 & ~inner0                      # straddle/boundary cells that fail
    sl = succ_lbW[fail0]
    margin0_diag = {
        "active_m0": int(active0.sum()),
        "fail_m0": int(fail0.sum()),
        "fail_inner": int(fi.sum()),          # lb_W(cell) >= 0 but successor leaks
        "fail_boundary_straddle": int(fb.sum()),
        "succ_lbW_on_fail_min": float(sl.min()) if sl.size else None,
        "succ_lbW_on_fail_p05": float(np.percentile(sl, 5)) if sl.size else None,
        "succ_lbW_on_fail_p50": float(np.percentile(sl, 50)) if sl.size else None,
        "fail_near_miss_-0.02": int((sl >= -0.02).sum()),   # verifier-slack band
        "fail_near_miss_-0.05": int((sl >= -0.05).sum()),
        "fail_deep_<-0.2": int((sl < -0.2).sum()),          # genuine leak band
    }
    np.savez(op(f"recurrence_cert_arrays{args.tag}.npz"),
             lbW=lbW, ubW=ubW, succ_lbW=succ_lbW)
    print(f"  [diag m=0] fail {int(fail0.sum())} = inner {int(fi.sum())} + "
          f"straddle {int(fb.sum())}; succ_lbW on fail: min {margin0_diag['succ_lbW_on_fail_min']:+.3f} "
          f"p50 {margin0_diag['succ_lbW_on_fail_p50']:+.3f}; near-miss(>=-0.02) "
          f"{margin0_diag['fail_near_miss_-0.02']}, deep(<-0.2) {margin0_diag['fail_deep_<-0.2']}")

    # ---- T4 shrink-refinement (Sec. 5.9): greatest certified invariant subset
    t4 = None
    if args.t4:
        rr = _reach_ranges(cfg, lat, boxes, ulo_full, uhi_full)
        t4_levels = []
        best = None
        for mt in [float(s) for s in args.t4_m.split(",")]:
            active = ubW >= mt
            rec_ok = succ_lbW >= mt
            print(f"  [T4] m={mt}: S_0 = {{W>=m}} cap K, active {int(active.sum())}, "
                  f"one-step value-pass {int((active&rec_ok).sum())}")
            keep, iters = t4_refine(active, rec_ok, rr, lat, verbose=False)
            inner_keep = keep & (lbW >= mt)     # fully-in cells -> sound volume
            vol = int(inner_keep.sum()) * cell_vol
            rho_t4 = float(vol / om["volume"]) if om["volume"] > 0 else 0.0
            rec = {"m": mt, "S0_active_cells": int(active.sum()),
                   "one_step_value_pass": int((active & rec_ok).sum()),
                   "fixedpoint_cells": int(keep.sum()),
                   "fixedpoint_inner_cells": int(inner_keep.sum()),
                   "gfp_iters": iters, "certified_volume": vol,
                   "rho_vs_omega_star": rho_t4}
            t4_levels.append(rec)
            print(f"       fixed point: {int(keep.sum())} cells "
                  f"({int(inner_keep.sum())} inner) in {iters} sweeps, rho={rho_t4:.4f}")
            if best is None or keep.sum() > best[1].sum():
                best = (rec, keep)
        if best is not None:
            np.save(op(f"recurrence_t4_keep{args.tag}.npy"), best[1])
        t4 = {
            "levels": t4_levels,
            "best": best[0] if best else None,
            "note": "Greatest fixed point of T_Phi (one-control witness, value gate "
                    "lb_W(succ)>=m + geometric reach subset keep).  Robustly forward "
                    "invariant; rho from inner (fully-in) kept cells (sound under-count). "
                    "Deepest non-empty fixed point = the certified viability core.",
        }

    headline = {
        "complete_certificate_exists": m_star is not None,
        "m_star": m_star["m"] if m_star else None,
        "rho_certified": m_star["rho_certified"] if m_star else 0.0,
        "rho_at_m0": levels[0]["rho_inner_vs_omega_star"],
        "pass_frac_at_m0": levels[0]["pass_frac"],
        "t4_best_rho": t4["best"]["rho_vs_omega_star"] if t4 and t4["best"] else None,
        "t4_best_cells": t4["best"]["fixedpoint_cells"] if t4 and t4["best"] else None,
    }
    report = {
        "kind": "recurrence_cbf_superlevel_certificate",
        "config": str(args.config),
        "out_dir": cfg.out_dir,
        "config_hash": cfg.hash(),
        "v_hash": file_hash(op("v.npz")),
        "pi_hash": file_hash(op("pi.npz")),
        "definition": "W = min(g, V_theta); S_m = {W >= m}; witness u = clip(pi_theta). "
                      "Certificate (Theorem A, recurrence form): if lb_W(f(C,[u_lo,u_hi],D)) >= m "
                      "on ALL active cells (ub_W >= m), then S_m is forward-invariant under the "
                      "witness and S_m subset K.  No GFP.  pass_frac is a PASS RATE (certifies "
                      "nothing unless 1.0); rho_inner is the certified set size, valid only where complete.",
        "lattice": {"n_cells": int(n),
                    "shape": [cert.n_cells_px, cert.n_cells_py, cert.n_cells_psi]},
        "omega_star_volume": om["volume"],
        "cell_volume": cell_vol,
        "headline": headline,
        "t4_certificate": t4,
        "fail_diagnosis_m0": margin0_diag,
        "levels": levels,
        "wall_s_total": round(time.time() - t_all, 1),
    }
    fname = f"recurrence_cert_report{args.tag}.json"
    save_json(cfg, fname, report)
    print(f"\n[recurrence-cert] headline: "
          f"{'COMPLETE at m*=%.3f, rho=%.4f' % (headline['m_star'], headline['rho_certified']) if m_star else 'NO complete certificate on this grid'};"
          f"  at m=0 pass {100*headline['pass_frac_at_m0']:.1f}%, inner rho {headline['rho_at_m0']:.4f}")
    print(f"  wrote {op(fname)}")


if __name__ == "__main__":
    main()
