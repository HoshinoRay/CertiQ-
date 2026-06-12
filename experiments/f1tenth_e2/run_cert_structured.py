"""F1TENTH E2-L / STRUCTURED -- sound Gate-D certificate of the STRUCTURED learned
value  V_theta(px,py,v) = C_theta(px,py) - D(v) - margin  (Path B).

Why structured: the exact-ceiling sweep showed a black-box MLP V_theta floors at a
worst-case one-step contraction error ~0.076 (it cannot reproduce the analytic
1-Lipschitz braking cancellation), so NO sound primitive can close it.  Replacing
the SPEED axis with the analytic braking distance D(v) and learning only the
~1-Lipschitz clearance C_theta(p) drops the exact worst-case to ~-0.01 -- so a
fine-enough SOUND certificate can now close on a non-empty subset.

This does NOT require the exact ceiling to be >=0 first: we hand the structured
object to the verifier and let SOUND sub-level pruning decide whether the thin
residual tail is boundary fuzz (pruned, large rho survives) or cascades to empty.

  * primitive: P1 direct composition (qcbf/certify/direct.py, structured variants)
  * candidate: Omega_c = {V_theta >= c};  c-search prunes the failing boundary.
  * IRON RULE: analytic ideal must pass first (reuses run_cert.py, 86,580 @ 80^3).
  * report: rho = Vol(Omega_cert)/Vol(Omega*), per resolution.

    python experiments/f1tenth_e2/run_cert_structured.py --quick
    python experiments/f1tenth_e2/run_cert_structured.py
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
                                         brake_cbf_bounds, g_bounds_sq)
from qcbf.certify.direct import (vtheta_box_bounds_structured,
                                 crown_brake_successor_lb_structured)
from qcbf.verify.bounds import SeqNet
from qcbf.util.progress import Progress
from experiments.f1tenth_e2.distill import train_clearance_net, clearance
from experiments.f1tenth_e2.run_cert import certified_set, verify_braking_decrease
from experiments.f1tenth_e2.run_cert_p1 import _grid


# --------------------------------------------------------------------------- #
def _eval_boxes(cfg, c_net, lo, hi, npsi, chunk=8192):
    """Per-box sound (ubV, lbV, gmin, lbVbrake-worst-heading) for (px,py,v) boxes."""
    lbV, ubV = vtheta_box_bounds_structured(
        cfg, c_net, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2], chunk)
    gmin = g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    lbb = np.full(len(lo), np.inf)
    for h in range(npsi):
        lbb = np.minimum(lbb, crown_brake_successor_lb_structured(
            cfg, c_net, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1],
            ps[h], ps[h + 1], lo[:, 2], hi[:, 2], chunk))
    return ubV, lbV, gmin, lbb


def _split_xy(lo, hi):
    """Split each (px,py,v) box into 2x2 along px,py (v kept; D(v) is exact)."""
    mx = 0.5 * (lo[:, 0] + hi[:, 0]); my = 0.5 * (lo[:, 1] + hi[:, 1])
    out_lo, out_hi = [], []
    for sx in (0, 1):
        for sy in (0, 1):
            L = lo.copy(); H = hi.copy()
            L[:, 0] = np.where(sx, mx, lo[:, 0]); H[:, 0] = np.where(sx, hi[:, 0], mx)
            L[:, 1] = np.where(sy, my, lo[:, 1]); H[:, 1] = np.where(sy, hi[:, 1], my)
            out_lo.append(L); out_hi.append(H)
    return np.concatenate(out_lo), np.concatenate(out_hi)


def _bank(cfg, c_net, c, lo, hi, npsi, chunk):
    """Evaluate boxes, bank certified volume, return the still-unresolved boxes.

    OUT (ubV<c) dropped; resolved_ok (gmin>=0 & lbb>=c) certified (volume added
    if also fully inside lbV>=c -- a closed straddler stays in {V>=c} but its
    out-of-set part is not counted).  Coarser npsi is a SOUND (looser) lbb, so
    banking at any npsi is sound."""
    ubV, lbV, gmin, lbb = _eval_boxes(cfg, c_net, lo, hi, npsi, chunk)
    vol = np.prod(hi - lo, axis=1)
    resolved_ok = (gmin >= 0.0) & (lbb >= c)
    cert = float(vol[resolved_ok & (lbV >= c)].sum())
    todo = ~((ubV < c) | resolved_ok)
    return cert, lo[todo], hi[todo]


def _certify_level(cfg, c_net, c, lo, hi, npsi, maxdepth=6, chunk=8192):
    """Sound POSITION branch-and-bound that {V_theta >= c} is brake-invariant.

    The Step-5 npsi benchmark showed the verified undershoot is NOT heading-bound
    (npsi 12->48 barely moves it) -- it is C_theta's spatial CROWN gap.  Splitting
    (px,py) drives it to the TRUE one-step clearance change C(p)-C(p+) (<= dt*v),
    which the analytic D recovery exactly cancels, so the undershoot -> <=0 and the
    box closes.  Heading stays at a fixed modest npsi.  Unknown at maxdepth -> FAIL.
    """
    cert_vol = 0.0
    cap = 600_000                                       # OOM guard: too many -> fail
    for depth in range(maxdepth + 1):
        if len(lo) == 0:
            return True, cert_vol
        cv, lo, hi = _bank(cfg, c_net, c, lo, hi, npsi, chunk)
        cert_vol += cv
        if len(lo) == 0:
            return True, cert_vol
        if depth == maxdepth or len(lo) > cap:
            return False, cert_vol                      # residual unknown -> fail
        lo, hi = _split_xy(lo, hi)
    return True, cert_vol


# --------------------------------------------------------------------------- #
def soundness_check(cfg, model, c_net, seed=0, M=4000, K=400):
    """Randomized: structured brake-successor lb <= true V_theta(f_brake) for
    sampled (x in cell, psi in slice, d in D).  Returns the min sample margin."""
    rng = np.random.default_rng(seed)
    px_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.2, M); px_hi = px_lo + rng.uniform(.02, .15, M)
    py_lo = rng.uniform(cfg.p_lo, cfg.p_hi - 0.2, M); py_hi = py_lo + rng.uniform(.02, .15, M)
    v_lo = rng.uniform(0, cfg.v_max - 0.3, M); v_hi = v_lo + rng.uniform(.02, .2, M)
    ps_lo = rng.uniform(-np.pi, np.pi - 0.5, M); ps_hi = ps_lo + rng.uniform(.02, .4, M)
    lb = crown_brake_successor_lb_structured(cfg, c_net, px_lo, px_hi, py_lo, py_hi,
                                             ps_lo, ps_hi, v_lo, v_hi)

    def Vth(x):
        return c_net.forward(x[:, :2])[:, 0] - _D(cfg, x[:, 3]) - cfg.cbf_margin

    u = rng.random((M, K, 4))
    px = px_lo[:, None] + u[..., 0] * (px_hi - px_lo)[:, None]
    py = py_lo[:, None] + u[..., 1] * (py_hi - py_lo)[:, None]
    v = v_lo[:, None] + u[..., 2] * (v_hi - v_lo)[:, None]
    ps = ps_lo[:, None] + u[..., 3] * (ps_hi - ps_lo)[:, None]
    da = rng.uniform(-cfg.d_a_max, cfg.d_a_max, (M, K))
    npx = px + cfg.dt * v * np.cos(ps); npy = py + cfg.dt * v * np.sin(ps)
    nv = np.clip(v + cfg.dt * (cfg.a_min + da), 0.0, cfg.v_max)
    Vp = (c_net.forward(np.stack([npx, npy], -1).reshape(-1, 2))[:, 0]
          - _D(cfg, nv.ravel()) - cfg.cbf_margin).reshape(M, K)
    return float((Vp - lb[:, None]).min())


def _D(cfg, v):
    from qcbf.dynamics.bicycle_accel import brake_distance
    return brake_distance(cfg, v)


# --------------------------------------------------------------------------- #
def certify_structured(cfg, c_net, npx, npy, nv, npsi, c_grid, ideal_vol,
                       maxdepth=6, chunk=8192, verbose=True):
    """Sound sub-level c-search on the STRUCTURED V_theta with ADAPTIVE boundary
    refinement (branch-and-bound).  rho = certified volume / ideal volume.
    """
    lo, hi, cell_vol = _grid(cfg, npx, npy, nv)
    # V bounds + C1 over the WHOLE grid (one CROWN pass); worst-heading brake
    # successor only for CANDIDATES (ubV>=cmin) -- the expensive psi loop skips
    # the >50% of cells already outside every candidate level.
    lbV, ubV = vtheta_box_bounds_structured(
        cfg, c_net, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2],
        chunk, progress="V-bounds" if verbose else None)
    gmin = g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])
    cmin = float(min(c_grid))
    cand0 = ubV >= cmin
    lbb = np.full(len(lo), -np.inf)
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    cl, ch = lo[cand0], hi[cand0]
    lbb_c = np.full(int(cand0.sum()), np.inf)
    pb = Progress(npsi, "base-psi") if verbose else None
    for h in range(npsi):
        lbb_c = np.minimum(lbb_c, crown_brake_successor_lb_structured(
            cfg, c_net, cl[:, 0], ch[:, 0], cl[:, 1], ch[:, 1],
            ps[h], ps[h + 1], cl[:, 2], ch[:, 2], chunk))
        if pb is not None:
            pb.update(h + 1)
    if pb is not None:
        pb.done()
    lbb[cand0] = lbb_c
    if verbose:
        print(f"  [base] {len(lo)} cells, {int(cand0.sum())} candidates", flush=True)

    best = None
    rho_curve = {}
    for c in c_grid:
        cand = ubV >= c                               # boxes touching {V>=c}
        if not cand.any():
            rho_curve[f"{c:.2f}"] = 0.0
            continue
        resolved = (gmin[cand] >= 0.0) & (lbb[cand] >= c)
        inside = lbV[cand] >= c
        vol = np.prod(hi[cand] - lo[cand], axis=1)
        cert_vol = float(vol[resolved & inside].sum())
        todo = ~resolved                              # unresolved candidates -> refine
        tlo, thi = lo[cand][todo], hi[cand][todo]
        passed, ref_vol = _certify_level(cfg, c_net, c, tlo, thi, npsi,
                                         maxdepth, chunk)
        cert_vol += ref_vol
        rho = float(cert_vol / max(ideal_vol, 1e-9))
        rho_curve[f"{c:.2f}"] = rho if passed else 0.0
        if verbose:
            print(f"    c={c:.2f}: cand {int(cand.sum())}, refine {int(todo.sum())}"
                  f" -> {'PASS' if passed else 'fail'}  rho={rho if passed else 0:.3f}",
                  flush=True)
        if passed and best is None:
            best = {"c": float(c), "rho": rho, "cert_vol": cert_vol}
            break
    return {
        "empty": False,
        "closes_at_c": best["c"] if best else None,
        "rho": best["rho"] if best else 0.0,
        "cert_vol": best["cert_vol"] if best else 0.0,
        "rho_curve": rho_curve,
        "undershoot_min": float((ubV[ubV >= 0.0] - lbb[ubV >= 0.0]).min())
        if (ubV >= 0.0).any() else None,
        "cell_m": round((cfg.p_hi - cfg.p_lo) / npx, 4),
    }


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    # target inner levels (pruning: Omega_cert subset {V>=0}); low c straddles
    # too much boundary to close under the OOM cap -- the inner level is the point.
    c_grid = (0.10, 0.15, 0.20, 0.25)
    t0 = time.time()

    if args.quick:
        # small base grid; heading refinement (npsi doubles per B&B depth) + position
        # splits supply the precision -- not a big brute-force grid (Step-5 strategy)
        res_list, npsi = [(60, 60, 40)], 12
        train_kw = dict(epochs=30, n=80_000)
    else:
        res_list, npsi = [(80, 80, 60), (100, 100, 60)], 12
        train_kw = dict(epochs=45, n=120_000)

    print("=" * 88)
    print("STRUCTURED V = C_theta(p) - D(v) -- sound Gate-D via direct composition + pruning")
    print("=" * 88, flush=True)

    C, mseC = train_clearance_net(cfg, seed=0, width=32, **train_kw)
    cnet = SeqNet.from_mlp(C)
    sm = soundness_check(cfg, model, cnet)
    print(f"[train] C_theta MSE(clearance)={mseC:.4f}; structured brake-succ lb "
          f"soundness min sample margin {sm:+.2e} "
          f"({'OK' if sm >= -1e-7 else 'UNSOUND!'})", flush=True)

    # IRON RULE: analytic ideal first (same plant, reuses the E2 PASS)
    refnp = res_list[-1]
    cset = certified_set(cfg, model, *refnp)
    dec = verify_braking_decrease(cfg, model, refnp[0], refnp[1], max(24, refnp[2] // 2), refnp[2])
    print(f"[ideal] analytic V @ {refnp}: {cset['accepted']} cells certified, "
          f"min braking decrease {dec:+.5f} -> {'PASS' if cset['accepted']>0 and dec>=-1e-9 else 'FAIL'}",
          flush=True)

    ideal_vol = cset["volume"]
    runs = []
    for res in res_list:
        print(f"\n=== structured cert  res={res}  npsi={npsi} (ideal vol "
              f"{ideal_vol:.2f}) ===", flush=True)
        r = certify_structured(cfg, cnet, *res, npsi, c_grid, ideal_vol)
        r["res"] = list(res)
        runs.append(r)
        print(f"  [struct] closes@c={r.get('closes_at_c')}  "
              f"rho={r.get('rho', 0):.3f}  undershoot_min={r.get('undershoot_min')}",
              flush=True)

    blob = {"mse_C": mseC, "soundness_min_margin": sm,
            "ideal_cells": cset["accepted"], "ideal_min_decrease": dec,
            "npsi": npsi, "runs": runs, "wall_s": round(time.time() - t0, 1)}
    (out / "structured_report.json").write_text(json.dumps(blob, indent=2))

    print("\n" + "=" * 88)
    print("STRUCTURED Gate-D result (object: C_theta(p)-D(v); plant FIXED)")
    print("=" * 88)
    print(f"IDEAL: {cset['accepted']} cells (iron rule).  C_theta MSE={mseC:.4f}, "
          f"bound soundness {sm:+.1e}")
    print(f"{'cell(m)':>8} | {'closes@c':>9} | {'rho':>6} | {'undershoot_min':>14}")
    for r in runs:
        print(f"{r['cell_m']:8.3f} | {str(r.get('closes_at_c')):>9} | "
              f"{r.get('rho', 0):6.3f} | {str(r.get('undershoot_min')):>14}")
    any_pass = any(r.get("rho", 0) > 0 for r in runs)
    print(f"\nVerdict: {'PASS -- non-empty sound certified subset of the LEARNED structured object' if any_pass else 'FAIL -- pruning erodes empty at these resolutions (finer needed)'}")
    print(f"Wrote {out / 'structured_report.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
