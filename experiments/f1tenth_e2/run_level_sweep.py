"""Path 1 -- STRICT INNER-LEVEL certification: certify Omega_c={V_theta>=c}, c>0.

Step-0 showed the deployed object has true C2 holes on the {V_theta~0} boundary
because it distils a ZERO-margin analytic CBF.  Path 1 (corrected): certify an
INNER level c>0 with the discounted CBF condition on B_c = V_theta - c,

    B_c(f(x,pi_b,d)) >= gamma B_c(x)
  <=> h_{B_c}(x,d) = V_theta(f) - gamma V_theta(x) - (1-gamma) c >= 0   on {V_theta~c},

so the verifier's required slack (1-gamma)c can exceed the approximation wobble.

This script runs, with EXACT forward evaluation (no bounds = the ceiling):
  Step 1   analytic ideal level sweep: rho_ideal(c) and the exact ideal ceiling
           min h_{B_c} (must stay >=0 -- the ideal is clean at every level).
  Step 1.5 FREE precursor (no retrain): sweep the shells of the EXISTING learned
           object; min h_{B_c}(c) over {V_theta in [c,c+delta]}.  If some inner c
           is already >=0, the inner level is certifiable with NO retraining.
  (Step 2/3, the shifted-target retrain, live in run_probe_ceiling.py --cert-level.)

    python experiments/f1tenth_e2/run_level_sweep.py            # baseline object
    python experiments/f1tenth_e2/run_level_sweep.py --c2fix    # C2-fix object
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

from qcbf.dynamics.bicycle_accel import BicycleAccelConfig, BicycleAccelModel
from experiments.f1tenth_e2.distill import GAMMA, distill, v_target
from experiments.f1tenth_e2.run_probe_ceiling import ceiling_margins


C_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
DELTA = 0.10                                          # shell thickness


def _rho_curve(Vfun, cfg, n=2_000_000, rng=None):
    """rho(c) = Vol({V>=c}) / Vol({V>=0}) on a uniform (px,py,v) domain sample."""
    rng = rng or np.random.default_rng(0)
    p = rng.uniform(cfg.p_lo, cfg.p_hi, (n, 2))
    v = rng.uniform(0.0, cfg.v_max, n)
    V = Vfun(p[:, 0], p[:, 1], v)
    base = float(np.mean(V >= 0.0))
    return {c: float(np.mean(V >= c) / max(base, 1e-9)) for c in C_GRID}


def analytic_sweep(cfg, model, n=4_000_000, rng=None):
    """Step 1: exact ideal ceiling h_{B_c} on each analytic shell + rho_ideal."""
    rng = rng or np.random.default_rng(1)
    dpro = np.array([[cfg.d_a_max, 0.0], [cfg.d_a_max, cfg.d_delta_max],
                     [cfg.d_a_max, -cfg.d_delta_max], [-cfg.d_a_max, 0.0], [0.0, 0.0]])

    def vt(px, py, v):
        return v_target(cfg, px, py, v)

    p = rng.uniform(cfg.p_lo, cfg.p_hi, (n, 2))
    ps = rng.uniform(-np.pi, np.pi, n)
    v = rng.uniform(0.0, cfg.v_max, n)
    X = np.column_stack([p[:, 0], p[:, 1], ps, v])
    Vx = vt(X[:, 0], X[:, 1], X[:, 3])
    u = np.column_stack([np.full(n, cfg.a_min), np.zeros(n)])     # brake witness
    Vf = np.full(n, np.inf)
    for d in dpro:
        xn = model.step(X, u, np.tile(d, (n, 1)))
        Vf = np.minimum(Vf, vt(xn[:, 0], xn[:, 1], xn[:, 3]))
    m2g = Vf - GAMMA * Vx                              # deployed CBF decrease
    rho = _rho_curve(lambda a, b, c: vt(a, b, c), cfg, rng=rng)
    rows = []
    for c in C_GRID:
        shell = (Vx >= c) & (Vx <= c + DELTA)
        h = m2g[shell] - (1.0 - GAMMA) * c
        rows.append({"c": c, "req_margin": (1 - GAMMA) * c, "rho_ideal": rho[c],
                     "ideal_min_hBc": float(h.min()) if shell.any() else None,
                     "ideal_frac_neg": float(np.mean(h < 0)) if shell.any() else None,
                     "n_shell": int(shell.sum())})
    return rows


def learned_sweep(cfg, model, v, q, pi, n_band=400_000, rng=None):
    """Step 1.5: sweep the shells of an EXISTING learned object (no retrain)."""
    rng = rng or np.random.default_rng(2)
    # sample broadly so every shell c in C_GRID is populated up to c+DELTA
    keep, got = [], 0
    while got < n_band and len(keep) < 60:
        p = rng.uniform(cfg.p_lo, cfg.p_hi, (300_000, 2))
        ps = rng.uniform(-np.pi, np.pi, 300_000)
        vv = rng.uniform(0.0, cfg.v_max, 300_000)
        X = np.column_stack([p[:, 0], p[:, 1], ps, vv])
        Vx = v.forward(X[:, [0, 1, 3]])[:, 0]
        m = (Vx >= -0.02) & (Vx <= max(C_GRID) + DELTA + 0.02)
        if m.any():
            keep.append(X[m]); got += int(m.sum())
    X = np.concatenate(keep)[:n_band]
    mm = ceiling_margins(cfg, model, v, q, pi, X)
    Vx, m2g = mm["Vx"], mm["m2_gamma"]
    rho = _rho_curve(lambda a, b, c: v.forward(np.column_stack([a, b, c]))[:, 0], cfg, rng=rng)
    rows = []
    for c in C_GRID:
        shell = (Vx >= c) & (Vx <= c + DELTA)
        if not shell.any():
            rows.append({"c": c, "n_shell": 0}); continue
        h = m2g[shell] - (1.0 - GAMMA) * c            # h_{B_c} on shell c
        rows.append({"c": c, "req_margin": (1 - GAMMA) * c, "rho_learned": rho[c],
                     "min_hBc": float(h.min()), "p01_hBc": float(np.percentile(h, 1)),
                     "frac_neg": float(np.mean(h < 0)), "n_shell": int(shell.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c2fix", action="store_true")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    t0 = time.time()
    seeds = [0] if args.quick else [0, 1, 2]
    train_kw = (dict(n_samples=20_000, reg_epochs=10, cbf_epochs=12) if args.quick
                else dict(n_samples=80_000, reg_epochs=30, cbf_epochs=30))

    print("=" * 92)
    print("PATH 1 -- STRICT INNER-LEVEL sweep (exact ceiling h_Bc = Vth(f) - g Vth(x) - (1-g)c)")
    print("=" * 92, flush=True)

    print("\n[Step 1] analytic ideal level sweep (the ceiling must stay >= 0):")
    asweep = analytic_sweep(cfg, model)
    print(f"  {'c':>5} | {'(1-g)c':>7} | {'rho_ideal':>9} | {'ideal min h_Bc':>14} | {'frac<0':>7}")
    for r in asweep:
        print(f"  {r['c']:5.2f} | {r['req_margin']:7.3f} | {r['rho_ideal']:9.3f} | "
              f"{r['ideal_min_hBc']:+14.4f} | {r['ideal_frac_neg']:7.3f}")

    print(f"\n[Step 1.5] EXISTING {'C2-fix' if args.c2fix else 'baseline'} learned object "
          f"-- shell sweep, NO retrain ({len(seeds)} seeds):", flush=True)
    per_seed = []
    for s in seeds:
        print(f"  --- seed {s} ---", flush=True)
        v, q, pi, diag = distill(cfg, s, 0.10, c2_fix=args.c2fix, verbose=False, **train_kw)
        rows = learned_sweep(cfg, model, v, q, pi)
        per_seed.append({"seed": s, "mse_v": diag["mse_V_target"], "rows": rows})
        print(f"  {'c':>5} | {'(1-g)c':>7} | {'rho_lrn':>7} | {'min h_Bc':>9} | "
              f"{'p01':>8} | {'frac<0':>7} | {'n':>7}")
        for r in rows:
            if r["n_shell"] == 0:
                continue
            print(f"  {r['c']:5.2f} | {r['req_margin']:7.3f} | {r['rho_learned']:7.3f} | "
                  f"{r['min_hBc']:+9.4f} | {r['p01_hBc']:+8.4f} | {r['frac_neg']:7.3f} | "
                  f"{r['n_shell']:7d}")

    # worst-over-seeds min h_Bc per c, find the smallest certifiable inner level
    print("\n[Verdict] worst-over-seeds min h_Bc per level (>=0 => inner level "
          "pointwise-clean):", flush=True)
    best_c = None
    for c in C_GRID:
        vals = [next((rr for rr in ps["rows"] if rr["c"] == c and rr["n_shell"]), None)
                for ps in per_seed]
        vals = [x for x in vals if x]
        if not vals:
            continue
        wmin = min(x["min_hBc"] for x in vals)
        wfn = max(x["frac_neg"] for x in vals)
        rho = float(np.mean([x["rho_learned"] for x in vals]))
        flag = "  <== CLEAN" if (wmin >= 0.0 and rho > 0.0) else ""
        if wmin >= 0.0 and rho > 0.0 and best_c is None:
            best_c = c
        print(f"  c={c:.2f}: worst min h_Bc {wmin:+.4f}, max frac<0 {wfn:.3f}, "
              f"rho_learned {rho:.3f}{flag}")
    if best_c is not None:
        print(f"\n  => SMALLEST pointwise-clean inner level (no retrain): c={best_c:.2f} "
              f"-- certify Omega_{best_c:.2f} directly.")
    else:
        print("\n  => no inner level is pointwise-clean on the existing object; "
              "Step 2 (shifted-target retrain) is needed.")

    blob = {"analytic": asweep, "per_seed": per_seed, "c2fix": args.c2fix,
            "best_clean_c": best_c, "gamma": GAMMA, "delta": DELTA,
            "wall_s": round(time.time() - t0, 1)}
    tag = "_c2fix" if args.c2fix else ""
    (out / f"level_sweep{tag}.json").write_text(json.dumps(blob, indent=2))
    print(f"\nWrote {out / f'level_sweep{tag}.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
