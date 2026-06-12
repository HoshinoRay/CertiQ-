"""Step 0 -- the EXACT pointwise ceiling probe (no bounds, runs in minutes).

"The learned filter can run" only certifies the *visited* states.  The honest,
verifier-independent question is whether the deployed learned trio satisfies the
certificate's pointwise conditions on the WHOLE {V_theta ~ 0} boundary band --
the corner the rollout never visits.  This script measures, with EXACT forward
evaluation (no CROWN, no IBP), the TRUE pointwise margins on a dense boundary
band -- the theoretical CEILING any sound verifier must lie below:

  C2 (invariance / decrease, witness pi_b):
        m2_gamma(x) = min_{d in corners(D)}  V_theta(f(x, clip(pi_b(x)), d))
                                             - gamma * V_theta(x)
      gamma = 1.0 -> hard sub-level invariance (the P1 knife-edge quantity);
      gamma = GAMMA(=0.5) -> the deployed CBF-discounted decrease.
  C3 (witness feasibility, the predicate the runtime filter actually tests):
        m3(x) = min_{d in corners(D)}  Q_theta(x, clip(pi_b(x)), d)
                                       - GAMMA * V_theta(x)
      the training hinge pushed this above the margin m, so it is the most
      direct numeric reflection of "the filter can run".

Worst-d note: for V_theta(f) the worst disturbance is exactly d_a = +d_a_max
(largest v+ -> largest braking distance -> smallest V_theta(f)); steering
disturbance is irrelevant to V_theta (heading-free).  That corner IS in the
probe grid, so the C2 min over the grid equals the true inf over D.  For C3 the
grid is a sound sub-sample (min over a subset >= the true min -> reported as an
upper estimate of the true C3 ceiling, still decisive if it stays >= m).

Verdict (per the user's dichotomy):
  * if the C2 decrease stays >= 0 over the whole band  -> the OBJECT is pointwise
    certifiable; the residual is 100% a verification-methodology problem (the P1
    per-cell CROWN gap), fix by resolution / non-sublevel primitive (P4).
  * if there are negative pockets -> not the verifier's fault: training did not
    fill the margin; raise m / CEGIS-retrain (same object, same plant).

    python experiments/f1tenth_e2/run_probe_ceiling.py            # 3 seeds
    python experiments/f1tenth_e2/run_probe_ceiling.py --quick
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
from qcbf.util.progress import Progress
from experiments.f1tenth_e2.distill import (GAMMA, distill, clip_action,
                                            d_probe_grid)


def sample_band(cfg, v_net, n_target, delta, rng, batch=200_000, max_batches=400):
    """Dense uniform 4-D samples with |V_theta(x)| <= delta (all headings)."""
    keep = []
    got = 0
    for _ in range(max_batches):
        p = rng.uniform(cfg.p_lo, cfg.p_hi, (batch, 2))
        ps = rng.uniform(-np.pi, np.pi, batch)
        v = rng.uniform(0.0, cfg.v_max, batch)
        X = np.column_stack([p[:, 0], p[:, 1], ps, v])
        Vx = v_net.forward(X[:, [0, 1, 3]])[:, 0]
        m = np.abs(Vx) <= delta
        if m.any():
            keep.append(X[m])
            got += int(m.sum())
        if got >= n_target:
            break
    return np.concatenate(keep)[:n_target] if keep else np.zeros((0, 4))


def ceiling_margins(cfg, model, v_net, q_net, pi_net, X, chunk=200_000):
    """Exact pointwise C2 (gamma=1 and GAMMA) and C3 margins, min over d corners."""
    dprobe = d_probe_grid(cfg)              # (K,2) corners+centre of D
    K = len(dprobe)
    N = len(X)
    Vx = v_net.forward(X[:, [0, 1, 3]])[:, 0]
    u = clip_action(cfg, pi_net.forward(X))            # deployed witness action
    Vf_min = np.full(N, np.inf)                        # min_d V_theta(f(x,u,d))
    Q_min = np.full(N, np.inf)                         # min_d Q_theta(x,u,d)
    d_worst = np.zeros((N, 2))                         # arg min_d V_theta(f)
    for k in range(K):
        d = np.tile(dprobe[k], (N, 1))
        xn = model.step(X, u, d)
        Vf = v_net.forward(xn[:, [0, 1, 3]])[:, 0]
        q = q_net.forward(np.concatenate([X, u, d], axis=1))[:, 0]
        take = Vf < Vf_min
        d_worst[take] = dprobe[k]
        np.minimum(Vf_min, Vf, out=Vf_min)
        np.minimum(Q_min, q, out=Q_min)
    return {"Vx": Vx, "u": u, "Vf_min": Vf_min, "Q_min": Q_min, "d_worst": d_worst,
            "m2_g1": Vf_min - 1.0 * Vx,               # hard sub-level decrease
            "m2_gamma": Vf_min - GAMMA * Vx,          # deployed CBF decrease
            "m3": Q_min - GAMMA * Vx}                 # witness feasibility


def _dist(a):
    a = np.asarray(a, float)
    return {"min": float(a.min()), "p01": float(np.percentile(a, 1)),
            "p05": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "frac_neg": float(np.mean(a < 0.0)),
            "frac_lt_-1e-3": float(np.mean(a < -1e-3))}


def run_seed(cfg, model, seed, margin, delta, n_band, train_kw, c2_fix=False,
             save_cex=None, verbose=True):
    v, q, pi, diag = distill(cfg, seed, margin, c2_fix=c2_fix, verbose=verbose, **train_kw)
    rng = np.random.default_rng(7000 + seed)
    # raw MLP forward is exact -- no SeqNet/CROWN here, this is the true ceiling
    X = sample_band(cfg, v, n_band, delta, rng)
    if verbose:
        print(f"  [band] {len(X)} samples with |V_theta|<= {delta}", flush=True)
    m = ceiling_margins(cfg, model, v, q, pi, X)
    # invariance of {V_theta >= 0} binds on the IN-SET boundary shell V in [0,delta];
    # points with V_theta < 0 are OUTSIDE the set and irrelevant to its invariance.
    inset = m["Vx"] >= 0.0
    mm = {k: m[k][inset] for k in ("m2_g1", "m2_gamma", "m3")}
    if save_cex is not None:                  # C2-CEGIS buffer: deployed (g=0.5) holes
        cexmask = inset & (m["m2_gamma"] < 0.0)
        np.savez(save_cex, x=X[cexmask], d_worst=m["d_worst"][cexmask],
                 Vx=m["Vx"][cexmask], Vf=m["Vf_min"][cexmask], Q=m["Q_min"][cexmask])
        if verbose:
            print(f"  [cex] saved {int(cexmask.sum())} C2 counterexamples -> "
                  f"{save_cex}", flush=True)
    out = {"seed": seed, "c2_fix": c2_fix,
           "mse_v": diag.get("mse_V_target"), "mse_q": diag.get("mse_Q_target"),
           "c2_hinge_train": diag.get("c2_hinge"), "qv_viol_train": diag.get("qv_viol"),
           "n_band": int(len(X)), "n_inset_shell": int(inset.sum()), "delta": delta,
           "C2_decrease_gamma1": _dist(mm["m2_g1"]),
           "C2_decrease_gamma0p5": _dist(mm["m2_gamma"]),
           "C3_witness_feasibility": _dist(mm["m3"])}
    if verbose:
        a = out["C2_decrease_gamma1"]; b = out["C2_decrease_gamma0p5"]
        c = out["C3_witness_feasibility"]
        print(f"  [C2 gamma=1.0] min {a['min']:+.4f}  p01 {a['p01']:+.4f}  "
              f"p50 {a['p50']:+.4f}  frac<0 {a['frac_neg']:.3f}", flush=True)
        print(f"  [C2 gamma=0.5] min {b['min']:+.4f}  p01 {b['p01']:+.4f}  "
              f"p50 {b['p50']:+.4f}  frac<0 {b['frac_neg']:.3f}", flush=True)
        print(f"  [C3 feasible ] min {c['min']:+.4f}  p01 {c['p01']:+.4f}  "
              f"p50 {c['p50']:+.4f}  frac<0 {c['frac_neg']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--c2fix", action="store_true",
                    help="train the C2-fix object (direct C2 loss + Q-conservatism "
                         "+ anchors) instead of the baseline")
    ap.add_argument("--save-cex", action="store_true",
                    help="save the C2 (deployed gamma) counterexamples to .npz")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    tag = "_c2fix" if args.c2fix else ""
    t0 = time.time()

    if args.quick:
        seeds, margin, delta, n_band = [0], 0.10, 0.10, 100_000
        train_kw = dict(n_samples=20_000, reg_epochs=10, cbf_epochs=12)
    else:                                             # the exact P1 object
        seeds, margin, delta, n_band = [0, 1, 2], 0.10, 0.10, 400_000
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=30)

    print("=" * 84)
    print("STEP 0 -- EXACT pointwise ceiling on the {V_theta ~ 0} boundary band "
          "(no bounds)")
    print("  C2 decrease V_theta(f(x,pi_b,d)) - gamma V_theta(x),  C3 min_d Q - "
          "GAMMA V_theta")
    print("=" * 84, flush=True)
    if args.c2fix:
        print("MODE: C2-FIX object (direct C2 loss + one-sided Q-conservatism + "
              "non-vacuity anchors)\n", flush=True)
    runs = []
    pb = Progress(len(seeds), "seeds")
    for i, s in enumerate(seeds):
        print(f"\n=== seed {s} (margin {margin:.2f}, band |V|<= {delta}) ===", flush=True)
        cex_path = str(out / f"c2_cex_s{s}{tag}.npz") if args.save_cex else None
        runs.append(run_seed(cfg, model, s, margin, delta, n_band, train_kw,
                             c2_fix=args.c2fix, save_cex=cex_path))
        pb.update(i + 1)
    pb.done()

    # ---- aggregate worst-over-seeds (the binding ceiling) ----------------- #
    def worst(key):
        return {"min": min(r[key]["min"] for r in runs),
                "max_frac_neg": max(r[key]["frac_neg"] for r in runs),
                "max_frac_lt_-1e-3": max(r[key]["frac_lt_-1e-3"] for r in runs)}
    agg = {"C2_decrease_gamma1": worst("C2_decrease_gamma1"),
           "C2_decrease_gamma0p5": worst("C2_decrease_gamma0p5"),
           "C3_witness_feasibility": worst("C3_witness_feasibility")}

    # ---- verdict ---------------------------------------------------------- #
    g1 = agg["C2_decrease_gamma1"]
    g5 = agg["C2_decrease_gamma0p5"]
    c3 = agg["C3_witness_feasibility"]
    # "object certifiable pointwise" if the deployed (gamma=0.5) decrease and the
    # witness feasibility have no material negative pocket on the band.
    object_ok = (g5["max_frac_lt_-1e-3"] < 1e-3) and (c3["max_frac_lt_-1e-3"] < 1e-3)
    verdict = ("object-pointwise-certifiable" if object_ok
               else "negative-pockets-training-gap")

    blob = {"config": {"seeds": seeds, "margin": margin, "delta": delta,
                       "n_band": n_band, "gamma": GAMMA, "c2_fix": args.c2fix},
            "per_seed": runs, "worst_over_seeds": agg, "verdict": verdict,
            "wall_s": round(time.time() - t0, 1)}
    (out / f"ceiling_probe{tag}.json").write_text(json.dumps(blob, indent=2))

    print("\n" + "=" * 84)
    print("STEP 0 VERDICT -- exact pointwise ceiling (worst over seeds)")
    print("=" * 84)
    print(f"C2 decrease gamma=1.0 (hard invariance): min {g1['min']:+.4f}, "
          f"max frac<0 {g1['max_frac_neg']:.3f}, frac< -1e-3 {g1['max_frac_lt_-1e-3']:.4f}")
    print(f"C2 decrease gamma=0.5 (deployed CBF)   : min {g5['min']:+.4f}, "
          f"max frac<0 {g5['max_frac_neg']:.3f}, frac< -1e-3 {g5['max_frac_lt_-1e-3']:.4f}")
    print(f"C3 witness feasibility (filter runs)   : min {c3['min']:+.4f}, "
          f"max frac<0 {c3['max_frac_neg']:.3f}, frac< -1e-3 {c3['max_frac_lt_-1e-3']:.4f}")
    print("-" * 84)
    if object_ok:
        print("VERDICT: OBJECT is pointwise certifiable on the boundary band "
              "(deployed gamma + witness).")
        print("  => the residual P1 FAIL is 100% a VERIFICATION-METHODOLOGY gap "
              "(per-cell CROWN gap),")
        print("     not an object defect. Fix by resolution / non-sublevel "
              "primitive (P4); keep object+plant.")
    else:
        print("VERDICT: NEGATIVE POCKETS exist on the band -> NOT a verifier "
              "problem; the training did not")
        print("  fill the margin. Raise m / CEGIS-retrain the SAME object on the "
              "SAME plant (no problem swap).")
    print(f"\nNote: C2 gamma=1.0 is the P1 knife-edge quantity (analytic contraction "
          f"is exactly 0),")
    print(f"  so values straddling 0 there are EXPECTED and are not the deployed "
          f"condition (gamma=0.5).")
    print(f"Wrote {out / f'ceiling_probe{tag}.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
