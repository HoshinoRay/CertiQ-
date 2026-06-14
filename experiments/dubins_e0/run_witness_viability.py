"""Empirical (pointwise, simulation) invariant set under the deployed witness.

This is the diagnostic that separates "the witness has a real invariant set" from
"the certificate is conservative".  It forward-simulates u = clip(pi_theta(x))
from every lattice-cell centre and counts how many stay safe (g >= 0) for a
horizon H, under several disturbance signals d_t:

  * d = 0            : nominal (the deterministic invariant set);
  * d = +d_max       : constant adversarial bias (the empirically worst);
  * d = -d_max       : the other constant bias;
  * d ~ U[-d_max,d_max] (a few seeds): typical stochastic;
  * robust (AND)     : survives ALL of the above signals -> a robust-set proxy
                       (an OVER-estimate of the true robust set, since the worst
                       adaptive adversary is at least this bad).

For each signal it records the surviving count at a sweep of horizons H so the
trend (and convergence to the invariant set) is visible.  Sizes are reported as
cell count, % of the domain, % of {V_theta >= 0}, and rho vs Vol(Omega*).

NOTE this is a *pointwise* set (cell centres), NOT the sound cell-worst certified
set; it is the empirical upper bound on what any witness-based certificate can
hope to certify, and it is a property of the WITNESS pi_theta alone (independent
of which value-function version V_theta/W_theta is used downstream).
"""
from __future__ import annotations

import argparse
import json
import time

from common import REPO

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.certify.volume import omega_star_volume
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import DubinsModel
from qcbf.nets.mlp import MLP
from qcbf.oracle.value_iteration import DubinsOracle


def _simulate(model, pi, umax, X0, H_checkpoints, d_signal, rng):
    """Return {H: alive_mask} over the checkpoints for a given disturbance rule.
    d_signal in {'zero','pos','neg','rand'}."""
    H_checkpoints = sorted(H_checkpoints)
    out = {}
    alive = np.ones(len(X0), bool)
    X = X0.copy()
    h = 0
    for Hc in H_checkpoints:
        while h < Hc:
            u = np.clip(pi.forward(X)[:, 0], -umax, umax)
            if d_signal == 'zero':
                d = np.zeros(len(X))
            elif d_signal == 'pos':
                d = np.full(len(X), model.cfg.d_max)
            elif d_signal == 'neg':
                d = np.full(len(X), -model.cfg.d_max)
            else:
                d = rng.uniform(-model.cfg.d_max, model.cfg.d_max, len(X))
            X = model.step(X, u, d)
            alive &= (model.g(X) >= 0.0)
            h += 1
        out[Hc] = alive.copy()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--art-dir", default="results/dubins_e0_pilot_2sided_v012",
                    help="Artifact dir for pi_theta (and V_theta for the {V>=0} denominator).")
    ap.add_argument("--horizons", default="1,5,20,50,100,200,500")
    ap.add_argument("--rand-seeds", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    dyn = cfg.dynamics
    art = REPO / args.art_dir
    model = DubinsModel(dyn)
    pi = MLP.load(str(art / "pi.npz"))
    v = MLP.load(str(art / "v.npz"))
    umax = dyn.control_max
    Hs = [int(s) for s in args.horizons.split(",")]

    lat = CellLattice.build(dyn, cfg.cert)
    boxes = lat.boxes()
    X0 = np.column_stack([0.5 * (boxes[:, 0] + boxes[:, 1]),
                          0.5 * (boxes[:, 2] + boxes[:, 3]),
                          0.5 * (boxes[:, 4] + boxes[:, 5])])
    n = len(X0)
    g0 = model.g(X0)
    V0 = v.forward(X0)[:, 0]
    inK = g0 >= 0.0
    inV = V0 >= 0.0
    cell_vol = float(lat.cell_volume)

    oracle = DubinsOracle(dyn, cfg.oracle, gamma=cfg.train.gamma_teach)
    V_star = np.load(art / "oracle.npz")["V"]
    om = omega_star_volume(oracle, V_star, cfg)

    def sizes(mask):
        c = int(mask.sum())
        return {"cells": c, "pct_domain": round(100 * c / n, 2),
                "pct_Vpos": round(100 * c / max(int(inV.sum()), 1), 2),
                "rho_vs_omega_star": round(c * cell_vol / om["volume"], 4)}

    t0 = time.time()
    rng = np.random.default_rng(0)
    signals = ["zero", "pos", "neg"] + [f"rand{i}" for i in range(args.rand_seeds)]
    # per signal: {H: alive_mask}
    runs = {}
    for sig in signals:
        base = sig[:-1] if sig.startswith("rand") else sig
        r = np.random.default_rng(1234 + (int(sig[-1]) if sig.startswith("rand") else 0))
        runs[sig] = _simulate(model, pi, umax, X0,
                              Hs, "rand" if base == "rand" else base, r)

    # trend tables (count surviving vs H), restricted to starts in {V>=0}
    trend = {}
    for sig in signals:
        trend[sig] = {H: int((runs[sig][H] & inV).sum()) for H in Hs}
    # robust-AND proxy across {zero,pos,neg,rand*}
    robust = {}
    for H in Hs:
        m = np.ones(n, bool)
        for sig in signals:
            m &= runs[sig][H]
        robust[H] = m
    trend["robust_AND"] = {H: int((robust[H] & inV).sum()) for H in Hs}

    Hmax = max(Hs)
    report = {
        "kind": "witness_viability_simulation",
        "art_dir": args.art_dir,
        "lattice_cells": n,
        "in_K": int(inK.sum()),
        "in_Vpos": int(inV.sum()),
        "omega_star": {"volume": om["volume"], "frac_domain": om["frac_of_domain"]},
        "horizons": Hs,
        "trend_count_in_Vpos_vs_H": {k: {str(h): c for h, c in v.items()}
                                     for k, v in trend.items()},
        "converged_sets_at_Hmax": {
            "nominal_d0": sizes(runs["zero"][Hmax]),
            "adversarial_pos": sizes(runs["pos"][Hmax]),
            "adversarial_neg": sizes(runs["neg"][Hmax]),
            "robust_AND_proxy": sizes(robust[Hmax]),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    if args.out:
        (REPO / args.out).write_text(json.dumps(report, indent=2))

    print(f"[viability] witness from {args.art_dir}  cells {n}  "
          f"in-K {int(inK.sum())}  in-V {int(inV.sum())}  "
          f"Omega* frac {om['frac_of_domain']:.3f}")
    print(f"  horizons: {Hs}")
    print(f"  surviving-in-{{V>=0}} count vs H (convergence trend):")
    hdr = "    sig         " + "".join(f"{H:>8d}" for H in Hs)
    print(hdr)
    for sig in signals + ["robust_AND"]:
        row = "".join(f"{trend[sig][H]:>8d}" for H in Hs)
        print(f"    {sig:<11s} {row}")
    print(f"\n  CONVERGED invariant sets at H={Hmax}:")
    for name, s in report["converged_sets_at_Hmax"].items():
        print(f"    {name:<18s}: {s['cells']:6d} cells  "
              f"{s['pct_domain']:5.2f}% domain  {s['pct_Vpos']:5.2f}% of {{V>=0}}  "
              f"rho={s['rho_vs_omega_star']:.4f} of Omega*")
    if args.out:
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
