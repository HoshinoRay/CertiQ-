"""Route-1 build: train the clamped recurrence barrier W_theta = min(g, W_tilde).

Loads the frozen artifact (V_theta as the W_tilde warm-start, pi_theta as the
deployed witness), trains W_tilde with the undiscounted cell-worst recurrence
backup under the FROZEN witness (qcbf.nets.certified_train.train_w_recurrence),
and writes the trained W_tilde as `v.npz` into a NEW artifact dir (with pi.npz /
oracle.npz copied through) so `run_recurrence_cert.py --t4` certifies the SAME
object the runtime deploys.

Certificate object: clamped recurrence barrier (NOT the discounted Q-CBF).
  S_m = {x : min(g(x), W_tilde(x)) >= m},  witness u = clip(pi_theta(x)).
C1 is structural (min(g,.) clamp); only W_tilde is trained; pi_theta is frozen.
"""
from __future__ import annotations

import argparse
import shutil
import time

from common import REPO

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.certify.spec import _control_range
from qcbf.config import ExperimentConfig
from qcbf.dynamics.dubins import g_bounds_on_box
from qcbf.nets.certified_train import train_w_recurrence
from qcbf.nets.mlp import MLP
from qcbf.verify.compiler import compile_policy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO / "experiments/dubins_e0/config_pilot.json"))
    ap.add_argument("--base-dir", default="results/dubins_e0_pilot_2sided_v012",
                    help="Frozen artifact to warm-start W_tilde (=V) and take pi/oracle from.")
    ap.add_argument("--out-dir", default="results/dubins_e0_pilot_wrec",
                    help="Where the trained W_tilde (as v.npz) + pi/oracle are written.")
    ap.add_argument("--m", type=float, default=0.0, help="Recurrence level m.")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--eps-warmup-frac", type=float, default=0.5)
    ap.add_argument("--up-w", type=float, default=1.0)
    ap.add_argument("--down-w", type=float, default=1.0,
                    help="0 => UP-only (let T4 do the sound exclusion afterwards).")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = ExperimentConfig.load(args.config)
    dyn, cert = cfg.dynamics, cfg.cert
    base = REPO / args.base_dir
    out = REPO / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    lat = CellLattice.build(dyn, cert)
    boxes = lat.boxes()
    n = lat.n_cells

    # warm-start W_tilde from V_theta; freeze pi_theta (deployed witness)
    w = MLP.load(str(base / "v.npz"))
    pi = MLP.load(str(base / "pi.npz"))
    pol_net = compile_policy(pi, dyn.control_max)

    # pool = g-possible cells (ub_g >= m): the only cells that can be in {g>=m}.
    gmin, gmax = g_bounds_on_box(dyn, boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    pool = np.flatnonzero(gmax >= args.m)
    print(f"[w-recurrence] cells {n}, pool (ub_g>={args.m}) = {len(pool)}")

    # frozen per-cell CROWN control range of the witness on the pool
    t0 = time.time()
    u_lo, u_hi = _control_range(pol_net, boxes[pool], dyn, cert)
    print(f"[w-recurrence] frozen witness control range ({time.time()-t0:.1f}s); "
          f"|u| range [{float(u_lo.min()):+.2f},{float(u_hi.max()):+.2f}]")

    rep = train_w_recurrence(
        w, boxes, pool, dyn, cert.c2_d_subsplit, args.m, u_lo, u_hi, gmin[pool],
        epochs=args.epochs, batch=args.batch, lr=args.lr,
        up_w=args.up_w, down_w=args.down_w, margin=args.margin, seed=args.seed,
        eps_start=0.0, eps_warmup_frac=args.eps_warmup_frac)

    # write the certified runtime artifact set (cert == deploy)
    w.save(str(out / "v.npz"))
    shutil.copy(base / "pi.npz", out / "pi.npz")
    shutil.copy(base / "oracle.npz", out / "oracle.npz")

    fi, ff = rep["init"], rep["final"]
    print(f"[w-recurrence] recurrence pass (eps=1 cell-worst, active set):")
    print(f"   init : active {fi['active']:6d}  pass {100*fi['pass_frac']:5.1f}%  "
          f"(gleak {fi['fail_gleak']}, wleak {fi['fail_wleak']})")
    print(f"   final: active {ff['active']:6d}  pass {100*ff['pass_frac']:5.1f}%  "
          f"(gleak {ff['fail_gleak']}, wleak {ff['fail_wleak']})")
    print(f"[w-recurrence] wrote {out/'v.npz'} (+ pi/oracle); wall {time.time()-t_all:.1f}s")

    report = {
        "kind": "clamped_recurrence_barrier_training",
        "config": str(args.config),
        "base_dir": args.base_dir, "out_dir": args.out_dir,
        "m": args.m, "epochs": args.epochs, "lr": args.lr,
        "note": "Trained W_tilde for W_theta=min(g,W_tilde); pi_theta frozen; "
                "undiscounted cell-worst recurrence backup; certify with "
                "run_recurrence_cert.py --t4 on out_dir.",
        "train": rep,
    }
    import json
    (out / "wrec_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
