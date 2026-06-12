"""Stage 3 -- the certificate: compile, verify, fixed point, c-sweep.

This stage is the trusted computing base of the experiment.  It loads the
FROZEN artifact, compiles the composite network h3, runs all verified
conditions and emits the certified cell mask plus a fully auditable report.
"""
import time

from common import load_cfg, out_path, save_json, update_manifest

import numpy as np

from qcbf.certify.csearch import omega_star_volume, run_c_sweep
from qcbf.certify.refine import precompute_certificate
from qcbf.nets.mlp import MLP
from qcbf.oracle.value_iteration import DubinsOracle
from qcbf.verify.bounds import SeqNet
from qcbf.verify.compiler import compile_h3
from qcbf.runtime.filter import CertifiedFilter  # noqa: F401  (API surface)


def main() -> None:
    cfg = load_cfg()
    t_all = time.time()
    v = MLP.load(str(out_path(cfg, "v.npz")))
    q = MLP.load(str(out_path(cfg, "q.npz")))
    pi = MLP.load(str(out_path(cfg, "pi.npz")))

    print(f"[certify] compiling h3 = Q(x, clip(pi(x)), d) - gamma V(x) - eps "
          f"(gamma={cfg.train.gamma}, eps={cfg.cert.eps_margin})")
    h3 = compile_h3(pi, q, v, cfg.train.gamma, cfg.cert.eps_margin,
                    cfg.dynamics.omega_max)
    print(f"[certify] compiled stages: "
          f"{[W.shape for W in h3.W]}")

    v_net = SeqNet.from_mlp(v)
    q_net = SeqNet.from_mlp(q)

    print(f"[certify] lattice {cfg.cert.n_cells_px}x{cfg.cert.n_cells_py}x"
          f"{cfg.cert.n_cells_psi}, {cfg.cert.n_u_cells} u-cells")
    pre = precompute_certificate(cfg, v_net, q_net, h3, verbose=True)

    # oracle volume reference
    oracle = DubinsOracle(cfg.dynamics, cfg.oracle)
    V_star = np.load(out_path(cfg, "oracle.npz"))["V"]
    om = omega_star_volume(oracle, V_star, cfg)
    print(f"[certify] Vol(Omega*) = {om['volume']:.3f} "
          f"({100*om['frac_of_domain']:.1f}% of domain)")

    sweep, best_mask = run_c_sweep(pre, cfg, om["volume"], verbose=True)
    n_best = int(best_mask.sum())
    rho_best = max(e["rho"] for e in sweep["entries"])
    gate_d = sweep["gate_d_pass"]
    print(f"[certify] GATE D: {'PASS' if gate_d else 'FAIL'}  "
          f"(best certified cells {n_best}, rho = {rho_best:.3f})")

    np.savez_compressed(out_path(cfg, "certificate.npz"),
                        accepted=best_mask,
                        cand=pre.cand, skip=pre.skip,
                        c3_lb=pre.c3_lb, lbV=pre.lbV, ubV=pre.ubV,
                        c1_ok=pre.c1_ok, c3_ok=pre.c3_ok)
    report = {
        "gate_d_pass": gate_d,
        "funnel": pre.funnel,
        "omega_star": om,
        "sweep": sweep["entries"],
        "best_cells": n_best,
        "best_rho": rho_best,
        "wall_s_stages": {k: round(v_, 1) for k, v_ in pre.wall_s.items()},
        "wall_s_total": round(time.time() - t_all, 1),
        "config_hash": cfg.hash(),
    }
    save_json(cfg, "cert_report.json", report)
    update_manifest(cfg, "certify", {"gate_d_pass": gate_d,
                                     "best_cells": n_best,
                                     "best_rho": round(rho_best, 4),
                                     "wall_s": report["wall_s_total"]})


if __name__ == "__main__":
    main()
