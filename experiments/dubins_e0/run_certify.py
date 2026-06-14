"""Stage 3 -- strict deployed Q-CBF specification verification (Theorem A).

This stage loads the FROZEN artifact and verifies, with gamma_deploy:

  C1: {V_theta >= 0} subset K.
  C3: witness liveness using u = pi_theta(x):
          min_d Q_theta(x, pi_theta(x), d) >= gamma_deploy V_theta(x) + eps.
  C4: min_d Q_theta(x,u,d) <= min_d V_theta(f(x,u,d)) on the runtime menu
      actions and on the witness action.

The robust one-step decrease is implied by the runtime gate + C4, so it is not
a separate check (see qcbf/certify/spec.py).
"""
import time

from common import file_hash, load_cfg, out_path, save_json, update_manifest

import numpy as np

from qcbf.certify.volume import omega_star_volume
from qcbf.certify.spec import run_strict_spec_certificate
from qcbf.nets.mlp import MLP
from qcbf.oracle.value_iteration import DubinsOracle
from qcbf.verify.bounds import SeqNet
from qcbf.verify.compiler import compile_h3, compile_policy
from qcbf.runtime.filter import CertifiedFilter  # noqa: F401  (API surface)


def main() -> None:
    cfg = load_cfg()
    t_all = time.time()
    v = MLP.load(str(out_path(cfg, "v.npz")))
    q = MLP.load(str(out_path(cfg, "q.npz")))
    pi = MLP.load(str(out_path(cfg, "pi.npz")))

    print(f"[certify] compiling C3 witness h3 = Q(x, clip(pi(x)), d) - gamma_deploy V(x) - eps "
          f"(gamma_deploy={cfg.train.gamma_deploy}, eps={cfg.cert.eps_margin})")
    h3 = compile_h3(pi, q, v, cfg.train.gamma_deploy, cfg.cert.eps_margin,
                    cfg.dynamics.control_max)
    print(f"[certify] compiled stages: "
          f"{[W.shape for W in h3.W]}")

    pol = compile_policy(pi, cfg.dynamics.control_max)
    v_net = SeqNet.from_mlp(v)
    q_net = SeqNet.from_mlp(q)

    print(f"[certify] lattice {cfg.cert.n_cells_px}x{cfg.cert.n_cells_py}x"
          f"{cfg.cert.n_cells_psi}, menu size {cfg.cert.n_u_cells}")
    spec = run_strict_spec_certificate(cfg, v_net, q_net, h3, pol, verbose=True)

    # oracle volume reference (teacher only; not part of the certificate)
    oracle = DubinsOracle(cfg.dynamics, cfg.oracle, gamma=cfg.train.gamma_teach)
    oz = np.load(out_path(cfg, "oracle.npz"))
    if "gamma_teach" in oz and abs(float(oz["gamma_teach"]) - cfg.train.gamma_teach) > 1e-12:
        raise ValueError("oracle.npz gamma_teach does not match config train.gamma_teach")
    V_star = oz["V"]
    om = omega_star_volume(oracle, V_star, cfg)
    print(f"[certify] Vol(Omega*) = {om['volume']:.3f} "
          f"({100*om['frac_of_domain']:.1f}% of domain)")

    best_mask = spec.accepted
    n_best = int(best_mask.sum())
    rho_best = n_best * spec.lattice.cell_volume / om["volume"] if om["volume"] > 0 else 0.0
    spec_pass = spec.report["spec_pass"]
    print(f"[certify] STRICT SPEC: {'PASS' if spec_pass else 'FAIL'}  "
          f"(accepted inner cells {n_best}, rho = {rho_best:.3f})")

    # Provenance stamp: bind this mask to the EXACT config + weights it was
    # certified against, so a later stage (audit) can refuse a stale mask.
    np.savez_compressed(out_path(cfg, "certificate.npz"),
                        accepted=best_mask,
                        lbV=spec.lbV, ubV=spec.ubV, gmin=spec.gmin,
                        superlevel_possible=spec.superlevel_possible,
                        superlevel_inner=spec.superlevel_inner,
                        c1_bad=spec.c1_bad,
                        c3_lb=spec.c3_lb,
                        c3_ok=spec.c3_ok,
                        q_consistency_ok=spec.q_consistency_ok,
                        q_consistency_margin=spec.q_consistency_margin,
                        witness_q_consistency_ok=spec.witness_q_consistency_ok,
                        witness_q_consistency_margin=spec.witness_q_consistency_margin,
                        config_hash=cfg.hash(),
                        v_hash=file_hash(out_path(cfg, "v.npz")),
                        q_hash=file_hash(out_path(cfg, "q.npz")),
                        pi_hash=file_hash(out_path(cfg, "pi.npz")),
                        strict_spec_pass=bool(spec_pass),
                        n_cells=np.array([cfg.cert.n_cells_px,
                                          cfg.cert.n_cells_py,
                                          cfg.cert.n_cells_psi]),
                        gamma_deploy=cfg.train.gamma_deploy)
    report = {
        "strict_spec_pass": spec_pass,
        "strict_spec": spec.report,
        "omega_star": om,
        "best_cells": n_best,
        "best_rho": rho_best,
        "wall_s_stages": spec.report["wall_s_stages"],
        "wall_s_total": round(time.time() - t_all, 1),
        "config_hash": cfg.hash(),
    }
    save_json(cfg, "cert_report.json", report)
    update_manifest(cfg, "certify", {"strict_spec_pass": spec_pass,
                                     "best_cells": n_best,
                                     "best_rho": round(rho_best, 4),
                                     "wall_s": report["wall_s_total"]})


if __name__ == "__main__":
    main()
