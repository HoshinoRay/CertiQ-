r"""Stage 1 -- ground-truth oracle: Dubins CBVF/HJ grid solve.

Produces \(V_{\mathrm{HJ}}\) on the oracle grid.  The paired action value is
defined by the same object:

    Q_HJ(x,u,d) = V_HJ(f(x,u,d)).

The oracle is the supervised-learning target and reference denominator; the
certificate itself still proves the frozen learned networks post hoc.
"""
from common import load_cfg, out_path, save_json, update_manifest

import numpy as np

from qcbf.oracle.value_iteration import DubinsOracle


def main() -> None:
    cfg = load_cfg()
    print(f"[oracle] {cfg.name}  grid "
          f"{cfg.oracle.n_px}x{cfg.oracle.n_py}x{cfg.oracle.n_psi}, "
          f"{cfg.oracle.n_u} controls x {cfg.oracle.n_d} disturbances, "
          f"gamma_teach={cfg.train.gamma_teach} (deploy={cfg.train.gamma_deploy})")
    oracle = DubinsOracle(cfg.dynamics, cfg.oracle, gamma=cfg.train.gamma_teach)
    sol = oracle.solve(verbose=True)

    frac = float(np.mean(sol["V"] >= 0.0))
    print(f"[oracle] done: {sol['iters']} sweeps, residual "
          f"{sol['residual']:.2e}, Omega* grid fraction {frac:.3f}")

    p = out_path(cfg, "oracle.npz")
    u_hj, q_robust = oracle.fallback_labels(sol["V"])
    np.savez_compressed(p, V=sol["V"], g=sol["g"], history=sol["history"],
                        iters=sol["iters"], wall_s=sol["wall_s"],
                        gamma_teach=cfg.train.gamma_teach,
                        px=oracle.grid.px, py=oracle.grid.py,
                        psi=oracle.grid.psi, u=oracle.grid.u,
                        d=oracle.grid.d, u_hj=u_hj,
                        q_robust=q_robust)
    save_json(cfg, "oracle_report.json", {
        "iters": sol["iters"], "residual": sol["residual"],
        "wall_s": sol["wall_s"], "omega_star_grid_fraction": frac,
        "gamma_teach": cfg.train.gamma_teach,
        "gamma_deploy": cfg.train.gamma_deploy,
        "q_definition": "Q_HJ(x,u,d) = V_HJ(f(x,u,d))"})
    update_manifest(cfg, "oracle", {"iters": sol["iters"],
                                    "residual": sol["residual"],
                                    "wall_s": round(sol["wall_s"], 1)})


if __name__ == "__main__":
    main()
