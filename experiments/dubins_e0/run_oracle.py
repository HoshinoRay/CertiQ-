"""Stage 1 -- ground-truth oracle: discounted-free Bellman-Isaacs VI.

Produces V* on the oracle grid (REFERENCE ONLY: the oracle quantifies how
much of the maximal robust invariant set the certificate recovers; it never
enters the soundness argument).
"""
from common import load_cfg, out_path, save_json, update_manifest

import numpy as np

from qcbf.oracle.value_iteration import DubinsOracle


def main() -> None:
    cfg = load_cfg()
    print(f"[oracle] {cfg.name}  grid "
          f"{cfg.oracle.n_px}x{cfg.oracle.n_py}x{cfg.oracle.n_psi}, "
          f"{cfg.oracle.n_u} controls x {cfg.oracle.n_d} disturbances")
    oracle = DubinsOracle(cfg.dynamics, cfg.oracle)
    sol = oracle.solve(verbose=True)

    frac = float(np.mean(sol["V"] >= 0.0))
    print(f"[oracle] done: {sol['iters']} sweeps, residual "
          f"{sol['residual']:.2e}, Omega* grid fraction {frac:.3f}")

    p = out_path(cfg, "oracle.npz")
    np.savez_compressed(p, V=sol["V"], g=sol["g"], history=sol["history"],
                        iters=sol["iters"], wall_s=sol["wall_s"])
    save_json(cfg, "oracle_report.json", {
        "iters": sol["iters"], "residual": sol["residual"],
        "wall_s": sol["wall_s"], "omega_star_grid_fraction": frac})
    update_manifest(cfg, "oracle", {"iters": sol["iters"],
                                    "residual": sol["residual"],
                                    "wall_s": round(sol["wall_s"], 1)})


if __name__ == "__main__":
    main()
