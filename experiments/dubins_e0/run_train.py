"""Stage 2 -- train and FREEZE the deployed artifact (V_theta, Q_theta, pi_phi).

Pipeline (training is never part of the trusted computing base):

  1. V_theta : regression on V_HJ at grid nodes + random interpolated states.
  2. Q_theta : regression on Q_HJ(x,u,d) = V_HJ(f(x,u,d)).
  3. pi_phi  : regression on the oracle robust-greedy witness labels
               u_HJ(x) = argmax_u min_d Q_HJ(x,u,d),
     then witness-margin fine-tuning (V, Q frozen): hinge-maximize
               min_k Q_theta(x, pi(x), d_k) - gamma_deploy V_theta(x)
     toward m_target on states inside Omega*.  This is non-vacuity
     pressure for C3, not a soundness assumption.

The oracle teacher uses the discounted safety backup (discount gamma_teach), so
V_HJ already carries a positive deployed-condition margin (~ (1 - gamma_deploy) V)
on the safe interior before any fine-tuning.

Weights are exported to .npz and frozen; later stages only ever read them.
"""
import time

from common import file_hash, load_cfg, out_path, save_json, update_manifest

import numpy as np

from qcbf.nets.mlp import MLP, train_regression, finetune_witness_margin
from qcbf.oracle.value_iteration import DubinsOracle


def main() -> None:
    cfg = load_cfg()
    tr, dyn, nets = cfg.train, cfg.dynamics, cfg.nets
    rng = np.random.default_rng(tr.seed)
    t_all = time.time()

    oracle = DubinsOracle(dyn, cfg.oracle, gamma=tr.gamma_teach)
    z = np.load(out_path(cfg, "oracle.npz"))
    if "gamma_teach" in z and abs(float(z["gamma_teach"]) - tr.gamma_teach) > 1e-12:
        raise ValueError("oracle.npz gamma_teach does not match config train.gamma_teach")
    V_star = z["V"]
    g = oracle.grid
    PX, PY, PSI = np.meshgrid(g.px, g.py, g.psi, indexing="ij")
    X_grid = np.stack([PX, PY, PSI], axis=-1).reshape(-1, 3)

    # ---- V regression ------------------------------------------------- #
    n_extra = len(X_grid) // 2
    X_extra = np.column_stack([
        rng.uniform(dyn.p_lo, dyn.p_hi, n_extra),
        rng.uniform(dyn.p_lo, dyn.p_hi, n_extra),
        rng.uniform(-np.pi, np.pi, n_extra)])
    X_v = np.vstack([X_grid, X_extra])

    Y_v = oracle.interp_V(V_star, X_v)
    v = MLP([3, *nets.v_hidden, 1], seed=tr.seed)
    print(f"[train] V_theta on {len(X_v)} samples")
    mse_v = train_regression(v, X_v, Y_v, tr.epochs_v, tr.batch, tr.lr,
                             seed=tr.seed, tag="V")

    # ---- Q regression ------------------------------------------------- #
    n = tr.n_q_samples
    Xq = np.column_stack([
        rng.uniform(dyn.p_lo, dyn.p_hi, n),
        rng.uniform(dyn.p_lo, dyn.p_hi, n),
        rng.uniform(-np.pi, np.pi, n),
        rng.uniform(-dyn.omega_max, dyn.omega_max, n),
        rng.uniform(-dyn.d_max, dyn.d_max, n)])
    Yq = oracle.q_star(V_star, Xq[:, :3], Xq[:, 3], Xq[:, 4])
    q = MLP([5, *nets.q_hidden, 1], seed=tr.seed + 1)
    print(f"[train] Q_theta on {n} samples")
    mse_q = train_regression(q, Xq, Yq, tr.epochs_q, tr.batch, tr.lr,
                             seed=tr.seed + 1, tag="Q")

    # ---- pi regression + witness-margin fine-tuning -------------------- #
    u_flat, _ = oracle.fallback_labels(V_star)
    Y_pi = u_flat.reshape(-1).astype(np.float64)
    pi = MLP([3, *nets.pi_hidden, 1], seed=tr.seed + 2)
    print(f"[train] pi_phi regression on {len(X_grid)} fallback labels")
    mse_pi = train_regression(pi, X_grid, Y_pi, tr.epochs_pi, tr.batch, tr.lr,
                              seed=tr.seed + 2, tag="pi")

    inside = oracle.interp_V(V_star, X_grid) >= 0.0
    X_m = X_grid[inside]
    anchor = Y_pi[inside]
    d_grid = np.linspace(-dyn.d_max, dyn.d_max, tr.margin_d_grid)
    print(f"[train] witness-margin fine-tuning on {len(X_m)} states "
          f"(m_target={tr.margin_target}, gamma_deploy={tr.gamma_deploy})")
    finetune_witness_margin(pi, q, v, X_m, d_grid, tr.gamma_deploy, dyn.omega_max,
                            tr.margin_target, tr.epochs_margin, tr.batch,
                            tr.lr * 0.5, seed=tr.seed + 3,
                            label_anchor=anchor)

    # ---- freeze -------------------------------------------------------- #
    paths = {}
    for name, net in (("v", v), ("q", q), ("pi", pi)):
        p = out_path(cfg, f"{name}.npz")
        net.save(str(p))
        paths[name] = file_hash(p)

    # quick witness-margin diagnostic on Omega* states
    from qcbf.nets.mlp import policy_forward
    if len(X_m) > 0:
        u = policy_forward(pi, X_m, dyn.omega_max)
        qmin = np.min(np.stack([
            q(np.column_stack([X_m, u, np.full(len(X_m), dk)])).ravel()
            for dk in d_grid]), axis=0)
        margin = qmin - tr.gamma_deploy * v(X_m).ravel()
        m_mean = float(margin.mean())
        m_p05 = float(np.percentile(margin, 5))
        m_frac = float(np.mean(margin >= cfg.cert.eps_margin))
    else:
        m_mean, m_p05, m_frac = float("nan"), float("nan"), 0.0
    diag = {"mse_v": mse_v, "mse_q": mse_q, "mse_pi": mse_pi,
            "witness_margin_mean": m_mean,
            "witness_margin_p05": m_p05,
            "witness_margin_frac_ge_eps": m_frac,
            "n_margin_states": int(len(X_m)),
            "wall_s": time.time() - t_all}
    print(f"[train] witness margin on Omega*: mean {diag['witness_margin_mean']:+.3f}, "
          f"p05 {diag['witness_margin_p05']:+.3f}, "
          f">=eps frac {diag['witness_margin_frac_ge_eps']:.3f}")
    save_json(cfg, "train_report.json", diag)
    update_manifest(cfg, "train", {**paths,
                                   "wall_s": round(diag["wall_s"], 1)})


if __name__ == "__main__":
    main()
