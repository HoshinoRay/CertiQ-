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

from qcbf.nets.mlp import (MLP, train_regression, train_v_cbf,
                           train_q_oneside, finetune_witness_margin)
from qcbf.nets.certified_train import (precompute_lbVf, train_q_certified,
                                       train_v_certified)
from qcbf.certify.cells import CellLattice
from qcbf.certify.spec import _menu
from qcbf.verify.bounds import SeqNet
from qcbf.verify.conditions import v_cell_bounds
from qcbf.dynamics.dubins import g_bounds_on_box
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
    g_v = oracle.model.g(X_v)            # exact safety margin at the V samples
    u_dec = np.linspace(-dyn.control_max, dyn.control_max, tr.v_dec_n_u)
    d_dec = np.linspace(-dyn.d_max, dyn.d_max, tr.v_dec_n_d)
    v = MLP([3, *nets.v_hidden, 1], seed=tr.seed)
    print(f"[train] V_theta CBF on {len(X_v)} samples "
          f"(floor m={tr.c1_floor_margin}, decrease m={tr.v_dec_margin})")
    mse_v = train_v_cbf(v, X_v, Y_v, g_v, oracle.model, u_dec, d_dec,
                        tr.gamma_deploy, tr.epochs_v, tr.batch, tr.lr,
                        tr.c1_floor_w, tr.c1_floor_margin,
                        tr.v_dec_w, tr.v_dec_margin, tr.teacher_fit_w,
                        seed=tr.seed, tag="V")

    # ---- lattice + exact g box bounds (cell-worst quantities) --------- #
    cert = cfg.cert
    lat = CellLattice.build(dyn, cert)
    boxesL = lat.boxes()
    gmin, _ = g_bounds_on_box(dyn, boxesL[:, 0], boxesL[:, 1],
                              boxesL[:, 2], boxesL[:, 3])

    # ---- V certified tightening (cell-worst CROWN-IBP; V analog of Q) -- #
    # V is otherwise trained pointwise, so its CELL-WORST slack is the binding
    # barrier: ub V leaks {V>=0} out of K (C1) and lb V(f) is too low for the
    # witness band (C4).  Train V's own sound cell-worst IBP bounds:
    #   C1 : ub_IBP V(C) < -m            on cells with g<0
    #   dec: min_d lb_IBP V(f(C,u*,D)) >= gamma*ub_IBP V(C) + m
    # IBP looser than CROWN => implies the deployed CROWN C1/band.  Runs BEFORE Q
    # so Q tracks the tighter V.  A teacher anchor keeps {V>=0} from collapsing.
    vcert_rep = {}
    if tr.cert_v_w > 0.0:
        v_net0 = SeqNet.from_mlp(v)
        _, ubV0 = v_cell_bounds(v_net0, boxesL, np.arange(lat.n_cells),
                                lat.n_cells, cert.chunk)
        # Boundary band: every cell near/in {V>=0} (covers ALL C1-bad cells,
        # ubV0>=0 & g<0).  C1-floor-only uses FULL coverage (no sampling) so every
        # unsafe boundary cell is driven down; only sample if the decrease push is
        # on (then the successor forwards make it expensive).
        vpool = np.flatnonzero(ubV0 >= -0.2)
        if tr.cert_v_dec_w > 0.0 and len(vpool) > tr.cert_n_cells:
            vpool = np.sort(rng.choice(vpool, tr.cert_n_cells, replace=False))
        menu_v = _menu(dyn, cert)
        vc = boxesL[vpool]
        xc_v = np.column_stack([0.5 * (vc[:, 0] + vc[:, 1]),
                                0.5 * (vc[:, 2] + vc[:, 3]),
                                0.5 * (vc[:, 4] + vc[:, 5])])
        anchor_Y = oracle.interp_V(V_star, xc_v)
        side = "C1-floor only" if tr.cert_v_dec_w == 0.0 else "C1 floor + decrease"
        print(f"[train] V certified ({side}, cell-worst CROWN-IBP) on {len(vpool)} "
              f"cells (c1 w={tr.cert_v_w} m={tr.cert_v_c1_margin}, "
              f"dec w={tr.cert_v_dec_w} m={tr.cert_v_dec_margin}, "
              f"anchor={tr.cert_v_anchor_w} on safe cells)")
        vcert_rep = train_v_certified(
            v, boxesL, vpool, dyn, menu_v, tr.cert_d_subsplit, tr.gamma_deploy,
            tr.cert_v_w, tr.cert_v_c1_margin, tr.cert_v_dec_w, tr.cert_v_dec_margin,
            tr.cert_v_anchor_w, tr.epochs_v, 512, tr.lr * 0.5,
            gmin[vpool], anchor_Y, seed=tr.seed + 7,
            eps_start=tr.cert_eps_start, eps_warmup_frac=tr.cert_eps_warmup_frac)
        print(f"  C1 leak frac {vcert_rep['c1_leak_frac_init']:.3f} -> "
              f"{vcert_rep['c1_leak_frac_final']:.3f} | band-open frac "
              f"{vcert_rep['band_open_frac_init']:.3f} -> "
              f"{vcert_rep['band_open_frac_final']:.3f}")

    # ---- V cell bounds (after any tightening; V now frozen) ------------ #
    v_net = SeqNet.from_mlp(v)
    lbV, ubV = v_cell_bounds(v_net, boxesL, np.arange(lat.n_cells),
                             lat.n_cells, cert.chunk)

    # ---- Q regression (tracks the now-frozen, tightened V) ------------- #
    n = tr.n_q_samples
    Xq = np.column_stack([
        rng.uniform(dyn.p_lo, dyn.p_hi, n),
        rng.uniform(dyn.p_lo, dyn.p_hi, n),
        rng.uniform(-np.pi, np.pi, n),
        rng.uniform(-dyn.omega_max, dyn.omega_max, n),
        rng.uniform(-dyn.d_max, dyn.d_max, n)])
    q = MLP([5, *nets.q_hidden, 1], seed=tr.seed + 1)
    print(f"[train] Q_theta tracks V_theta(f) from below on {n} samples "
          f"(C4 one-sided, w={tr.c4_oneside_w}, margin={tr.c4_oneside_margin})")
    mse_q = train_q_oneside(q, v, Xq, oracle.model, tr.epochs_q, tr.batch,
                            tr.lr, tr.c4_oneside_w, tr.c4_oneside_margin,
                            seed=tr.seed + 1, tag="Q")

    # ---- Q certified tightening (two-sided CROWN-IBP) ------------------ #
    cand = np.flatnonzero((ubV >= 0.0) & (gmin >= 0.0))
    qcert_rep = {}
    if len(cand) and tr.cert_c4_w > 0.0:        # EXPERIMENTAL, off by default
        pool = np.sort(rng.choice(cand, min(tr.cert_n_cells, len(cand)),
                                  replace=False))
        menu_c = _menu(dyn, cert)
        side = "two-sided C4+C3" if tr.cert_c3_w > 0.0 else "one-sided C4"
        print(f"[train] Q certified ({side}, CROWN-IBP eps-schedule) on "
              f"{len(pool)} active cells (menu {len(menu_c)}, "
              f"margin={tr.cert_c4_margin}, c3_w={tr.cert_c3_w}, "
              f"eps {tr.cert_eps_start}->1 over {tr.cert_eps_warmup_frac} of epochs)")
        lbVf = precompute_lbVf(v, boxesL, pool, dyn, menu_c,
                               tr.cert_d_subsplit, cert.chunk)
        # C3 gate threshold per pool cell: gamma_deploy * ubV(C) + eps (the same
        # RHS the verifier's witness-C3 uses); ubV from the V cell-bounds above.
        gate_thresh = tr.gamma_deploy * ubV[pool] + cert.eps_margin
        qcert_rep = train_q_certified(
            q, boxesL, pool, lbVf, dyn, menu_c, tr.cert_d_subsplit,
            tr.cert_c4_w, tr.cert_c4_margin, anchor_w=0.1,
            epochs=tr.epochs_q, batch=512, lr=tr.lr * 0.5,
            v=v, model=oracle.model, seed=tr.seed + 5,
            eps_start=tr.cert_eps_start, eps_warmup_frac=tr.cert_eps_warmup_frac,
            gamma_deploy=tr.gamma_deploy,
            c3_w=tr.cert_c3_w, gate_thresh=gate_thresh)
        print(f"  cell-worst C4 viol (eps=1) {qcert_rep['init_cert_c4_viol']:.4f} "
              f"-> {qcert_rep['final_cert_c4_viol']:.4f}")
        if qcert_rep.get('menu_gate_feas_final') is not None:
            print(f"  certified menu-gate (C3) feasible "
                  f"{qcert_rep['menu_gate_feas_init']:.2f} -> "
                  f"{qcert_rep['menu_gate_feas_final']:.2f}")
        print(f"  C3 gate proxy (center) mean "
              f"{qcert_rep['gate_margin_mean_init']:+.4f} (>=0 "
              f"{qcert_rep['gate_frac_ge0_init']:.2f}) -> "
              f"{qcert_rep['gate_margin_mean_final']:+.4f} (>=0 "
              f"{qcert_rep['gate_frac_ge0_final']:.2f})")

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
            "vcert": vcert_rep,
            "qcert": qcert_rep,
            "wall_s": time.time() - t_all}
    print(f"[train] witness margin on Omega*: mean {diag['witness_margin_mean']:+.3f}, "
          f"p05 {diag['witness_margin_p05']:+.3f}, "
          f">=eps frac {diag['witness_margin_frac_ge_eps']:.3f}")
    save_json(cfg, "train_report.json", diag)
    update_manifest(cfg, "train", {**paths,
                                   "wall_s": round(diag["wall_s"], 1)})


if __name__ == "__main__":
    main()
