"""F1TENTH E2-L / P1 -- GROW-FROM-SEED certified invariant-set expansion.

Implements the go/no-go experiment of grow_from_seed_certified_expansion.md Sec.9
(P1).  The current RFC certificate is the brake-to-stop funnel S_brake covering
rho_brake ~ 0.80 of the analytic safe volume.  THIS experiment tests the central
LFP claim: the verified grow operator

    G_V(R) = R u ( K n V_0 n Pre_ver(R) ) ,   R_0 = S_brake ,

grows the certified set OUTWARD by adding heading-specific cells that can DRIVE
into the already-certified set in one sound step -- and every iterate stays a
sound robust invariant set (anytime soundness, no gfp erosion).  Unlike the dead
sub-level route (run_cert_p1.py), this certifies a strictly larger invariant set,
not a level set of V_theta.

Pipeline (iron-rule first):
  (0) MC soundness self-check of the successor envelope Enc (A3).
  (1) IDEAL iron-rule: grow a deliberately SHRUNK analytic seed {V>=c_seed} and
      confirm it expands back toward {V>=0} -- the primitive must non-vacuously
      grow on a KNOWN-invariant ideal (Theorem C empirically) before we trust any
      learned result.
  (2) LEARNED grow: seed = S_brake (frozen V_theta funnel), V_0 = {lb V_theta>=0}.
      Report rho(k), rho_inf vs rho_brake, layer histogram, Q-top-m proposal value
      of Q_theta, failure decomposition, timing.
  (3) Adversarial audit of the layered policy sigma (cbv must be 0).

    python experiments/f1tenth_e2/run_cert_p1_grow.py --quick
    python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44
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

try:                                  # never let a stray glyph crash a long run
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         brake_cbf_bounds, g_bounds_sq,
                                         heading_successor_interval,
                                         successor_box)
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.certify.grow import Grid4D, GrowEngine
from qcbf.certify.viab_reference import robust_capture_basin
from qcbf.dynamics.dubins import wrap_angle
from experiments.f1tenth_e2.distill import (GAMMA, distill, d_probe_grid,
                                            racing_steer)
from experiments.f1tenth_e2.run_cert import certified_set
from experiments.f1tenth_e2.run_cert_rfc import base_grid, brake_funnel_cert


# --------------------------------------------------------------------------- #
def make_menu(cfg):
    """Finite action menu U_menu (A5): 4 accel x 5 steer = 20 commands."""
    accs = [cfg.a_min, 0.5 * cfg.a_min, 0.0, cfg.a_max]
    steers = [-cfg.delta_max, -0.5 * cfg.delta_max, 0.0,
              0.5 * cfg.delta_max, cfg.delta_max]
    return [(a, d) for a in accs for d in steers]


def make_encode_fn(cfg):
    """Sound one-step envelope Enc(c,u): (px+,py+,v+) box over full D plus the
    raw (un-wrapped) psi+ interval.  This is the ONLY plant hook (A3)."""
    def encode(acmd, dcmd, lo, hi):
        nx_lo, nx_hi, ny_lo, ny_hi, nv_lo, nv_hi = successor_box(
            cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2],
            lo[:, 3], hi[:, 3], acmd, acmd, -cfg.d_a_max, cfg.d_a_max)
        np_lo, np_hi = heading_successor_interval(
            cfg, lo[:, 2], hi[:, 2], lo[:, 3], hi[:, 3], dcmd)
        return nx_lo, nx_hi, ny_lo, ny_hi, np_lo, np_hi, nv_lo, nv_hi
    return encode


# --------------------------------------------------------------------------- #
def mc_soundness(cfg, model, grid, menu, n_per_action=200_000, seed=0,
                 verbose=True):
    """A3 soundness: for random states in random cells and random d, the TRUE
    successor must lie in the Enc box (positions / v / un-wrapped heading)."""
    rng = np.random.default_rng(900 + seed)
    npx, npy, npsi, nv = grid.shape
    enc = make_encode_fn(cfg)
    worst = 0
    for (acmd, dcmd) in menu:
        n = n_per_action
        i = rng.integers(0, npx, n); j = rng.integers(0, npy, n)
        k = rng.integers(0, npsi, n); l = rng.integers(0, nv, n)
        lo = np.column_stack([grid.pxe[i], grid.pye[j], grid.pse[k], grid.ve[l]])
        hi = np.column_stack([grid.pxe[i + 1], grid.pye[j + 1],
                              grid.pse[k + 1], grid.ve[l + 1]])
        u = lo + rng.random((n, 4)) * (hi - lo)            # random state in cell
        da = rng.uniform(-cfg.d_a_max, cfg.d_a_max, n)
        dd = rng.uniform(-cfg.d_delta_max, cfg.d_delta_max, n)
        # true successor (un-wrapped heading, matched to Enc's raw interval)
        px1 = u[:, 0] + cfg.dt * u[:, 3] * np.cos(u[:, 2])
        py1 = u[:, 1] + cfg.dt * u[:, 3] * np.sin(u[:, 2])
        psi1 = u[:, 2] + cfg.dt * (u[:, 3] / cfg.wheelbase) * np.tan(dcmd + dd)
        v1 = np.clip(u[:, 3] + cfg.dt * (acmd + da), 0.0, cfg.v_max)
        nx_lo, nx_hi, ny_lo, ny_hi, np_lo, np_hi, nv_lo, nv_hi = enc(acmd, dcmd, lo, hi)
        bad = ((px1 < nx_lo - 1e-9) | (px1 > nx_hi + 1e-9)
               | (py1 < ny_lo - 1e-9) | (py1 > ny_hi + 1e-9)
               | (psi1 < np_lo - 1e-9) | (psi1 > np_hi + 1e-9)
               | (v1 < nv_lo - 1e-9) | (v1 > nv_hi + 1e-9))
        worst = max(worst, int(bad.sum()))
    if verbose:
        print(f"  [mc-sound] {len(menu)} actions x {n_per_action} samples: "
              f"max Enc violations = {worst}  -> {'SOUND' if worst == 0 else 'UNSOUND'}",
              flush=True)
    return worst


# --------------------------------------------------------------------------- #
def cell_lookup(grid, X):
    """Map continuous states (N,4) to 4-D cell indices (heading wrapped)."""
    npx, npy, npsi, nv = grid.shape
    wpx, wpy, wps, wv = grid.widths
    i = np.clip(((X[:, 0] - grid.pxe[0]) / wpx).astype(int), 0, npx - 1)
    j = np.clip(((X[:, 1] - grid.pye[0]) / wpy).astype(int), 0, npy - 1)
    k = np.mod(((wrap_angle(X[:, 2]) - grid.pse[0]) / wps).astype(int), npsi)
    l = np.clip((X[:, 3] / wv).astype(int), 0, nv - 1)
    return i, j, k, l


def _worst_d_for_R(cfg, model, grid, R, X, u):
    """Pick, per state, the disturbance corner whose successor is FARTHEST from R
    (most likely to escape), as a sound-ish greedy adversary on the cert claim."""
    corners = ([cfg.d_a_max, cfg.d_delta_max], [cfg.d_a_max, -cfg.d_delta_max],
               [-cfg.d_a_max, cfg.d_delta_max], [-cfg.d_a_max, -cfg.d_delta_max])
    best = None; best_in = None
    for dc in corners:
        dd = np.tile(dc, (len(X), 1))
        i, j, k, l = cell_lookup(grid, model.step(X, u, dd))
        inR = R[i, j, k, l]
        if best is None:
            best, best_in = dd.copy(), inR
        else:
            take = inR < best_in          # prefer d that makes successor leave R
            best_in = np.where(take, inR, best_in)
            best = np.where(take[:, None], dd, best)
    return best


def audit_grown_onestep(cfg, model, grid, R, layer, witness, menu,
                        n_roll=20000, seed=3, verbose=True):
    """DIRECT grow-soundness audit: from states in GROWN cells (layer>=1), apply
    the recorded witness action under the worst-of-corners adversary and confirm
    the successor stays in R AND g>=0.  By the certificate this MUST be 0 -- any
    escape is an Enc/index bug."""
    grown_cells = np.argwhere(layer >= 1)
    if len(grown_cells) == 0:
        return {"n": 0, "escapes": 0, "g_violations": 0}
    rng = np.random.default_rng(seed)
    menu_arr = np.array(menu)
    sel = grown_cells[rng.integers(0, len(grown_cells), n_roll)]
    i, j, k, l = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
    u01 = rng.random((n_roll, 4))
    X = np.column_stack([
        grid.pxe[i] + u01[:, 0] * (grid.pxe[i + 1] - grid.pxe[i]),
        grid.pye[j] + u01[:, 1] * (grid.pye[j + 1] - grid.pye[j]),
        grid.pse[k] + u01[:, 2] * (grid.pse[k + 1] - grid.pse[k]),
        grid.ve[l] + u01[:, 3] * (grid.ve[l + 1] - grid.ve[l])])
    u = menu_arr[witness[i, j, k, l]]
    d = _worst_d_for_R(cfg, model, grid, R, X, u)
    Xn = model.step(X, u, d)
    ii, jj, kk, ll = cell_lookup(grid, Xn)
    escapes = int((~R[ii, jj, kk, ll]).sum())
    gviol = int((model.g(Xn) < 0).sum())
    if verbose:
        print(f"  [audit:grown-1step] {n_roll} samples in grown cells, witness + "
              f"worst-d:  R-escapes={escapes}  g-viol={gviol}  "
              f"-> {'SOUND' if escapes == 0 and gviol == 0 else 'VIOLATION'}",
              flush=True)
    return {"n": n_roll, "escapes": escapes, "g_violations": gviol}


def audit_layered(cfg, model, grid, R, layer, witness, menu, n_omega3d,
                  n_roll=4000, horizon=300, seed=1, verbose=True):
    """End-to-end falsification of the layered policy sigma: at x in R, brake if
    seed (layer 0) else apply the witness u*_c; out-of-R states brake (safety
    net).  Roll under extremal / greedy d and confirm g>=0 throughout (cbv must
    be 0).  ``grown_escape`` counts grown-cell -> non-R transitions only (the
    seed brake-to-stop trajectory legitimately leaves the heading-free grid mask
    while staying safe, so total escapes are not a defect)."""
    rng = np.random.default_rng(seed)
    R_cells = np.argwhere(R)
    if len(R_cells) == 0:
        return {"certified_but_violated": 0, "note": "empty R"}
    menu_arr = np.array(menu)

    def sample(n):
        sel = R_cells[rng.integers(0, len(R_cells), n)]
        i, j, k, l = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        u = rng.random((n, 4))
        return np.column_stack([
            grid.pxe[i] + u[:, 0] * (grid.pxe[i + 1] - grid.pxe[i]),
            grid.pye[j] + u[:, 1] * (grid.pye[j + 1] - grid.pye[j]),
            grid.pse[k] + u[:, 2] * (grid.pse[k + 1] - grid.pse[k]),
            grid.ve[l] + u[:, 3] * (grid.ve[l + 1] - grid.ve[l])])

    def policy(X):
        i, j, k, l = cell_lookup(grid, X)
        inR = R[i, j, k, l]; lay = layer[i, j, k, l]; wit = witness[i, j, k, l]
        u = np.column_stack([np.full(len(X), cfg.a_min), racing_steer(cfg, X)])
        grown = inR & (lay >= 1) & (wit >= 0)
        if grown.any():
            u[grown] = menu_arr[wit[grown]]
        return u, grown

    out = {}
    for mode in ("extremal", "greedy"):
        X = sample(n_roll)
        ming = model.g(X).copy()
        grown_escape = 0
        for t in range(horizon):
            u, grown = policy(X)
            if mode == "extremal":
                d = np.column_stack([rng.choice([-cfg.d_a_max, cfg.d_a_max], len(X)),
                                     rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))])
            else:                                 # greedy: worst d for next g
                best = None; bestg = None
                for dc in ([cfg.d_a_max, cfg.d_delta_max], [cfg.d_a_max, -cfg.d_delta_max],
                           [-cfg.d_a_max, cfg.d_delta_max], [-cfg.d_a_max, -cfg.d_delta_max]):
                    dd = np.tile(dc, (len(X), 1)); gn = model.g(model.step(X, u, dd))
                    if bestg is None:
                        bestg, best = gn, dd
                    else:
                        take = gn < bestg; bestg = np.where(take, gn, bestg)
                        best = np.where(take[:, None], dd, best)
                d = best
            Xn = model.step(X, u, d)
            ii, jj, kk, ll = cell_lookup(grid, Xn)
            grown_escape += int((grown & ~R[ii, jj, kk, ll]).sum())
            X = Xn
            np.minimum(ming, model.g(X), out=ming)
        out[mode] = {"min_g": float(ming.min()), "g_violations": int((ming < 0).sum()),
                     "grown_escape_steps": int(grown_escape)}
        if verbose:
            r = out[mode]
            print(f"  [audit:{mode:8s}] min g {r['min_g']:+.4f}  g-viol "
                  f"{r['g_violations']}  grown->nonR {r['grown_escape_steps']}", flush=True)
    out["certified_but_violated"] = int(sum(out[m]["g_violations"]
                                            for m in ("extremal", "greedy")))
    return out


# --------------------------------------------------------------------------- #
def q_proposal_value(cfg, q_net, grid, res_diag, menu, m_list=(1, 3, 5),
                     verbose=True):
    """Q-top-m: among GROWN cells, fraction whose winning witness action ranks in
    the top-m of the menu ordered by min_d Q_theta(center,u,d) -- quantifies the
    value of using Q_theta to PROPOSE actions (it never affects soundness)."""
    ci, cj, ck, cl = res_diag["_cand"]
    added = res_diag["_added"]
    # grown candidates only (added & not seed) with a recorded witness
    grown = np.flatnonzero(added)
    if len(grown) == 0:
        return {"n": 0}
    gi, gj, gk, gl = ci[grown], cj[grown], ck[grown], cl[grown]
    centers = np.column_stack([
        0.5 * (grid.pxe[gi] + grid.pxe[gi + 1]),
        0.5 * (grid.pye[gj] + grid.pye[gj + 1]),
        0.5 * (grid.pse[gk] + grid.pse[gk + 1]),
        0.5 * (grid.ve[gl] + grid.ve[gl + 1])])
    n = len(centers)
    menu_arr = np.array(menu)
    dprobe = d_probe_grid(cfg)
    qmin = np.full((n, len(menu)), np.inf)
    for a, (acmd, dcmd) in enumerate(menu):
        ua = np.tile([acmd, dcmd], (n, 1))
        qa = np.full(n, np.inf)
        for dk in dprobe:
            z = np.concatenate([centers, ua, np.tile(dk, (n, 1))], axis=1)
            qa = np.minimum(qa, q_net.forward(z)[:, 0])
        qmin[:, a] = qa
    win = res_diag["_witness_for_grown"]               # winning action per grown cell
    q_win = qmin[np.arange(n), win]                     # min_d Q of the witness action
    rank = (qmin > q_win[:, None]).sum(axis=1)          # # actions strictly preferred
    out = {"n": int(n)}
    for m in m_list:
        out[f"top{m}"] = float(np.mean(rank < m))
    out["mean_rank"] = float(rank.mean())
    if verbose:
        s = "  ".join(f"top{m}={out[f'top{m}']:.3f}" for m in m_list)
        print(f"  [Q-propose] over {n} grown cells: {s}  (mean rank "
              f"{out['mean_rank']:.2f}/{len(menu)})", flush=True)
    return out


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def capture_calibration(cfg, model, grid, eng, omega3d, menu, n_d=3, verbose=True):
    """Calibrate the SOUND R_inf against a reference robust capture basin (Sec.
    3 of the user's review).  Splits the residual Omega*\\R_inf into method
    conservatism vs intrinsic capture-basin gap, and exposes whether the true
    capture basin exceeds the brake-anchored Omega* (Viab > Omega*)."""
    npx, npy, npsi, nv = grid.shape
    seed4d = np.broadcast_to(eng.seed3d[:, :, None, :], grid.shape).copy()
    omega4d = np.broadcast_to(omega3d[:, :, None, :], grid.shape)
    denom = max(int(omega3d.sum()) * npsi, 1)
    Capt_KV0, _ = robust_capture_basin(grid, model, cfg, seed4d, eng.VK, menu,
                                       n_d=n_d, tag="capt-KV0", verbose=verbose)
    Capt_K, _ = robust_capture_basin(grid, model, cfg, seed4d, eng.K, menu,
                                     n_d=n_d, tag="capt-K", verbose=verbose)
    nR = int(eng.R.sum()); nKV0 = int(Capt_KV0.sum()); nK = int(Capt_K.sum())
    out = {
        "rho_R": nR / denom, "rho_capt_KV0": nKV0 / denom, "rho_capt_K": nK / denom,
        "recovery_within_anchor": nR / max(nKV0, 1),
        "conservatism": (nKV0 - nR) / denom,
        "v0_gate_loss": (nK - nKV0) / denom,
        "capture_within_omega": int((Capt_K & omega4d).sum()) / denom,
        "capture_gap_of_omega": 1.0 - int((Capt_K & omega4d).sum()) / denom,
        "beyond_omega_frac": int((Capt_K & ~omega4d).sum()) / denom,
        "R_not_in_capt": int((eng.R & ~Capt_KV0).sum()),    # sanity: ~0 (sound⊆ref)
        "R_not_in_omega": int((eng.R & ~omega4d).sum()),    # B1: ~0
        "n_d_grid": n_d,
    }
    if verbose:
        print(f"  [calib] rho  R_inf={out['rho_R']:.3f}  capt(K&V0)="
              f"{out['rho_capt_KV0']:.3f}  capt(K)={out['rho_capt_K']:.3f}", flush=True)
        print(f"  [calib] recovery within anchor = {out['recovery_within_anchor']:.3f} "
              f"(1=no conservatism); conservatism {out['conservatism']:+.3f}, "
              f"V0-gate {out['v0_gate_loss']:+.3f}", flush=True)
        print(f"  [calib] capture gap of Omega* = {out['capture_gap_of_omega']:.3f}; "
              f"capture BEYOND Omega* (Viab>Omega*) = {out['beyond_omega_frac']:.3f}", flush=True)
        print(f"  [calib] checks: R-not-in-capt={out['R_not_in_capt']} (sound subset ref), "
              f"R-not-in-Omega={out['R_not_in_omega']} (B1)", flush=True)
    return out


def _phi_feasible(cfg, vn, qn, X, U, gamma, dprobe):
    """Learned predicate Phi_theta(x): min_d Q_theta(x,u,d) >= gamma V_theta(x)."""
    Vx = vn.forward(X[:, [0, 1, 3]])[:, 0]
    qmin = np.full(len(X), np.inf)
    for dk in dprobe:
        q = qn.forward(np.concatenate([X, U, np.tile(dk, (len(X), 1))], 1))[:, 0]
        qmin = np.minimum(qmin, q)
    return qmin >= gamma * Vx


def false_feasible_mass(cfg, vn, qn, grid, eng, aver, menu, gamma, verbose=True):
    """Theorem-S quantitative: over R_inf cells, count (c,u) with u in Phi_theta
    (learned says feasible) but u NOT in A_ver (Enc(c,u) leaves R_inf).  Such
    pairs prove the unrestricted Phi_theta filter is over-permissive."""
    dprobe = d_probe_grid(cfg)
    Rc = np.argwhere(eng.R)
    i, j, k, l = Rc[:, 0], Rc[:, 1], Rc[:, 2], Rc[:, 3]
    centers = np.column_stack([0.5 * (grid.pxe[i] + grid.pxe[i + 1]),
                               0.5 * (grid.pye[j] + grid.pye[j + 1]),
                               0.5 * (grid.pse[k] + grid.pse[k + 1]),
                               0.5 * (grid.ve[l] + grid.ve[l + 1])])
    bits = aver[i, j, k, l]
    n = len(centers); ff = 0; phi_tot = 0
    for a, (acmd, dcmd) in enumerate(menu):
        Ua = np.tile([acmd, dcmd], (n, 1))
        phi = _phi_feasible(cfg, vn, qn, centers, Ua, gamma, dprobe)
        in_aver = (bits & np.uint32(1 << a)) > 0
        ff += int((phi & ~in_aver).sum())
        phi_tot += int(phi.sum())
    out = {"n_R_cells": n, "false_feasible_pairs": ff, "phi_feasible_pairs": phi_tot,
           "false_feasible_frac_of_phi": ff / max(phi_tot, 1)}
    if verbose:
        print(f"  [Thm-S] false-feasible (c,u): {ff} of {phi_tot} Phi-feasible pairs "
              f"({100*out['false_feasible_frac_of_phi']:.1f}%) escape R_inf", flush=True)
    return out


def deployed_threeway(cfg, model, grid, eng, aver, vn, qn, menu, gamma,
                      n_roll=2500, horizon=220, seed=7, verbose=True):
    """Three-way rollout from R_inf (the main-claim comparison):
       shield  : race the max-accel action in A_ver(c) (certified safe), report
                 Q-agreement (chosen action also Phi_theta-feasible);
       naive   : race (a_max, racing steer) iff Phi_theta else brake (RFC naive);
       brake   : always brake.
    Under extremal+greedy d.  Shield cbv MUST be 0 (A_ver invariance); naive may
    be >0 (Theorem S); brake is safe but makes no progress."""
    rng = np.random.default_rng(seed)
    menu_arr = np.array(menu)
    acc_order = np.argsort(-menu_arr[:, 0])         # actions by descending accel
    dprobe = d_probe_grid(cfg)
    Rc = np.argwhere(eng.R)

    def sample(n):
        sel = Rc[rng.integers(0, len(Rc), n)]
        i, j, k, l = sel[:, 0], sel[:, 1], sel[:, 2], sel[:, 3]
        u = rng.random((n, 4))
        return np.column_stack([
            grid.pxe[i] + u[:, 0] * (grid.pxe[i + 1] - grid.pxe[i]),
            grid.pye[j] + u[:, 1] * (grid.pye[j + 1] - grid.pye[j]),
            grid.pse[k] + u[:, 2] * (grid.pse[k + 1] - grid.pse[k]),
            grid.ve[l] + u[:, 3] * (grid.ve[l + 1] - grid.ve[l])])

    d0 = np.zeros((1, 2))                            # nominal d for Q-ranking

    def shield_u(X):
        """Deployed certified Q-CBF shield: among the CERTIFIED-safe actions
        A_ver(c) (every one keeps R_inf invariant), let the learned Q_theta PICK
        -- argmax_a Q_theta(x, menu[a], 0) over a in A_ver.  Q guides style/speed,
        the verifier guarantees safety; if A_ver empty (pure seed) brake."""
        i, j, k, l = cell_lookup(grid, X)
        bits = aver[i, j, k, l]
        n = len(X)
        qscore = np.full((n, len(menu)), -np.inf)
        for a, (acmd, dcmd) in enumerate(menu):
            ina = (bits & np.uint32(1 << a)) > 0
            if not ina.any():
                continue
            Ua = np.tile([acmd, dcmd], (n, 1))
            q = qn.forward(np.concatenate([X, Ua, np.tile(d0, (n, 1))], 1))[:, 0]
            qscore[:, a] = np.where(ina, q, -np.inf)
        chosen = np.argmax(qscore, axis=1)
        has = np.isfinite(qscore[np.arange(n), chosen])   # A_ver non-empty
        u = np.column_stack([np.full(n, cfg.a_min), racing_steer(cfg, X)])
        u[has] = menu_arr[chosen[has]]
        agree = np.zeros(n, bool)                    # Phi_theta endorses the pick?
        if has.any():
            agree[has] = _phi_feasible(cfg, vn, qn, X[has], u[has], gamma, dprobe)
        return u, has, agree

    def naive_u(X):
        v = X[:, 3]
        u_race = np.column_stack([np.where(v < cfg.v_max, cfg.a_max, 0.0),
                                  racing_steer(cfg, X)])
        feas = _phi_feasible(cfg, vn, qn, X, u_race, gamma, dprobe)
        u_brake = np.column_stack([np.full(len(X), cfg.a_min), racing_steer(cfg, X)])
        return np.where(feas[:, None], u_race, u_brake), feas

    def brake_u(X):
        return np.column_stack([np.full(len(X), cfg.a_min), racing_steer(cfg, X)]), None

    out = {}
    for name, fn in (("shield", shield_u), ("naive", naive_u), ("brake", brake_u)):
        worst = {}
        for mode in ("extremal", "greedy"):
            X = sample(n_roll)
            ming = model.g(X).copy()
            race_steps = 0.0; speed_sum = 0.0; agree_steps = 0.0
            for t in range(horizon):
                r = fn(X)
                u = r[0]
                if name == "shield":
                    race_steps += float(r[1].mean()); agree_steps += float(r[2].mean())
                elif name == "naive":
                    race_steps += float(r[1].mean())
                if mode == "extremal":
                    d = np.column_stack([rng.choice([-cfg.d_a_max, cfg.d_a_max], len(X)),
                                         rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))])
                else:
                    best = None; bestg = None
                    for dc in ([cfg.d_a_max, cfg.d_delta_max], [cfg.d_a_max, -cfg.d_delta_max],
                               [-cfg.d_a_max, cfg.d_delta_max], [-cfg.d_a_max, -cfg.d_delta_max]):
                        dd = np.tile(dc, (len(X), 1)); gn = model.g(model.step(X, u, dd))
                        if bestg is None:
                            bestg, best = gn, dd
                        else:
                            take = gn < bestg; bestg = np.where(take, gn, bestg)
                            best = np.where(take[:, None], dd, best)
                    d = best
                X = model.step(X, u, d)
                np.minimum(ming, model.g(X), out=ming)
                speed_sum += float(X[:, 3].mean())
            worst[mode] = {"g_violations": int((ming < 0).sum()),
                           "min_g": float(ming.min()),
                           "race_frac": race_steps / horizon,
                           "mean_speed": speed_sum / horizon,
                           "q_agree_frac": agree_steps / horizon if name == "shield" else None}
        out[name] = {"cbv": int(worst["extremal"]["g_violations"] + worst["greedy"]["g_violations"]),
                     "min_g": min(worst["extremal"]["min_g"], worst["greedy"]["min_g"]),
                     "race_frac": float(np.mean([worst[m]["race_frac"] for m in worst])),
                     "mean_speed": float(np.mean([worst[m]["mean_speed"] for m in worst])),
                     "q_agree_frac": (float(np.mean([worst[m]["q_agree_frac"] for m in worst]))
                                      if name == "shield" else None)}
        if verbose:
            r = out[name]
            extra = (f" q-agree {100*r['q_agree_frac']:.0f}%" if r["q_agree_frac"] is not None else "")
            print(f"  [deploy:{name:6s}] cbv={r['cbv']:5d}  min g {r['min_g']:+.3f}  "
                  f"race {100*r['race_frac']:3.0f}%  mean v {r['mean_speed']:.2f}{extra}",
                  flush=True)
    return out


# --------------------------------------------------------------------------- #
def grow_run(grid, seed3d, K3d, V03d, menu, encode_fn, n_omega3d, tag, verbose=True):
    eng = GrowEngine(grid, seed3d, K3d, V03d, menu, encode_fn)
    t0 = time.time()
    res = eng.run(n_omega3d, verbose=verbose)
    res["wall_s"] = round(time.time() - t0, 1)
    # expose witness-for-grown (for the Q-proposal diagnostic)
    ci, cj, ck, cl = res["_cand"]
    added = res["_added"]
    res["_witness_for_grown"] = eng.witness[ci[added], cj[added], ck[added], cl[added]]
    res["_engine"] = eng
    if verbose:
        print(f"  [{tag}] rho_seed={res['rho_seed']:.4f} -> rho_inf="
              f"{res['rho_inf']:.4f}  (+{res['rho_inf'] - res['rho_seed']:+.4f}), "
              f"{res['n_grown_4d']} cells grown in {res['n_waves']} waves "
              f"({res['wall_s']:.0f}s)", flush=True)
    return res


def _strip(res):
    """Drop the heavy/non-serialisable internals before writing JSON."""
    return {k: v for k, v in res.items()
            if not k.startswith("_") and k != "rho_hist"} | {
        "rho_hist": [round(float(x), 5) for x in res["rho_hist"]]}


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--res", type=int, default=0, help="grid edge (e.g. 44)")
    ap.add_argument("--npsi", type=int, default=0,
                    help="override heading slices (diagnostic: heading conservatism)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true",
                    help="also run capture-basin calibration + A_ver saturation "
                         "+ deployed three-way rollout + Theorem-S mass")
    ap.add_argument("--dump", action="store_true",
                    help="save the certified-set grid (R, layer, masks, edges) to "
                         "an npz for the slice/onion figures")
    ap.add_argument("--c-seed-ideal", type=float, default=0.30,
                    help="shrunk analytic seed level for the iron-rule grow")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.quick:
        npx = npy = 24; nv = 18; npsi = 12
        train_kw = dict(n_samples=30_000, reg_epochs=12, cbf_epochs=12)
        n_mc = 50_000
    else:
        n = args.res or 44
        npx = npy = n; nv = int(round(n * 0.75)); npsi = args.npsi or 16
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=30)
        n_mc = 200_000

    grid = Grid4D.make(cfg.p_lo, cfg.p_hi, cfg.v_max, npx, npy, npsi, nv)
    menu = make_menu(cfg)
    encode_fn = make_encode_fn(cfg)
    print("=" * 88)
    print("E2-L / P1 -- GROW-FROM-SEED certified invariant-set expansion (lfp)")
    print(f"  grid {npx}x{npy}x{npsi}x{nv} (4-D), menu {len(menu)} actions, "
          f"seed = S_brake, V_0 = {{lb V_theta >= 0}}")
    print("=" * 88, flush=True)

    # ---- (0) Enc soundness self-check (A3) --------------------------------- #
    print("\n[0] Enc envelope soundness (A3) Monte-Carlo self-check", flush=True)
    mc = mc_soundness(cfg, model, grid, menu, n_per_action=n_mc, seed=args.seed)

    # ---- base grid (analytic Omega*, learned V bounds, g) ------------------ #
    n_omega3d = certified_set(cfg, model, npx, npy, nv)["accepted"]
    # analytic V box bounds for the ideal seed (reuse exact closed-form)
    px = np.linspace(cfg.p_lo, cfg.p_hi, npx + 1)
    py = np.linspace(cfg.p_lo, cfg.p_hi, npy + 1)
    vv = np.linspace(0.0, cfg.v_max, nv + 1)
    PXl, PYl, Vl = np.meshgrid(px[:-1], py[:-1], vv[:-1], indexing="ij")
    PXh, PYh, Vh = np.meshgrid(px[1:], py[1:], vv[1:], indexing="ij")
    aVlo, _ = brake_cbf_bounds(cfg, PXl.ravel(), PXh.ravel(), PYl.ravel(),
                               PYh.ravel(), Vl.ravel(), Vh.ravel())
    gmin = g_bounds_sq(cfg, PXl.ravel(), PXh.ravel(), PYl.ravel(), PYh.ravel())
    aVlo = aVlo.reshape(npx, npy, nv); gmin = gmin.reshape(npx, npy, nv)

    # ---- (1) IDEAL iron-rule: grow a shrunk analytic seed ------------------ #
    print(f"\n[1] IDEAL iron-rule: grow shrunk analytic seed "
          f"{{V_analytic>={args.c_seed_ideal}}} toward {{V>=0}}", flush=True)
    ideal_seed3d = aVlo >= args.c_seed_ideal
    K3d = gmin >= 0.0
    ideal_V03d = aVlo >= 0.0
    n_ideal_omega = int(ideal_V03d.sum())
    print(f"  [ideal] seed {int(ideal_seed3d.sum())} / V0 {int(ideal_V03d.sum())} "
          f"3-D cells (Omega* {n_omega3d})", flush=True)
    ideal = grow_run(grid, ideal_seed3d, K3d, ideal_V03d, menu, encode_fn,
                     n_ideal_omega, "ideal")
    ideal_grows = ideal["n_grown_4d"] > 0
    print(f"  [ideal] non-vacuity: primitive {'GROWS' if ideal_grows else 'does NOT grow'} "
          f"the ideal ({ideal['n_grown_4d']} cells) -> "
          f"{'primitive OK' if ideal_grows else 'PRIMITIVE TOO WEAK'}", flush=True)
    print(f"  [ideal] failure decomposition: {ideal['blocked_breakdown']} "
          f"(learned v0_blocked excess over this = the lb V_theta CROWN-gate)", flush=True)

    # ---- distill the LEARNED trio (training = non-vacuity aid only) -------- #
    print(f"\n[2] LEARNED grow: distill frozen (V_theta,Q_theta,pi_b) seed={args.seed}",
          flush=True)
    v, q, pi, ddiag = distill(cfg, args.seed, margin=0.10, **train_kw)
    vn, qn = SeqNet.from_mlp(v), SeqNet.from_mlp(q)
    print(f"  [distill] MSE(V)={ddiag['mse_V_target']:.4f} "
          f"MSE(Q)={ddiag['mse_Q_target']:.4f}", flush=True)

    # learned V_theta box bounds + S_brake seed (reuse the RFC funnel)
    bg = base_grid(cfg, vn, npx, npy, nv, verbose=False)
    seed_mask, fdiag = brake_funnel_cert(cfg, vn, bg, npsi, verbose=False)
    seed3d = seed_mask.reshape(npx, npy, nv)
    lbV3d = bg["lbV"].reshape(npx, npy, nv)
    V03d = lbV3d >= 0.0
    VK3d = V03d & K3d
    rho_brake = float(seed3d.sum() / max(n_omega3d, 1))
    print(f"  [seed] S_brake {int(seed3d.sum())} cells, V_0 {int(V03d.sum())}, "
          f"K n V_0 {int(VK3d.sum())}  (Omega* {n_omega3d}) -> rho_brake={rho_brake:.4f}",
          flush=True)

    learned = grow_run(grid, seed3d, K3d, V03d, menu, encode_fn, n_omega3d, "learned")

    # ---- Q-proposal value + audit ----------------------------------------- #
    print("\n[3] Q_theta proposal value (offline accelerator only)", flush=True)
    qprop = q_proposal_value(cfg, qn, grid, learned, menu)

    print("\n[4] Adversarial audit of the layered policy sigma (cbv must be 0)",
          flush=True)
    eng = learned["_engine"]
    aud_grown = audit_grown_onestep(cfg, model, grid, eng.R, eng.layer,
                                    eng.witness, menu, seed=args.seed + 2)
    aud = audit_layered(cfg, model, grid, eng.R, eng.layer, eng.witness, menu,
                        n_omega3d, seed=args.seed + 1)
    aud["grown_onestep"] = aud_grown

    # ---- (5) FULL: calibration + A_ver saturation + deployed rollout -------- #
    calib = aver_diag = thmS = deploy = None
    if args.full:
        omega3d_mask = aVlo >= 0.0
        print("\n[5] Capture-basin calibration (Omega* vs reference Viab)", flush=True)
        calib = capture_calibration(cfg, model, grid, eng, omega3d_mask, menu)
        print("\n[6] A_ver saturation table + deployed Q-CBF filter (main claim)",
              flush=True)
        aver, acount = eng.compute_aver()
        nz = acount[eng.R]
        aver_diag = {"mean_A_ver": float(nz.mean()), "frac_A_ver_empty": float((nz == 0).mean()),
                     "frac_A_ver_ge1": float((nz >= 1).mean())}
        print(f"  [A_ver] mean |A_ver|={aver_diag['mean_A_ver']:.2f} actions/cell, "
              f"{100*aver_diag['frac_A_ver_ge1']:.1f}% of R_inf has a certified racing "
              f"action (rest brake-only)", flush=True)
        thmS = false_feasible_mass(cfg, vn, qn, grid, eng, aver, menu, GAMMA)
        deploy = deployed_threeway(cfg, model, grid, eng, aver, vn, qn, menu, GAMMA)

    # ---- verdict ----------------------------------------------------------- #
    drho = learned["rho_inf"] - rho_brake
    if drho >= 0.10:
        signal = "STRONG-GO (Delta-rho >= +0.10)"
    elif drho >= 0.05:
        signal = "GO (Delta-rho >= +0.05)"
    elif drho > 1e-3:
        signal = "WEAK (0 < Delta-rho < +0.05) -- diagnose seed/menu before scaling"
    else:
        signal = "NO-GO (Delta-rho ~ 0) -- run diagnostics + multi-funnel seed"
    gate = bool(mc == 0 and aud["certified_but_violated"] == 0
                and aud_grown["escapes"] == 0 and aud_grown["g_violations"] == 0
                and learned["n_R_4d"] >= learned["n_seed_4d"])

    blob = {
        "config": {"res": [npx, npy, npsi, nv], "menu_size": len(menu),
                   "seed": args.seed, "gamma": GAMMA, "c_seed_ideal": args.c_seed_ideal,
                   "train_kw": train_kw},
        "mc_soundness_violations": mc,
        "n_omega3d": n_omega3d,
        "rho_brake": rho_brake,
        "ideal": _strip(ideal), "ideal_grows": ideal_grows,
        "learned": _strip(learned),
        "delta_rho": drho,
        "q_proposal": qprop,
        "audit": aud,
        "calibration": calib, "a_ver": aver_diag, "theorem_s": thmS, "deployed": deploy,
        "distill": {k: ddiag.get(k) for k in ("mse_V_target", "mse_Q_target")},
        "signal": signal, "gate_sound": gate,
        "wall_s": round(time.time() - t0, 1),
    }
    rep_name = "p1_grow_full.json" if args.full else "p1_grow_report.json"
    (out / rep_name).write_text(json.dumps(blob, indent=2))

    if args.dump:                       # grid for the slice / onion figures
        omega3d_mask = aVlo >= 0.0
        np.savez_compressed(
            out / f"p1_grow_grid_res{npx}.npz",
            R=eng.R, layer=eng.layer, seed3d=seed3d, V03d=V03d, K3d=K3d,
            omega3d=omega3d_mask, pxe=grid.pxe, pye=grid.pye, pse=grid.pse,
            ve=grid.ve, res=np.array([npx, npy, npsi, nv]),
            rho_brake=rho_brake, rho_inf=learned["rho_inf"])
        print(f"  [dump] wrote {out / f'p1_grow_grid_res{npx}.npz'}", flush=True)

    print("\n" + "=" * 88)
    print("P1 GROW-FROM-SEED  preliminary result")
    print("=" * 88)
    print(f"  Enc soundness (MC)        : {mc} violations "
          f"({'SOUND' if mc == 0 else 'UNSOUND'})")
    print(f"  ideal iron-rule           : grows {ideal['n_grown_4d']} cells "
          f"(rho {ideal['rho_seed']:.3f}->{ideal['rho_inf']:.3f}) "
          f"-> {'primitive validated' if ideal_grows else 'PRIMITIVE WEAK'}")
    print(f"  rho_brake (seed)          : {rho_brake:.4f}")
    print(f"  rho_inf (grown)           : {learned['rho_inf']:.4f}")
    print(f"  Delta-rho                 : {drho:+.4f}   -> {signal}")
    print(f"  cells grown / waves       : {learned['n_grown_4d']} / {learned['n_waves']}")
    print(f"  failure decomposition     : {learned['blocked_breakdown']}")
    print(f"  grown 1-step soundness    : escapes={aud_grown['escapes']} "
          f"g-viol={aud_grown['g_violations']} (of {aud_grown['n']})")
    print(f"  audit certified-but-viol  : {aud['certified_but_violated']} "
          f"(gate {'PASS' if gate else 'FAIL'})")
    if args.full:
        print(f"  capture recovery / gap    : recovery {calib['recovery_within_anchor']:.3f}, "
              f"V0-gate {calib['v0_gate_loss']:+.3f}, capture-gap "
              f"{calib['capture_gap_of_omega']:.3f}, beyond-Omega "
              f"{calib['beyond_omega_frac']:+.3f}")
        print(f"  deployed cbv (shield/naive/brake): "
              f"{deploy['shield']['cbv']} / {deploy['naive']['cbv']} / {deploy['brake']['cbv']}"
              f"   shield mean-v {deploy['shield']['mean_speed']:.2f} "
              f"(brake {deploy['brake']['mean_speed']:.2f})")
        print(f"  Theorem-S false-feasible  : {thmS['false_feasible_pairs']} pairs "
              f"({100*thmS['false_feasible_frac_of_phi']:.1f}% of Phi-feasible escape R_inf)")
    print(f"Wrote {out / rep_name}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
