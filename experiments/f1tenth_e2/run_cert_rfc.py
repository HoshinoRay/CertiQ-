"""F1TENTH E2-L / RFC -- PHASE 1 of the recursive-feasibility certificate.

Sound, claim-aligned Gate-D on the DEPLOYED LEARNED trio (V_t, Q_t, pi_b) -- a
plain full-expressivity ReLU MLP, NO analytic D(v) baked in, NO architectural
Lipschitz.  The naive sub-level target {V_t>=0} is genuinely not one-step
invariant (real C2 holes, exact probe), so we certify the hole-free TERMINAL CORE

    Omega_term = {V_t >= c_term}

robustly forward-invariant under the BRAKING fallback (the fallback-pinned
filter), via the RELATIONAL decrease bound (qcbf/certify/direct.py
crown_relational_decrease_lb) that cancels the per-cell oscillation tax a
decoupled successor bound pays.  Conditions, all DIRECT on the frozen nets +
true f, no V_t~V_target:

  T1 (safety)   : g(cell) >= 0                                  (exact g_bounds_sq)
  T2 (=C3)      : min_d Q_t(x, pi_b, d) >= gamma V_t + eps       (CROWN, witness live)
  T3 (landing)  : L_rel(cell) = lbV + relational_G(rho=1) >= c_term  (brake-invariant)

Phase-1 scope (per the plan's staging): ONE-PASS terminal-core only -- no GFP, no
funnel, no all-feasible, no multi-step.  Goal: Omega_term != empty, rho_term > 0.
Iron rule: the analytic ideal must pass first (reuses run_cert.py: 86,580 @ 80^3,
exact braking decrease +0.0000) -- that is the rho denominator Vol(Omega*).

    python experiments/f1tenth_e2/run_cert_rfc.py --quick
    python experiments/f1tenth_e2/run_cert_rfc.py
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

from qcbf.dynamics.bicycle_accel import (BicycleAccelConfig, BicycleAccelModel,
                                         g_bounds_sq, successor_box)
from qcbf.verify.bounds import SeqNet, crown_bounds_chunked
from qcbf.certify.direct import (crown_brake_successor_lb,
                                 crown_relational_decrease_lb)
from qcbf.util.progress import Progress
from experiments.f1tenth_e2.distill import (GAMMA, distill, clip_action,
                                            d_probe_grid, racing_steer)
from experiments.f1tenth_e2.run_cert import certified_set, verify_braking_decrease


# --------------------------------------------------------------------------- #
def base_grid(cfg, v_net, npx, npy, nv, chunk=4096, verbose=True):
    """(px,py,v) cell lattice: CROWN V_t bounds + exact g over each position box."""
    px = np.linspace(cfg.p_lo, cfg.p_hi, npx + 1)
    py = np.linspace(cfg.p_lo, cfg.p_hi, npy + 1)
    vv = np.linspace(0.0, cfg.v_max, nv + 1)
    PXl, PYl, Vl = np.meshgrid(px[:-1], py[:-1], vv[:-1], indexing="ij")
    PXh, PYh, Vh = np.meshgrid(px[1:], py[1:], vv[1:], indexing="ij")
    lo = np.column_stack([PXl.ravel(), PYl.ravel(), Vl.ravel()])
    hi = np.column_stack([PXh.ravel(), PYh.ravel(), Vh.ravel()])
    lbV, ubV = crown_bounds_chunked(v_net, lo, hi, True, chunk,
                                    progress="V-bounds" if verbose else None)
    lbV, ubV = lbV[:, 0], ubV[:, 0]
    gmin = g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])
    cell_vol = ((cfg.p_hi - cfg.p_lo) / npx) * ((cfg.p_hi - cfg.p_lo) / npy) * (cfg.v_max / nv)
    return {"lo": lo, "hi": hi, "lbV": lbV, "ubV": ubV, "gmin": gmin,
            "cell_vol": cell_vol, "res": (npx, npy, nv)}


# --------------------------------------------------------------------------- #
def c_term_probe(cfg, model, v_net, pi_net, levels, delta=0.05, eta_term=0.0,
                 n_samples=300_000, seed=0, verbose=True):
    """EXACT pointwise witness-brake hard-non-decrease (gamma=1) probe.

    For dense states x (all headings), the brake fallback u=(a_min, witness steer)
    and worst d (d_a=+d_a_max -> largest v+), measure
        h1(x) = min_d V_t(f(x,u,d)) - V_t(x).
    For each candidate level c, report the shell {V_t in [c, c+delta]} min margin.
    c_term = smallest c such that the shell AND every shell above it have
    min h1 >= +eta_term (hole-free core).  This sets the SCALAR c_term only; it
    never enters a banked cell's soundness (firewall).
    """
    rng = np.random.default_rng(7000 + seed)
    keep = []
    got = 0
    cmax = float(max(levels)) + delta
    for _ in range(400):
        p = rng.uniform(cfg.p_lo, cfg.p_hi, (200_000, 2))
        ps = rng.uniform(-np.pi, np.pi, 200_000)
        v = rng.uniform(0.0, cfg.v_max, 200_000)
        X = np.column_stack([p[:, 0], p[:, 1], ps, v])
        Vx = v_net.forward(X[:, [0, 1, 3]])[:, 0]
        m = (Vx >= -0.02) & (Vx <= cmax)            # in-set band up to the levels
        if m.any():
            keep.append(X[m]); got += int(m.sum())
        if got >= n_samples:
            break
    X = np.concatenate(keep)[:n_samples] if keep else np.zeros((0, 4))
    Vx = v_net.forward(X[:, [0, 1, 3]])[:, 0]
    # brake fallback: a = a_min, witness steering (irrelevant to heading-free V);
    # worst d for V_t(f) is d_a = +d_a_max (largest v+ -> deepest into the band).
    u = clip_action(cfg, pi_net.forward(X))
    u = np.column_stack([np.full(len(X), cfg.a_min), u[:, 1]])
    d = np.column_stack([np.full(len(X), cfg.d_a_max), np.zeros(len(X))])
    Vf = v_net.forward(model.step(X, u, d)[:, [0, 1, 3]])[:, 0]
    h1 = Vf - Vx                                     # exact hard-non-decrease margin

    table = []
    for c in levels:
        shell = (Vx >= c) & (Vx < c + delta)
        n = int(shell.sum())
        mn = float(h1[shell].min()) if n else float("nan")
        table.append({"c": float(c), "n": n, "min_h1": mn,
                      "frac_neg": float(np.mean(h1[shell] < 0.0)) if n else 0.0})
    # smallest c whose shell-and-above are all >= eta_term
    c_term = None
    for i, row in enumerate(table):
        above = [r for r in table[i:] if r["n"] > 0]
        if above and all(r["min_h1"] >= eta_term for r in above):
            c_term = row["c"]; break
    if verbose:
        print(f"  [probe] exact witness-brake gamma=1 non-decrease on shells "
              f"(delta={delta}, {len(X)} samples):", flush=True)
        for r in table:
            print(f"    c={r['c']:.3f}: n={r['n']:7d}  min h1={r['min_h1']:+.4f}  "
                  f"frac<0={r['frac_neg']:.3f}", flush=True)
        print(f"  [probe] -> c_term = {c_term} (smallest hole-free, "
              f"eta_term={eta_term})", flush=True)
    return c_term, table


# --------------------------------------------------------------------------- #
def witness_feasibility(cfg, v_net, q_net, pi_net, lo, hi, ubV, idx, npsi,
                        gamma, eps, chunk=4096):
    """T2 = C3: min_d Q_t(x, clip(pi_b(x)), d) >= gamma V_t(x) + eps over the core
    cells (CROWN, decoupled pi->Q but sound).  Returns bool mask over `idx`."""
    if len(idx) == 0:
        return np.zeros(0, bool)
    slo, shi, sub_ub = lo[idx], hi[idx], ubV[idx]
    Ms = len(idx)
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    Dlo = np.array([-cfg.d_a_max, -cfg.d_delta_max])
    Dhi = np.array([cfg.d_a_max, cfg.d_delta_max])
    h3 = np.full((npsi, Ms), np.inf)
    for h in range(npsi):
        xl = np.column_stack([slo[:, 0], slo[:, 1], np.full(Ms, ps[h]), slo[:, 2]])
        xh = np.column_stack([shi[:, 0], shi[:, 1], np.full(Ms, ps[h + 1]), shi[:, 2]])
        yl, yh = crown_bounds_chunked(pi_net, xl, xh, True, chunk)
        qlo, _ = crown_bounds_chunked(
            q_net,
            np.column_stack([xl, np.clip(yl[:, 0], cfg.a_min, cfg.a_max),
                             np.clip(yl[:, 1], -cfg.delta_max, cfg.delta_max),
                             np.tile(Dlo, (Ms, 1))]),
            np.column_stack([xh, np.clip(yh[:, 0], cfg.a_min, cfg.a_max),
                             np.clip(yh[:, 1], -cfg.delta_max, cfg.delta_max),
                             np.tile(Dhi, (Ms, 1))]), True, chunk)
        h3[h] = np.minimum(h3[h], qlo[:, 0] - gamma * sub_ub - eps)
    return (h3 >= 0.0).all(axis=0)


# --------------------------------------------------------------------------- #
def terminal_core_cert(cfg, v_net, q_net, pi_net, bg, c_term, npsi,
                       gamma=GAMMA, eps=5e-3, chunk=4096, verbose=True):
    """One-pass terminal-core certificate at c_term (fallback-pinned, brake).

    Core candidates = {lbV >= c_term}.  T1 g>=0, T2 C3 witness feasibility, T3
    relational brake-landing  L_rel = lbV + min_psi G_rel(rho=1) >= c_term.
    Also computes the DECOUPLED successor floor for the artifact-removal report.
    Returns (mask over all cells, diag).
    """
    lo, hi, lbV, ubV, gmin = bg["lo"], bg["hi"], bg["lbV"], bg["ubV"], bg["gmin"]
    core = np.flatnonzero(lbV >= c_term)            # fully-inside-core cells
    if verbose:
        print(f"  [core] {len(core)} cells with lbV >= c_term={c_term:.3f}", flush=True)
    if len(core) == 0:
        return np.zeros(len(lbV), bool), {"n_core": 0, "rho_proxy": 0.0}

    clo, chi = lo[core], hi[core]
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    # worst-heading relational decrease and decoupled successor floor
    G_rel = np.full(len(core), np.inf)
    L_dec = np.full(len(core), np.inf)
    pb = Progress(npsi, "cert-psi") if verbose else None
    for h in range(npsi):
        G_rel = np.minimum(G_rel, crown_relational_decrease_lb(
            cfg, v_net, clo[:, 0], chi[:, 0], clo[:, 1], chi[:, 1],
            ps[h], ps[h + 1], clo[:, 2], chi[:, 2], rho_level=1.0, chunk=chunk))
        L_dec = np.minimum(L_dec, crown_brake_successor_lb(
            cfg, v_net, clo[:, 0], chi[:, 0], clo[:, 1], chi[:, 1],
            ps[h], ps[h + 1], clo[:, 2], chi[:, 2]))
        if pb is not None:
            pb.update(h + 1)
    if pb is not None:
        pb.done()
    L_rel = lbV[core] + G_rel                        # relational successor lb <= V(f)

    t1 = gmin[core] >= 0.0
    t3_rel = L_rel >= c_term
    t3_dec = L_dec >= c_term                          # decoupled baseline (no relational)
    # T2 only needs to be checked where T1 & T3 already hold (cheaper)
    cand = np.flatnonzero(t1 & t3_rel)
    t2_sub = witness_feasibility(cfg, v_net, q_net, pi_net, lo, hi, ubV,
                                 core[cand], npsi, gamma, eps, chunk)
    t2 = np.zeros(len(core), bool)
    t2[cand] = t2_sub

    accept = t1 & t2 & t3_rel
    mask = np.zeros(len(lbV), bool)
    mask[core[accept]] = True

    diag = {
        "n_core": int(len(core)),
        "t1_pass": int(t1.sum()), "t2_pass": int(t2.sum()),
        "t3_rel_pass": int(t3_rel.sum()), "t3_dec_pass": int(t3_dec.sum()),
        "accepted": int(accept.sum()),
        # artifact removal: relational vs decoupled successor floor on the core
        "L_rel_min": float(L_rel.min()), "L_dec_min": float(L_dec.min()),
        "Lrel_minus_Ldec_med": float(np.median(L_rel - L_dec)),
        "G_rel_min": float(G_rel.min()),
    }
    if verbose:
        print(f"  [T1 g>=0]      {diag['t1_pass']}/{len(core)}", flush=True)
        print(f"  [T3 landing]   relational {diag['t3_rel_pass']}/{len(core)}  "
              f"(decoupled would give {diag['t3_dec_pass']})", flush=True)
        print(f"  [T2 C3 live]   {diag['t2_pass']}/{int(len(cand))} candidates", flush=True)
        print(f"  [artifact]     L_rel-L_dec median {diag['Lrel_minus_Ldec_med']:+.4f}  "
              f"(min floor rel {diag['L_rel_min']:+.4f} vs dec {diag['L_dec_min']:+.4f})",
              flush=True)
        print(f"  [Omega_term]   accepted {diag['accepted']} cells", flush=True)
    return mask, diag


# --------------------------------------------------------------------------- #
def _brake_traj_safe(cfg, v_net, bx0, bx1, by0, by1, pl, ph, bv0, bv1,
                     n_stop=22, chunk=4096):
    """SOUND check: braking (a=a_min, full D, heading spread by max turn) from each
    initial (px,py,psi,v) box keeps lbV>=0 AND g>=0 at EVERY step until v=0 (a
    fixed point).  Returns a bool mask (True = whole braking trajectory safe).
    This is the Mode-B brake-to-stop trajectory bound: a finite composition, NOT
    a set-closure GFP -> no carving erosion; only the heading box widens (bounded,
    since braking kills v fast)."""
    bx0, bx1 = bx0.copy(), bx1.copy(); by0, by1 = by0.copy(), by1.copy()
    bv0, bv1 = bv0.copy(), bv1.copy(); pl, ph = pl.copy(), ph.copy()
    L, dt, tmax = cfg.wheelbase, cfg.dt, np.tan(cfg.delta_max + cfg.d_delta_max)
    ok = np.ones(len(bx0), bool)
    active = bv1 > 1e-9
    for _ in range(n_stop):
        idx = np.flatnonzero(active & ok)
        if len(idx):
            lbVh, _ = crown_bounds_chunked(
                v_net, np.column_stack([bx0[idx], by0[idx], bv0[idx]]),
                np.column_stack([bx1[idx], by1[idx], bv1[idx]]), True, chunk)
            gh = g_bounds_sq(cfg, bx0[idx], bx1[idx], by0[idx], by1[idx])
            ok[idx[(lbVh[:, 0] < 0.0) | (gh < 0.0)]] = False
        npx0, npx1, npy0, npy1, nv0, nv1 = successor_box(
            cfg, bx0, bx1, by0, by1, pl, ph, bv0, bv1,
            cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)
        T = dt * (bv1 / L) * tmax
        pl = pl - T; ph = ph + T
        bx0, bx1, by0, by1, bv0, bv1 = npx0, npx1, npy0, npy1, nv0, nv1
        active = bv1 > 1e-9
        if not active.any():
            break
    st = np.flatnonzero(ok)                          # final stopped (v=0) check
    if len(st):
        lbVf, _ = crown_bounds_chunked(
            v_net, np.column_stack([bx0[st], by0[st], np.zeros(len(st))]),
            np.column_stack([bx1[st], by1[st], np.zeros(len(st))]), True, chunk)
        gf = g_bounds_sq(cfg, bx0[st], bx1[st], by0[st], by1[st])
        ok[st[(lbVf[:, 0] < 0.0) | (gf < 0.0)]] = False
    return ok


def brake_funnel_cert(cfg, v_net, bg, npsi, n_stop=22, chunk=4096, verbose=True):
    """Maximal brake-invariant subset of {V_t>=0}: a cell is certified iff EVERY
    heading slice's braking trajectory stays safe (``_brake_traj_safe``).  This is
    the FALLBACK-PINNED filter (always brake) -- Phase 1."""
    lo, hi, ubV = bg["lo"], bg["hi"], bg["ubV"]
    work = np.flatnonzero(ubV >= 0.0)
    if len(work) == 0:
        return np.zeros(len(ubV), bool), {"n_work": 0, "accepted": 0}
    wlo, whi = lo[work], hi[work]
    ps_edges = np.linspace(-np.pi, np.pi, npsi + 1)
    cell_ok = np.ones(len(work), bool)
    pb = Progress(npsi, "brake-funnel") if verbose else None
    for hslice in range(npsi):
        cell_ok &= _brake_traj_safe(
            cfg, v_net, wlo[:, 0], whi[:, 0], wlo[:, 1], whi[:, 1],
            np.full(len(work), ps_edges[hslice]), np.full(len(work), ps_edges[hslice + 1]),
            wlo[:, 2], whi[:, 2], n_stop, chunk)
        if pb is not None:
            pb.update(hslice + 1)
    if pb is not None:
        pb.done()
    mask = np.zeros(len(ubV), bool)
    mask[work[cell_ok]] = True
    diag = {"n_work": int(len(work)), "accepted": int(cell_ok.sum()), "n_stop": n_stop}
    if verbose:
        print(f"  [funnel] {int(cell_ok.sum())}/{len(work)} cells brake-trajectory "
              f"safe (V_t>=0 & g>=0 to stop, {npsi} headings)", flush=True)
    return mask, diag


# --------------------------------------------------------------------------- #
def _racing_infeasible(cfg, v_net, q_net, lo, hi, pl, ph, gamma, chunk=4096):
    """SOUND skip test (Lemma E antecedent): the racing action u_race=(a_max,*)
    is Q-INFEASIBLE over the (cell x heading-slice), i.e. the learned predicate
    Phi = [min_d Q_t(x,u_race,d) >= gamma V_t] is provably FALSE everywhere ->
    the deployed filter BRAKES there (no racing closure needed).

    min_d Q_t <= min_k upper(Q_t(.,d_k)); if that < gamma*lbV then Phi is false.
    Steering uses the full [-delta_max, delta_max] (conservative: harder to skip,
    sound).  Returns bool mask (True = racing provably infeasible -> skip)."""
    n = len(lo)
    xl = np.column_stack([lo[:, 0], lo[:, 1], pl, lo[:, 2]])     # (px,py,psi,v) box
    xh = np.column_stack([hi[:, 0], hi[:, 1], ph, hi[:, 2]])
    lbV, _ = crown_bounds_chunked(v_net, lo[:, :3], hi[:, :3], True, chunk)
    lbV = lbV[:, 0]
    dprobe = d_probe_grid(cfg)
    min_uppderQ = np.full(n, np.inf)
    a_lo = np.full(n, cfg.a_max); a_hi = np.full(n, cfg.a_max)
    s_lo = np.full(n, -cfg.delta_max); s_hi = np.full(n, cfg.delta_max)
    for dk in dprobe:
        zlo = np.column_stack([xl, a_lo, s_lo, np.full(n, dk[0]), np.full(n, dk[1])])
        zhi = np.column_stack([xh, a_hi, s_hi, np.full(n, dk[0]), np.full(n, dk[1])])
        _, qhi = crown_bounds_chunked(q_net, zlo, zhi, True, chunk)
        min_uppderQ = np.minimum(min_uppderQ, qhi[:, 0])
    return min_uppderQ < gamma * lbV                # Phi provably false


def deployed_filter_cert(cfg, v_net, q_net, bg, s_brake_mask, npsi,
                         gamma=GAMMA, n_stop=22, chunk=4096, verbose=True):
    """CORE-CLAIM certificate: the DEPLOYED min-intervention learned Q-CBF filter
    (race u_race=(a_max, racing steer) iff Phi=[min_d Q_t>=gamma V_t], else brake)
    keeps the certified set invariant & safe.

    On S_brake (Phase-1 brake-invariant set), a cell is DEPLOYED-certified iff for
    EVERY heading slice: either (a) racing is provably Q-infeasible there (filter
    brakes -> S_brake handles it), or (b) the racing branch is safe -- one racing
    step stays V_t>=0 & g>=0 AND braking FROM the racing successor stays safe
    (``_brake_traj_safe``).  Soundness: over a cell Phi may hold for some x and
    fail for others, so we require BOTH the brake branch (already S_brake) and,
    unless Phi is provably false, the race branch.  The removed cells are exactly
    where the LEARNED Q_t over-permits racing into an unsafe successor (Theorem-S
    false feasibility) -- rho_brake - rho_deployed measures that gap.
    """
    lo, hi = bg["lo"], bg["hi"]
    cand = np.flatnonzero(s_brake_mask)             # only S_brake cells can qualify
    if len(cand) == 0:
        return np.zeros(len(s_brake_mask), bool), {"n_cand": 0, "accepted": 0}
    clo, chi = lo[cand], hi[cand]
    ps_edges = np.linspace(-np.pi, np.pi, npsi + 1)
    L, dt, tmax = cfg.wheelbase, cfg.dt, np.tan(cfg.delta_max + cfg.d_delta_max)
    cell_ok = np.ones(len(cand), bool)
    n_skip = np.zeros(len(cand), int)               # slices where racing skipped
    pb = Progress(npsi, "deployed") if verbose else None
    for hslice in range(npsi):
        pl = np.full(len(cand), ps_edges[hslice]); ph = np.full(len(cand), ps_edges[hslice + 1])
        # (a) antecedent skip: racing provably infeasible -> filter brakes
        phi_false = _racing_infeasible(cfg, v_net, q_net, clo, chi, pl, ph, gamma, chunk)
        n_skip += phi_false.astype(int)
        # (b) racing branch: one accel step (a_max, full D), heading spread by max turn
        need = ~phi_false
        race_ok = np.ones(len(cand), bool)
        idx = np.flatnonzero(need)
        if len(idx):
            # racing accel = a_max for v<v_max, but 0 (speed cap) at v=v_max;
            # cells touching v_max must cover BOTH -> a in [0, a_max] (sound).
            top = chi[idx, 2] >= cfg.v_max - 1e-9
            a_cmd_lo = np.where(top, 0.0, cfg.a_max)
            rx0, rx1, ry0, ry1, rv0, rv1 = successor_box(
                cfg, clo[idx, 0], chi[idx, 0], clo[idx, 1], chi[idx, 1],
                pl[idx], ph[idx], clo[idx, 2], chi[idx, 2],
                a_cmd_lo, cfg.a_max, -cfg.d_a_max, cfg.d_a_max)
            # racing step must itself be in {V_t>=0} & safe
            lbVr, _ = crown_bounds_chunked(
                v_net, np.column_stack([rx0, ry0, rv0]),
                np.column_stack([rx1, ry1, rv1]), True, chunk)
            gr = g_bounds_sq(cfg, rx0, rx1, ry0, ry1)
            step_ok = (lbVr[:, 0] >= 0.0) & (gr >= 0.0)
            # then braking from the racing successor must stay safe (full steer spread)
            Tr = dt * (rv1 / L) * tmax
            traj_ok = _brake_traj_safe(cfg, v_net, rx0, rx1, ry0, ry1,
                                       pl[idx] - Tr, ph[idx] + Tr, rv0, rv1, n_stop, chunk)
            r = step_ok & traj_ok
            race_ok[idx] = r
        # cell ok at this slice: brake handled by S_brake; if racing possible it must be safe
        cell_ok &= (phi_false | race_ok)
        if pb is not None:
            pb.update(hslice + 1)
    if pb is not None:
        pb.done()
    mask = np.zeros(len(s_brake_mask), bool)
    mask[cand[cell_ok]] = True
    diag = {"n_cand": int(len(cand)), "accepted": int(cell_ok.sum()),
            "mean_skip_frac": float(n_skip.mean() / npsi)}
    if verbose:
        print(f"  [deployed] {int(cell_ok.sum())}/{len(cand)} S_brake cells survive "
              f"the racing closure (racing Q-infeasible on {diag['mean_skip_frac']*100:.0f}% "
              f"of slices on avg)", flush=True)
    return mask, diag


# --------------------------------------------------------------------------- #
def audit_core(cfg, model, v_net, mask, bg, c_term, n_roll=2000, horizon=300,
               seed=1, verbose=True):
    """Light falsification of the CERTIFIED claim: from states inside Omega_term,
    roll the BRAKE fallback (a_min + any steering) under extremal/greedy d and
    confirm V_t stays >= c_term and g >= 0 (certified-but-violated must be 0)."""
    lo, hi = bg["lo"][mask], bg["hi"][mask]
    if len(lo) == 0:
        return {"certified_but_violated": 0, "note": "empty core"}
    rng = np.random.default_rng(seed)

    def sample(n):
        i = rng.integers(0, len(lo), n)
        u = rng.random((n, 3))
        p = lo[i] + u * (hi[i] - lo[i])             # (px,py,v) in a core cell
        ps = rng.uniform(-np.pi, np.pi, n)          # any heading
        return np.column_stack([p[:, 0], p[:, 1], ps, p[:, 2]])

    out = {}
    for mode in ("extremal", "greedy"):
        X = sample(n_roll)
        minV = v_net.forward(X[:, [0, 1, 3]])[:, 0].copy()
        ming = model.g(X).copy()
        for t in range(horizon):
            steer = np.clip(rng.uniform(-cfg.delta_max, cfg.delta_max, len(X)),
                            -cfg.delta_max, cfg.delta_max)
            u = np.column_stack([np.full(len(X), cfg.a_min), steer])  # brake fallback
            if mode == "extremal":
                d = np.column_stack([rng.choice([-cfg.d_a_max, cfg.d_a_max], len(X)),
                                     rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))])
            else:                                    # greedy: worst d_a (+), worst steer-d
                d = np.column_stack([np.full(len(X), cfg.d_a_max),
                                     rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))])
            X = model.step(X, u, d)
            np.minimum(minV, v_net.forward(X[:, [0, 1, 3]])[:, 0], out=minV)
            np.minimum(ming, model.g(X), out=ming)
        out[mode] = {"min_V": float(minV.min()), "min_g": float(ming.min()),
                     "V_below_cterm": int((minV < c_term - 1e-6).sum()),
                     "g_violations": int((ming < 0).sum())}
        if verbose:
            r = out[mode]
            print(f"  [audit:{mode:8s}] min V {r['min_V']:+.4f} (>= c_term? "
                  f"{r['min_V'] >= c_term - 1e-6})  min g {r['min_g']:+.4f}  "
                  f"g-viol {r['g_violations']}", flush=True)
    out["certified_but_violated"] = int(sum(out[m]["g_violations"]
                                            for m in ("extremal", "greedy")))
    return out


# --------------------------------------------------------------------------- #
def audit_deployed(cfg, model, v_net, q_net, pi_net, mask, bg, n_roll=2000,
                   horizon=300, seed=1, verbose=True):
    """Falsify the DEPLOYED min-intervention Q-CBF filter (the certified object):
    race u_race=(a_max, racing steer) iff min_d Q_t(x,u_race,d) >= gamma V_t(x),
    else brake -- rolled under extremal/greedy d from the deployed-certified cells.
    certified-but-violated must be 0."""
    lo, hi = bg["lo"][mask], bg["hi"][mask]
    if len(lo) == 0:
        return {"certified_but_violated": 0, "note": "empty deployed set"}
    rng = np.random.default_rng(seed)
    dprobe = d_probe_grid(cfg)

    def sample(n):
        i = rng.integers(0, len(lo), n); u = rng.random((n, 3))
        p = lo[i] + u * (hi[i] - lo[i])
        return np.column_stack([p[:, 0], p[:, 1], rng.uniform(-np.pi, np.pi, n), p[:, 2]])

    def deployed_u(X):
        v = X[:, 3]
        u_race = np.column_stack([np.where(v < cfg.v_max, cfg.a_max, 0.0),
                                  racing_steer(cfg, X)])
        Vx = v_net.forward(X[:, [0, 1, 3]])[:, 0]
        qmin = np.full(len(X), np.inf)
        for dk in dprobe:
            q = q_net.forward(np.concatenate([X, u_race, np.tile(dk, (len(X), 1))], 1))[:, 0]
            qmin = np.minimum(qmin, q)
        feas = qmin >= GAMMA * Vx                    # the learned predicate Phi
        u_brake = np.column_stack([np.full(len(X), cfg.a_min), racing_steer(cfg, X)])
        return np.where(feas[:, None], u_race, u_brake), feas

    out = {}
    for mode in ("extremal", "greedy"):
        X = sample(n_roll)
        ming = model.g(X).copy(); minV = v_net.forward(X[:, [0, 1, 3]])[:, 0].copy()
        race_frac = 0.0
        for t in range(horizon):
            u, feas = deployed_u(X)
            race_frac += feas.mean()
            if mode == "extremal":
                d = np.column_stack([rng.choice([-cfg.d_a_max, cfg.d_a_max], len(X)),
                                     rng.choice([-cfg.d_delta_max, cfg.d_delta_max], len(X))])
            else:                                    # greedy: worst d for next g
                best = None; bestg = None
                for dc in ([cfg.d_a_max, cfg.d_delta_max], [cfg.d_a_max, -cfg.d_delta_max],
                           [-cfg.d_a_max, cfg.d_delta_max], [cfg.d_a_max, 0.0]):
                    dd = np.tile(dc, (len(X), 1)); gn = model.g(model.step(X, u, dd))
                    if bestg is None: bestg, best = gn, dd
                    else:
                        take = gn < bestg; bestg = np.where(take, gn, bestg)
                        best = np.where(take[:, None], dd, best)
                d = best
            X = model.step(X, u, d)
            np.minimum(ming, model.g(X), out=ming)
            np.minimum(minV, v_net.forward(X[:, [0, 1, 3]])[:, 0], out=minV)
        out[mode] = {"min_g": float(ming.min()), "min_V": float(minV.min()),
                     "g_violations": int((ming < 0).sum()), "race_frac": float(race_frac / horizon)}
        if verbose:
            r = out[mode]
            print(f"  [audit:{mode:8s}] min g {r['min_g']:+.4f}  min V {r['min_V']:+.4f}  "
                  f"g-viol {r['g_violations']}  raced {100*r['race_frac']:.0f}%", flush=True)
    out["certified_but_violated"] = int(sum(out[m]["g_violations"] for m in ("extremal", "greedy")))
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--probe-only", action="store_true",
                    help="distill (full fidelity) + exact c_term probe, then STOP "
                         "(the cheap Gate-0a/0b firewall before any verifier run)")
    ap.add_argument("--c2fix", action="store_true",
                    help="distill the C2-repair object (distill c2_fix) -- the "
                         "locked Gate-0b path (soft, firewall-safe; no D(v)/hard-Lip)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=0,
                    help="override grid edge (uses FULL training); e.g. --res 48")
    args = ap.parse_args()
    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    out = REPO / "results" / "f1tenth_e2"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    eps = 5e-3
    levels = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

    if args.quick and not args.probe_only:
        res, npsi = (48, 48, 36), 12
        train_kw = dict(n_samples=30_000, reg_epochs=12, cbf_epochs=12)
        n_probe = 150_000
    else:                              # full-fidelity training (probe needs it)
        res, npsi = (80, 80, 60), 16
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=30)
        n_probe = 400_000
    if args.res:                       # moderate-grid override, full training
        n = args.res
        res, npsi = (n, n, int(round(n * 0.75))), 12
        train_kw = dict(n_samples=80_000, reg_epochs=30, cbf_epochs=30)
        n_probe = 400_000

    print("=" * 88)
    print("E2-L / RFC PHASE 1 -- relational terminal-core certificate (fallback-pinned)")
    print("  DEPLOYED learned trio, no D(v), no hard-Lipschitz; brake-invariant {V_t>=c_term}")
    print("=" * 88, flush=True)

    # ---- IRON RULE: analytic ideal first (the rho denominator Omega*) -------- #
    ideal = certified_set(cfg, model, 80, 80, 60)
    dec = verify_braking_decrease(cfg, model)
    print(f"[ideal] analytic V certifies {ideal['accepted']} cells @ 80^3, "
          f"exact braking decrease {dec:+.6f} -> "
          f"{'PASS' if ideal['accepted'] > 0 and dec >= -1e-9 else 'FAIL'} "
          f"(Omega* vol {ideal['volume']:.2f})", flush=True)

    # ---- distill the LEARNED trio (training is for non-vacuity only) --------- #
    v, q, pi, ddiag = distill(cfg, args.seed, margin=0.10, c2_fix=args.c2fix, **train_kw)
    vn, qn, pn = SeqNet.from_mlp(v), SeqNet.from_mlp(q), SeqNet.from_mlp(pi)
    print(f"[distill] seed={args.seed} c2_fix={args.c2fix} "
          f"MSE(V)={ddiag['mse_V_target']:.4f} "
          f"MSE(Q)={ddiag['mse_Q_target']:.4f} brake-agree={ddiag['pi_brake_agreement']:.2f}",
          flush=True)

    # ---- STEP 0: exact c_term probe (sets the scalar only; firewall) -------- #
    c_term, ptable = c_term_probe(cfg, model, vn, pn, levels, delta=0.05,
                                  eta_term=0.0, n_samples=n_probe, seed=args.seed)
    if args.probe_only:
        verdict = "Gate-0a (hole-free core EXISTS)" if c_term is not None else "Gate-0b (no hole-free core)"
        print(f"\n[PROBE-ONLY] {verdict}  c_term={c_term}  c2_fix={args.c2fix}", flush=True)
        (out / f"rfc_probe{'_c2fix' if args.c2fix else ''}.json").write_text(json.dumps(
            {"probe_only": True, "c2_fix": args.c2fix, "c_term": c_term,
             "probe": ptable, "distill": {k: ddiag.get(k) for k in
             ("mse_V_target", "mse_Q_target", "pi_brake_agreement", "c2_hinge")},
             "wall_s": round(time.time() - t0, 1)}, indent=2))
        return
    if c_term is None:
        # Gate 0b: no hole-free sub-level CORE (gamma=1 non-decrease holes are
        # level-independent -- exact).  Pivot to the maximal brake-invariant
        # SUBSET of {V_t>=0} via the SOUND H-step brake-to-stop funnel (only the
        # ~1-6% boundary EXIT cells, V_t(f)<0, obstruct it -- far fewer than the
        # 15% non-decrease holes).  Still the frozen learned object, fallback-pinned.
        print("[gate 0b] no hole-free sub-level core (level-independent non-decrease "
              "holes) -> pivot to the H-step brake-invariant SUBSET of {V_t>=0}",
              flush=True)
        print(f"\n=== brake-to-stop funnel  res={res} npsi={npsi} ===", flush=True)
        bg = base_grid(cfg, vn, *res)
        mask, fdiag = brake_funnel_cert(cfg, vn, bg, npsi)
        # C3 liveness: the LEARNED Q_t agrees the brake witness is feasible on the
        # certified set (min_d Q_t(x,pi_b,d) >= gamma V_t + eps) -- ties Q_t in, so
        # this is a deployed Q-CBF filter cert, not merely V_t brake-safety.
        cidx = np.flatnonzero(mask)
        c3 = witness_feasibility(cfg, vn, qn, pn, bg["lo"], bg["hi"], bg["ubV"],
                                 cidx, npsi, GAMMA, eps)
        fdiag["c3_live_frac"] = float(c3.mean()) if len(c3) else 0.0
        print(f"  [C3]   witness-feasible (Q_t live) on {fdiag['c3_live_frac']*100:.1f}% "
              f"of the certified set", flush=True)
        ideal_same = certified_set(cfg, model, *res)
        rho = float(fdiag["accepted"] * bg["cell_vol"] / max(ideal_same["volume"], 1e-9))
        print(f"  [rho]  funnel (fallback-pinned) {fdiag['accepted']} cells -> "
              f"rho_brake = {rho:.3f} (Omega* {ideal_same['accepted']} cells)", flush=True)

        # ---- CORE CLAIM: the DEPLOYED min-intervention Q-CBF filter ---------- #
        # (race when Q_t says feasible, else brake).  rho_brake - rho_deployed =
        # the LEARNED Q_t false-feasibility gap (Theorem S), verified post hoc.
        print(f"\n=== deployed Q-CBF filter (racing closure)  res={res} npsi={npsi} ===",
              flush=True)
        dmask, ddiag2 = deployed_filter_cert(cfg, vn, qn, bg, mask, npsi)
        n_carved = ddiag2["n_cand"] - ddiag2["accepted"]
        # SOUNDNESS: the racing closure proves S_brake is deployed-INVARIANT iff
        # it holds on ALL of S_brake (n_carved==0); then rho_deployed=rho_brake.
        # If cells are carved, the learned Q_t exhibits FALSE FEASIBILITY (permits
        # racing into a non-brake-safe state) -- the carved set is NOT a proven
        # invariant (multi-step racing can escape), it is the Theorem-S diagnostic.
        invariant = (n_carved == 0)
        rho_dep = rho if invariant else 0.0          # only a PROVEN invariant counts
        aud = audit_deployed(cfg, model, vn, qn, pn,
                             mask if invariant else dmask, bg)
        gate = bool(invariant and aud["certified_but_violated"] == 0)
        if invariant:
            print(f"  [rho]  racing closure holds on ALL {ddiag2['n_cand']} S_brake "
                  f"cells -> S_brake is DEPLOYED-INVARIANT, rho_deployed = {rho:.3f} "
                  f"(SOUND)", flush=True)
        else:
            print(f"  [Theorem-S] racing closure FAILS on {n_carved}/{ddiag2['n_cand']} "
                  f"S_brake cells -> learned Q_t FALSE FEASIBILITY; the racing filter "
                  f"is NOT a proven invariant (sound deployed object = brake-pinned "
                  f"rho={rho:.3f}). Q-conservative retraining needed to certify racing.",
                  flush=True)

        blob = {"phase": 2, "route": "deployed_qcbf_filter", "seed": args.seed,
                "c2_fix": args.c2fix, "res": list(res), "npsi": npsi,
                "ideal": ideal, "ideal_same_res": ideal_same,
                "rho_brake": rho, "rho_deployed": rho_dep,
                "false_feasibility_gap": rho - rho_dep,
                "funnel": fdiag, "deployed": ddiag2, "audit": aud, "probe": ptable,
                "gate_pass": gate, "wall_s": round(time.time() - t0, 1)}
        (out / "rfc_phase2_report.json").write_text(json.dumps(blob, indent=2))
        print("\n" + "=" * 88)
        print(f"DEPLOYED Q-CBF FILTER {'PASS' if gate else 'FAIL'}: "
              f"rho_deployed={rho_dep:.3f} (brake-pinned {rho:.3f})  "
              f"{ddiag2['accepted']} cells  certified-but-violated={aud['certified_but_violated']}")
        print(f"Wrote {out / 'rfc_phase2_report.json'}  ({blob['wall_s']:.0f}s)")
        return

    # ---- STEP 1: terminal-core certificate ---------------------------------- #
    print(f"\n=== terminal-core cert at c_term={c_term:.3f}  res={res} npsi={npsi} ===",
          flush=True)
    bg = base_grid(cfg, vn, *res)
    mask, diag = terminal_core_cert(cfg, vn, qn, pn, bg, c_term, npsi, eps=eps)
    # rho denominator: analytic Omega* volume on the SAME (px,py,v) resolution
    ideal_same = certified_set(cfg, model, *res)
    rho_term = float(diag["accepted"] * bg["cell_vol"] / max(ideal_same["volume"], 1e-9))
    print(f"  [rho]  Omega_term {diag['accepted']} cells, vol "
          f"{diag['accepted'] * bg['cell_vol']:.3f} / Omega* {ideal_same['volume']:.3f} "
          f"-> rho_term = {rho_term:.3f}", flush=True)

    # ---- light audit -------------------------------------------------------- #
    aud = audit_core(cfg, model, vn, mask, bg, c_term)

    gate1 = bool(diag["accepted"] > 0 and aud["certified_but_violated"] == 0)
    blob = {"phase": 1, "seed": args.seed, "c_term": c_term, "eps": eps,
            "res": list(res), "npsi": npsi, "ideal": ideal, "ideal_same_res": ideal_same,
            "rho_term": rho_term, "cert": diag, "audit": aud, "probe": ptable,
            "distill": {k: ddiag.get(k) for k in ("mse_V_target", "mse_Q_target",
                                                  "pi_brake_agreement")},
            "gate1_pass": gate1, "wall_s": round(time.time() - t0, 1)}
    (out / "rfc_phase1_report.json").write_text(json.dumps(blob, indent=2))

    print("\n" + "=" * 88)
    print(f"PHASE 1 VERDICT: {'PASS' if gate1 else 'FAIL'}  "
          f"rho_term={rho_term:.3f}  Omega_term={diag['accepted']} cells  "
          f"certified-but-violated={aud['certified_but_violated']}")
    print(f"  relational vs decoupled landing: T3 pass {diag['t3_rel_pass']} vs "
          f"{diag['t3_dec_pass']} (artifact tax removed: median "
          f"{diag['Lrel_minus_Ldec_med']:+.4f})")
    print(f"  Gate 1 {'MET -> proceed to Phase 2' if gate1 else 'NOT met -> report floor, do not widen scope'}")
    print(f"Wrote {out / 'rfc_phase1_report.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
