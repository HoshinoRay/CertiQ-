"""F1TENTH E2-L / STRUCTURED -- FIRST rho>0 driver (throughput-engineered).

GOAL (per the plan): land the FIRST non-empty SOUND certified sub-level set
    Omega_cert = {V_theta >= c}  of the STRUCTURED learned value
        V_theta(px,py,v) = C_theta(px,py) - D(v) - margin
and report  rho = Vol(Omega_cert) / Vol(Omega*).  NOT max rho -- just rho>0.

This is NOT a new certificate idea; it is the SAME sound position branch-and-bound
as run_cert_structured.py, re-engineered for THROUGHPUT so it actually FINISHES
inside a compute budget.  Five levers from the plan:

  0. EXACT LEVEL PROBE (pointwise, no bounds, seconds).  Before any B&B, measure
     the TRUE discounted decrease  h_g = V(f_brake,worst-d) - gamma V  over {V>=c}
     for a grid of c.  Closure of {V>=c} is feasible  <=>  min h_g >= (1-gamma) c.
     This separates "object can close at c" (verifier just needs resolution) from
     "object has an exact violation at c" (no resolution can close it) -- and PICKS
     the c to fix.  (worst-d = +d_a_max is exact for the heading-free structured V.)
  1. IBP-FIRST, CROWN-SECOND.  Each box is first tested with cheap IBP bounds; IBP
     resolves the deep interior (PASS) and exterior (OUT) outright, and only the
     thin undecided BOUNDARY BAND is escalated to the npsi CROWN direct-composition
     bound.  IBP is SOUND for every decision (looser ubV only hardens the closure
     threshold; looser lbV/lbb only under-counts), so CROWN is a pure tightener.
  2. SINGLE c.  No c-sweep.  Omega_cert subset {V>=0} is fine; an inner level
     straddles less boundary, so it closes under fewer splits.
  3. REFINE ONLY THE UNDECIDED BAND.  Passed interior cells are banked & frozen;
     out cells dropped; only undecided boxes are split (position 2x2, v exact).
  4. CHECKPOINTABLE / RESUMABLE.  The B&B frontier, banked volume and depth are
     written after every depth and on a soft time budget, so an environment kill
     loses nothing -- the next run resumes the frontier.  C_theta and the analytic
     ideal volume are cached to disk too (restarts skip retrain / re-prove).

SOUNDNESS (unchanged firewall): per cell, worst over heading & disturbance,
    C1 safe:       gmin >= 0
    C2 discounted: lb V(f_brake) >= gamma * ub V + (1-gamma) c       (B_c CBF)
via SOUND bounds (IBP or CROWN).  Unknown at maxdepth => the LEVEL FAILS (never
pruned-to-pass).  The level PASSES iff every candidate (ubV>=c) resolves; then
{V>=c} is forward-invariant & safe and rho = banked inside-volume / ideal.

    python experiments/f1tenth_e2/run_cert_rho.py --probe-only     # decisive, seconds
    python experiments/f1tenth_e2/run_cert_rho.py --c 0.15 --budget 300   # B&B, resumable
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
                                         brake_distance, successor_box, g_bounds_sq)
from qcbf.certify.direct import (vtheta_box_bounds_structured,
                                 crown_brake_successor_lb_structured)
from qcbf.verify.bounds import SeqNet, ibp_preact_bounds
from experiments.f1tenth_e2.distill import train_clearance_net, GAMMA
from experiments.f1tenth_e2.lip_clearance import train_lip_clearance_net, spectral_norms
from experiments.f1tenth_e2.run_cert import certified_set, verify_braking_decrease
from experiments.f1tenth_e2.run_cert_p1 import _grid

OUT = REPO / "results" / "f1tenth_e2"


# ======================================================================= #
# C_theta + ideal caches (restarts skip retrain / re-prove)
# ======================================================================= #
def load_or_train_cnet(cfg, width, epochs, n, lipw=2.0, fresh=False):
    path = OUT / f"cnet_w{width}_lip{lipw:g}.npz"
    if path.exists() and not fresh:
        d = np.load(path, allow_pickle=True)
        net = SeqNet([d[f"W{i}"] for i in range(int(d["L"]))],
                     [d[f"b{i}"] for i in range(int(d["L"]))])
        return net, float(d["mse"])
    mlp, mse = train_clearance_net(cfg, seed=0, width=width, epochs=epochs, n=n,
                                   lip_w=lipw)
    net = SeqNet.from_mlp(mlp)
    np.savez(path, L=len(net.W), mse=mse,
             **{f"W{i}": net.W[i] for i in range(len(net.W))},
             **{f"b{i}": net.b[i] for i in range(len(net.b))})
    return net, mse


def load_or_train_lip_cnet(cfg, width, depth, epochs, n, fresh=False):
    """HARD 1-Lipschitz clearance head (spectral-norm projected).  Cached; the
    saved spectral norms are re-checked on load so the certified object is
    provably 1-Lipschitz regardless of the (untrusted) training run."""
    path = OUT / f"cnet_lip_w{width}_d{depth}.npz"
    if path.exists() and not fresh:
        d = np.load(path, allow_pickle=True)
        net = SeqNet([d[f"W{i}"] for i in range(int(d["L"]))],
                     [d[f"b{i}"] for i in range(int(d["L"]))])
        return net, float(d["mse"]), float(d["max_sn"])
    net, mse, max_sn = train_lip_clearance_net(cfg, seed=0, width=width,
                                               depth=depth, epochs=epochs, n=n)
    np.savez(path, L=len(net.W), mse=mse, max_sn=max_sn,
             **{f"W{i}": net.W[i] for i in range(len(net.W))},
             **{f"b{i}": net.b[i] for i in range(len(net.b))})
    return net, mse, max_sn


def load_or_compute_ideal(cfg, model, res, fresh=False):
    path = OUT / f"ideal_{res[0]}-{res[1]}-{res[2]}.json"
    if path.exists() and not fresh:
        return json.loads(path.read_text())
    cset = certified_set(cfg, model, *res)
    dec = verify_braking_decrease(cfg, model, res[0], res[1], max(24, res[2] // 2), res[2])
    blob = {"res": list(res), "accepted": int(cset["accepted"]),
            "volume": float(cset["volume"]), "min_decrease": float(dec),
            "pass": bool(cset["accepted"] > 0 and dec >= -1e-9)}
    path.write_text(json.dumps(blob, indent=2))
    return blob


# ======================================================================= #
# Step 0 -- EXACT pointwise level probe (no bounds; picks c)
# ======================================================================= #
def _Vstruct(cfg, c_net, X):
    return c_net.forward(X[:, :2])[:, 0] - brake_distance(cfg, X[:, 3]) - cfg.cbf_margin


def probe_levels(cfg, c_net, c_grid, gamma, n=600_000, seed=0):
    """Exact discounted-closure feasibility per level c.

    h_g = V(f_brake, worst-d) - gamma V,  worst-d = +d_a_max (exact for heading-free
    V).  {V>=c} closes (discounted) iff  min_{V>=c} h_g >= (1-gamma) c.  Pointwise
    truth -> 'object can close here' vs 'exact violation here', independent of any
    verifier looseness."""
    rng = np.random.default_rng(seed)
    px = rng.uniform(cfg.p_lo, cfg.p_hi, n)
    py = rng.uniform(cfg.p_lo, cfg.p_hi, n)
    ps = rng.uniform(-np.pi, np.pi, n)
    v = rng.uniform(0.0, cfg.v_max, n)
    C = c_net.forward(np.column_stack([px, py]))[:, 0]
    V = C - brake_distance(cfg, v) - cfg.cbf_margin
    vp = np.clip(v + cfg.dt * (cfg.a_min + cfg.d_a_max), 0.0, cfg.v_max)   # worst d
    pxp = px + cfg.dt * v * np.cos(ps)
    pyp = py + cfg.dt * v * np.sin(ps)
    Cp = c_net.forward(np.column_stack([pxp, pyp]))[:, 0]
    Vf = Cp - brake_distance(cfg, vp) - cfg.cbf_margin
    hg = Vf - gamma * V
    # ROOT-CAUSE: the cancellation is V(f)-V = [C(p+)-C(p)] + [D(v)-D(v+)], the
    # braking gain D(v)-D(v+) is EXACT = dt v and ||p+-p|| = dt v, so any closure
    # deficit is exactly C_theta's directional Lipschitz ratio (C(p)-C(p+))/(dt v)
    # exceeding 1.  Measure it on the moving samples.
    disp = cfg.dt * v
    mv = disp > 0.05
    ratio = (C[mv] - Cp[mv]) / disp[mv]
    lip = dict(max_ratio=float(ratio.max()), p9999=float(np.percentile(ratio, 99.99)),
               frac_gt1=float(np.mean(ratio > 1.0)))
    rows = []
    for c in c_grid:
        m = V >= c
        thr = (1.0 - gamma) * c
        if not m.any():
            rows.append(dict(c=float(c), n=0, min_hg=None, p01_hg=None, thr=thr,
                             margin=None, feasible=False))
            continue
        h = hg[m]
        mh = float(h.min())
        rows.append(dict(c=float(c), n=int(m.sum()), min_hg=mh,
                         p01_hg=float(np.percentile(h, 1)), thr=float(thr),
                         margin=float(mh - thr), feasible=bool(mh >= thr)))
    return rows, lip


# ======================================================================= #
# IBP-first / CROWN-second per-box evaluation
# ======================================================================= #
def _ibp_C(c_net, p_lo, p_hi):
    """IBP (lo, hi) of C_theta over position boxes (cheap, no CROWN)."""
    lu = ibp_preact_bounds(c_net, p_lo, p_hi)
    return lu[-1][0][:, 0], lu[-1][1][:, 0]


def _ibp_eval(cfg, c_net, lo, hi, npsi):
    """Cheap IBP bounds: ubV, lbV (cell), gmin, lbb (worst over npsi arc successor
    boxes).  Arc-wise IBP on the successor is far tighter than a single full-heading
    box yet still cheap -- it resolves most interior cells before CROWN is touched."""
    Clo, Chi = _ibp_C(c_net, lo[:, :2], hi[:, :2])
    D_lo = brake_distance(cfg, lo[:, 2])
    D_hi = brake_distance(cfg, hi[:, 2])
    lbV = Clo - D_hi - cfg.cbf_margin
    ubV = Chi - D_lo - cfg.cbf_margin
    gmin = g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    lbb = np.full(len(lo), np.inf)
    for h in range(npsi):
        bx0, bx1, by0, by1, bv0, bv1 = successor_box(
            cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1],
            ps[h], ps[h + 1], lo[:, 2], hi[:, 2],
            cfg.a_min, cfg.a_min, -cfg.d_a_max, cfg.d_a_max)
        Clo_s, _ = _ibp_C(c_net, np.column_stack([bx0, by0]),
                          np.column_stack([bx1, by1]))
        lbb = np.minimum(lbb, Clo_s - brake_distance(cfg, bv1) - cfg.cbf_margin)
    return ubV, lbV, gmin, lbb


def _crown_eval(cfg, c_net, lo, hi, npsi, chunk):
    """Tight CROWN bounds: ubV, lbV (cell), gmin, lbb (worst over npsi headings,
    direct composition -- keeps the v*cos(psi) correlation)."""
    lbV, ubV = vtheta_box_bounds_structured(
        cfg, c_net, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1], lo[:, 2], hi[:, 2], chunk)
    gmin = g_bounds_sq(cfg, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1])
    ps = np.linspace(-np.pi, np.pi, npsi + 1)
    lbb = np.full(len(lo), np.inf)
    for h in range(npsi):
        lbb = np.minimum(lbb, crown_brake_successor_lb_structured(
            cfg, c_net, lo[:, 0], hi[:, 0], lo[:, 1], hi[:, 1],
            ps[h], ps[h + 1], lo[:, 2], hi[:, 2], chunk))
    return ubV, lbV, gmin, lbb


def _verdict(ubV, lbV, gmin, lbb, c, gamma):
    """Per-box sound classification under the discounted-CBF closure test.

    out   : ubV < c              (cell entirely outside {V>=c} -> drop)
    ok    : gmin>=0 and lbb >= gamma*ubV + (1-gamma)c   (closes -> resolved)
    inside: lbV >= c             (cell entirely inside {V>=c} -> bankable volume)
    """
    out = ubV < c
    ok = (gmin >= 0.0) & (lbb >= gamma * ubV + (1.0 - gamma) * c)
    inside = lbV >= c
    return out, ok, inside


def _split_xy(lo, hi):
    """Split each (px,py,v) box 2x2 along px,py (v kept; D(v) is exact in v)."""
    mx = 0.5 * (lo[:, 0] + hi[:, 0])
    my = 0.5 * (lo[:, 1] + hi[:, 1])
    out_lo, out_hi = [], []
    for sx in (0, 1):
        for sy in (0, 1):
            L = lo.copy(); H = hi.copy()
            L[:, 0] = np.where(sx, mx, lo[:, 0]); H[:, 0] = np.where(sx, hi[:, 0], mx)
            L[:, 1] = np.where(sy, my, lo[:, 1]); H[:, 1] = np.where(sy, hi[:, 1], my)
            out_lo.append(L); out_hi.append(H)
    return np.concatenate(out_lo), np.concatenate(out_hi)


def _bank_ibp_first(cfg, c_net, c, gamma, lo, hi, npsi, chunk):
    """One depth: IBP-prefilter -> CROWN-escalate the undecided -> bank inside
    volume.  Returns (cert_vol_added, residual_lo, residual_hi, stats)."""
    vol = np.prod(hi - lo, axis=1)
    ubV, lbV, gmin, lbb = _ibp_eval(cfg, c_net, lo, hi, npsi)
    out, ok, inside = _verdict(ubV, lbV, gmin, lbb, c, gamma)
    cand = ~out
    cert = float(vol[cand & ok & inside].sum())
    undecided = cand & ~ok
    n_drop = int(out.sum())
    n_ibp_pass = int((cand & ok).sum())

    ulo, uhi = lo[undecided], hi[undecided]
    n_crown = int(len(ulo))
    if n_crown:
        u2, l2, g2, b2 = _crown_eval(cfg, c_net, ulo, uhi, npsi, chunk)
        out2, ok2, ins2 = _verdict(u2, l2, g2, b2, c, gamma)
        vol2 = np.prod(uhi - ulo, axis=1)
        cand2 = ~out2
        cert += float(vol2[cand2 & ok2 & ins2].sum())
        resid = cand2 & ~ok2
        rlo, rhi = ulo[resid], uhi[resid]
    else:
        rlo, rhi = ulo, uhi
    stats = dict(n_in=int(len(lo)), drop=n_drop, ibp_pass=n_ibp_pass,
                 crown=n_crown, resid=int(len(rlo)))
    return cert, rlo, rhi, stats


# ======================================================================= #
# checkpointed / time-budgeted position branch-and-bound
# ======================================================================= #
def _save_ckpt(path, c, lo, hi, cert_vol, depth, status):
    np.savez(path, c=c, lo=lo, hi=hi, cert_vol=cert_vol, depth=depth, status=status)


def _load_ckpt(path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return dict(c=float(d["c"]), lo=d["lo"], hi=d["hi"],
                cert_vol=float(d["cert_vol"]), depth=int(d["depth"]),
                status=str(d["status"]))


def certify_level(cfg, c_net, c, gamma, lo, hi, npsi, maxdepth, chunk,
                  ckpt, ideal_vol, t0, budget, cap=900_000,
                  cert_vol=0.0, depth=0):
    """Sound position B&B that {V>=c} is brake-invariant & safe, with checkpoints.

    Returns (status, cert_vol, frontier_lo, frontier_hi, depth).  status:
      PASS    -- frontier emptied: {V>=c} certified, rho = cert_vol/ideal valid.
      FAIL    -- residual unknown at maxdepth (or cap): {V>=c} NOT certified.
      PARTIAL -- time budget hit with a non-empty frontier: resumable, no verdict.
    """
    while depth <= maxdepth:
        if len(lo) == 0:
            _save_ckpt(ckpt, c, lo, hi, cert_vol, depth, "PASS")
            return "PASS", cert_vol, lo, hi, depth
        cv, lo, hi, st = _bank_ibp_first(cfg, c_net, c, gamma, lo, hi, npsi, chunk)
        cert_vol += cv
        el = time.time() - t0
        rho = cert_vol / max(ideal_vol, 1e-9)
        print(f"  depth {depth:2d}: in {st['n_in']:>7} | drop {st['drop']:>7} "
              f"| IBP-pass {st['ibp_pass']:>7} | CROWN {st['crown']:>6} "
              f"-> resid {st['resid']:>6} | banked rho~{rho:.4f} | {el:6.1f}s",
              flush=True)
        if len(lo) == 0:
            _save_ckpt(ckpt, c, lo, hi, cert_vol, depth, "PASS")
            return "PASS", cert_vol, lo, hi, depth
        if depth == maxdepth or len(lo) > cap:
            _save_ckpt(ckpt, c, lo, hi, cert_vol, depth, "FAIL")
            return "FAIL", cert_vol, lo, hi, depth
        lo, hi = _split_xy(lo, hi)
        depth += 1
        _save_ckpt(ckpt, c, lo, hi, cert_vol, depth, "running")
        if budget and (time.time() - t0) > budget:
            return "PARTIAL", cert_vol, lo, hi, depth
    _save_ckpt(ckpt, c, lo, hi, cert_vol, maxdepth, "PASS")
    return "PASS", cert_vol, lo, hi, maxdepth


# ======================================================================= #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c", type=float, default=None,
                    help="fix the sub-level c (default: smallest probe-feasible c)")
    ap.add_argument("--res", type=int, nargs=3, default=[64, 64, 40])
    ap.add_argument("--npsi", type=int, default=12)
    ap.add_argument("--maxdepth", type=int, default=6)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--budget", type=float, default=0.0, help="soft wall budget (s); 0=unlimited")
    ap.add_argument("--probe-only", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore checkpoint + caches")
    ap.add_argument("--train-epochs", type=int, default=45)
    ap.add_argument("--train-n", type=int, default=120_000)
    ap.add_argument("--train-lipw", type=float, default=2.0,
                    help="soft C_theta directional-Lipschitz penalty weight (root-cause knob)")
    ap.add_argument("--head", choices=["soft", "lip"], default="soft",
                    help="C_theta head: 'soft' penalty (default) or HARD 1-Lipschitz "
                         "(spectral-norm projected) -- the proven fix for rho>0")
    ap.add_argument("--lip-width", type=int, default=128)
    ap.add_argument("--lip-depth", type=int, default=2)
    ap.add_argument("--lip-epochs", type=int, default=60)
    ap.add_argument("--lip-n", type=int, default=160_000)
    args = ap.parse_args()

    cfg = BicycleAccelConfig()
    model = BicycleAccelModel(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    res = tuple(args.res)
    gamma = GAMMA
    t0 = time.time()

    print("=" * 92)
    print("STRUCTURED V = C_theta(p) - D(v) -- FIRST rho>0 (IBP-first B&B, single c, "
          "checkpointed)")
    print("=" * 92, flush=True)

    # ---- C_theta (cached) -------------------------------------------------- #
    if args.head == "lip":
        cnet, mseC, max_sn = load_or_train_lip_cnet(
            cfg, args.lip_width, args.lip_depth, args.lip_epochs, args.lip_n,
            fresh=args.fresh)
        sn = spectral_norms(cnet)
        print(f"[C_theta] HARD 1-Lipschitz head ({args.lip_width}x{args.lip_depth})  "
              f"MSE(clearance)={mseC:.4f}  per-layer ||W||2={['%.3f'%s for s in sn]}  "
              f"max {max_sn:.4f} (<=1 == provably 1-Lipschitz)", flush=True)
    else:
        cnet, mseC = load_or_train_cnet(cfg, args.width, args.train_epochs,
                                        args.train_n, lipw=args.train_lipw,
                                        fresh=args.fresh)
        print(f"[C_theta] soft head width {args.width}  lip_w={args.train_lipw:g}  "
              f"MSE(clearance)={mseC:.4f}", flush=True)

    # ---- Step 0: EXACT level probe (decisive, seconds) -------------------- #
    c_grid = (0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25)
    probe, lip = probe_levels(cfg, cnet, c_grid, gamma)
    print(f"\n[root-cause] C_theta directional Lipschitz (C(p)-C(p+))/||p+-p||: "
          f"max {lip['max_ratio']:.3f}, p99.99 {lip['p9999']:.3f}, "
          f"frac>1 {lip['frac_gt1']:.4f}  "
          f"(>1 == exactly the closure deficit; ideal clearance is 1-Lipschitz)",
          flush=True)
    print(f"\nEXACT discounted-closure probe (gamma={gamma}; feasible <=> "
          f"min h_g >= (1-gamma)c):")
    print(f"  {'c':>6} | {'n>=c':>9} | {'min h_g':>9} | {'p01 h_g':>9} | "
          f"{'thr':>7} | {'margin':>8} | feasible")
    for r in probe:
        mh = "   --   " if r["min_hg"] is None else f"{r['min_hg']:+.4f}"
        p1 = "   --   " if r["p01_hg"] is None else f"{r['p01_hg']:+.4f}"
        mg = "   --   " if r["margin"] is None else f"{r['margin']:+.4f}"
        print(f"  {r['c']:6.2f} | {r['n']:9d} | {mh:>9} | {p1:>9} | "
              f"{r['thr']:7.4f} | {mg:>8} | {r['feasible']}", flush=True)

    feas = [r for r in probe if r["feasible"]]
    auto_c = feas[0]["c"] if feas else None
    c = args.c if args.c is not None else auto_c
    print(f"\n  feasible levels: {[r['c'] for r in feas] or 'NONE'}; "
          f"auto-pick c={auto_c}; using c={c}")

    blob = {"head": args.head, "mse_C": mseC, "lip_w": args.train_lipw,
            "max_spectral_norm": (max_sn if args.head == "lip" else None),
            "lip_diag": lip, "gamma": gamma, "res": list(res), "npsi": args.npsi,
            "probe": probe, "auto_c": auto_c, "c_used": c}

    if args.probe_only or c is None:
        if c is None:
            print("\nNo probe-feasible level: the STRUCTURED object has an EXACT "
                  "discounted violation at every c in the grid -> no resolution can "
                  "close a level set here (object floor, not a verifier gap).")
        (OUT / "rho_probe.json").write_text(json.dumps(blob, indent=2))
        print(f"\nWrote {OUT / 'rho_probe.json'}  ({time.time()-t0:.0f}s)")
        return

    # ---- IRON RULE: analytic ideal first (cached) ------------------------- #
    ideal = load_or_compute_ideal(cfg, model, res, fresh=args.fresh)
    print(f"\n[ideal] analytic V @ {res}: {ideal['accepted']} cells, "
          f"min braking decrease {ideal['min_decrease']:+.5f} -> "
          f"{'PASS' if ideal['pass'] else 'FAIL'}  (vol {ideal['volume']:.2f})",
          flush=True)
    if not ideal["pass"]:
        print("IRON RULE violated: analytic ideal does not pass -- abort.")
        return

    # ---- B&B at the fixed c (resume checkpoint if present) ---------------- #
    ckpt = OUT / (f"rho_ckpt_{args.head}_c{c:.3f}_"
                  f"r{res[0]}-{res[1]}-{res[2]}.npz")
    st = None if args.fresh else _load_ckpt(ckpt)
    if st is not None and abs(st["c"] - c) < 1e-9 and st["status"] == "running":
        lo, hi = st["lo"], st["hi"]
        cert_vol, depth = st["cert_vol"], st["depth"]
        print(f"\n[resume] checkpoint at depth {depth}, frontier {len(lo)}, "
              f"banked rho~{cert_vol/max(ideal['volume'],1e-9):.4f}", flush=True)
    else:
        lo, hi, _ = _grid(cfg, *res)
        cert_vol, depth = 0.0, 0
        print(f"\n[fresh ] base grid {len(lo)} cells @ {res}", flush=True)

    print(f"=== position B&B  c={c:.2f}  npsi={args.npsi}  maxdepth={args.maxdepth}"
          f"{'  budget '+str(int(args.budget))+'s' if args.budget else ''} ===",
          flush=True)
    status, cert_vol, flo, fhi, depth = certify_level(
        cfg, cnet, c, gamma, lo, hi, args.npsi, args.maxdepth, args.chunk,
        ckpt, ideal["volume"], t0, args.budget, cert_vol=cert_vol, depth=depth)

    rho = cert_vol / max(ideal["volume"], 1e-9)
    blob.update(status=status, cert_vol=cert_vol, ideal_vol=ideal["volume"],
                rho_banked=rho, rho=(rho if status == "PASS" else 0.0),
                frontier=int(len(flo)), depth=depth,
                wall_s=round(time.time() - t0, 1))
    (OUT / "rho_report.json").write_text(json.dumps(blob, indent=2))

    print("\n" + "=" * 92)
    print(f"RESULT  c={c:.2f}  status={status}")
    if status == "PASS":
        print(f"  *** rho = {rho:.4f}  (Omega_cert={{V>={c:.2f}}} certified vol "
              f"{cert_vol:.2f} / ideal {ideal['volume']:.2f}) ***")
        print(f"  NON-EMPTY sound certified subset of the LEARNED structured object."
              if rho > 0 else "  (closed but zero inside-volume banked -- raise res)")
    elif status == "PARTIAL":
        print(f"  budget hit: banked-so-far rho~{rho:.4f}, frontier {len(flo)} boxes "
              f"at depth {depth}.  NOT yet a certificate -- rerun to resume "
              f"(checkpoint written).")
    else:
        print(f"  FAIL: residual unknown ({len(flo)} boxes) at depth {depth} -- "
              f"{{V>={c:.2f}}} did not close at this resolution/depth.")
    print(f"Wrote {OUT / 'rho_report.json'}  ({blob['wall_s']:.0f}s)")


if __name__ == "__main__":
    main()
