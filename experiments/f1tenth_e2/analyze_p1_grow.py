"""Post-process the P1 grow-from-seed resolution sweep.

Reads the per-resolution logs `results/f1tenth_e2/p1_grow_res*.log` and the
`p1_grow_full.json` calibration, and emits:

  (1) the resolution table (rho_brake, rho_inf, Delta-rho, 4-way failure
      decomposition) for learned + ideal;
  (2) an EMPIRICAL scaling extrapolation rho_inf(h) = rho_star - a*h^p
      (h = position cell width = 6/npx), pure-numpy fit, h->0 coverage --
      NOT called Richardson because h_p, h_v, dpsi do not co-refine;
  (3) the residual stacked decomposition of Omega* - rho_inf^learned into
      {V_theta gate, verifier conservatism, capture-basin gap} from the
      capture calibration.

    python experiments/f1tenth_e2/analyze_p1_grow.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
LOGDIR = REPO / "results" / "f1tenth_e2"


def _f(pat, text, grp=1, cast=float):
    m = re.search(pat, text)
    return cast(m.group(grp)) if m else None


def parse_log(path):
    t = path.read_text(errors="ignore")
    g = re.search(r"grid (\d+)x(\d+)x(\d+)x(\d+)", t)
    if not g:
        return None
    npx, npy, npsi, nv = map(int, g.groups())
    rec = {"file": path.name, "npx": npx, "npsi": npsi, "nv": nv,
           "h": 6.0 / npx}
    rec["rho_brake"] = _f(r"rho_brake \(seed\)\s*:\s*([\d.]+)", t)
    rec["rho_inf"] = _f(r"rho_inf \(grown\)\s*:\s*([\d.]+)", t)
    rec["drho"] = _f(r"Delta-rho\s*:\s*([+\-][\d.]+)", t)
    im = re.search(r"\[ideal\] rho_seed=([\d.]+) -> rho_inf=([\d.]+)", t)
    if im:
        rec["ideal_seed"], rec["ideal_inf"] = float(im.group(1)), float(im.group(2))
        rec["ideal_drho"] = rec["ideal_inf"] - rec["ideal_seed"]
    # the LEARNED summary line is "failure decomposition     : {...}" (spaces
    # before the colon); the ideal line is "...decomposition: {...}" (no space).
    bm = re.search(r"failure decomposition\s+:\s*(\{[^}]*\})", t)
    if bm:
        rec["breakdown"] = eval(bm.group(1))
    return rec


def fit_scaling(hs, ys):
    """Least-squares fit y = y_star - a*h^p over a small p-grid (pure numpy).
    Returns (y_star, a, p, rms)."""
    hs = np.asarray(hs, float); ys = np.asarray(ys, float)
    if len(hs) < 3:
        return None
    best = None
    for p in np.linspace(0.5, 2.5, 41):
        A = np.column_stack([np.ones_like(hs), -hs ** p])      # [y_star, a]
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        rms = float(np.sqrt(np.mean((A @ coef - ys) ** 2)))
        if best is None or rms < best[-1]:
            best = (float(coef[0]), float(coef[1]), float(p), rms)
    return best


def main():
    logs = sorted(LOGDIR.glob("p1_grow_res*.log"))
    recs = [r for r in (parse_log(p) for p in logs) if r and r["rho_inf"]]
    # de-dup by (npx,npsi), keep last
    by = {}
    for r in recs:
        by[(r["npx"], r["npsi"])] = r
    recs = sorted(by.values(), key=lambda r: (r["npsi"], r["npx"]))

    print("=" * 92)
    print("P1 grow resolution sweep")
    print("=" * 92)
    print(f"{'grid':>16} {'h':>6} {'rho_brake':>10} {'rho_inf':>9} {'d_rho':>8} "
          f"{'ideal_d':>8}  failure decomposition (frontier/v0/k/dom)")
    for r in recs:
        bd = r.get("breakdown", {})
        bstr = (f"{bd.get('frontier_blocked','-')}/{bd.get('v0_blocked','-')}/"
                f"{bd.get('k_blocked','-')}/{bd.get('domain_blocked','-')}")
        idr = f"{r.get('ideal_drho', float('nan')):+.4f}" if r.get("ideal_drho") is not None else "  -  "
        print(f"{r['npx']}x{r['npx']}x{r['npsi']}x{r['nv']:>3}".rjust(16)
              + f" {r['h']:.3f} {r['rho_brake']:>10.4f} {r['rho_inf']:>9.4f} "
              f"{r['drho']:>+8.4f} {idr:>8}  {bstr}")

    # ---- scaling extrapolation (npsi=16 family) --------------------------- #
    fam = [r for r in recs if r["npsi"] == 16]
    if len(fam) >= 3:
        hs = [r["h"] for r in fam]
        print("\n" + "-" * 92)
        print(f"Empirical scaling fit rho_inf(h)=rho_star - a*h^p   (npsi=16, "
              f"{len(fam)} points, h=6/npx)")
        for key, lab in (("rho_inf", "learned"), ("ideal_inf", "ideal ")):
            ys = [r.get(key) for r in fam]
            if all(y is not None for y in ys):
                fit = fit_scaling(hs, ys)
                if fit:
                    ystar, a, p, rms = fit
                    print(f"  {lab}: rho_star(h->0) = {ystar:.3f}  (a={a:.3f}, p={p:.2f}, "
                          f"rms={rms:.4f})  [finest grid {ys[-1]:.3f}]")

    # ---- residual decomposition from the --full calibration --------------- #
    fj = LOGDIR / "p1_grow_full.json"
    if fj.exists():
        blob = json.loads(fj.read_text())
        cal = blob.get("calibration"); dep = blob.get("deployed")
        ts = blob.get("theorem_s"); av = blob.get("a_ver")
        if cal:
            print("\n" + "-" * 92)
            print(f"Capture-basin calibration @ res {blob['config']['res']} "
                  f"(reference, n_d={cal['n_d_grid']} -- optimistic upper ref):")
            print(f"  rho   R_inf={cal['rho_R']:.3f}   capt(K&V0)={cal['rho_capt_KV0']:.3f}"
                  f"   capt(K)={cal['rho_capt_K']:.3f}")
            print(f"  recovery within anchor   = {cal['recovery_within_anchor']:.3f}  "
                  f"(method conservatism = {cal['conservatism']:+.3f} of Omega*)")
            print(f"  V0-gate loss             = {cal['v0_gate_loss']:+.3f}  "
                  f"(reachable in K but lb V_theta<0)")
            print(f"  capture BEYOND Omega*    = {cal['beyond_omega_frac']:+.3f}  "
                  f"(Viab > Omega*: brake-anchor understates the safe set)")
            print("  residual stack  Omega* = R_inf + conservatism + V0-gate + (capture-gap)")
            print(f"                  {cal['rho_R']:.3f} + {cal['conservatism']:.3f} + "
                  f"{cal['v0_gate_loss']:.3f} + ...   (capt(K) exceeds 1 => gap<0)")
        if av:
            print(f"\n  A_ver: mean {av['mean_A_ver']:.2f} certified actions/cell, "
                  f"{100*av['frac_A_ver_ge1']:.0f}% of R_inf can race")
        if dep:
            print("  Deployed three-way (cbv / min g / race% / mean v):")
            for name in ("shield", "naive", "brake"):
                d = dep[name]
                qa = f" q-agree {100*d['q_agree_frac']:.0f}%" if d.get("q_agree_frac") is not None else ""
                print(f"    {name:6s}: cbv {d['cbv']:5d}  min g {d['min_g']:+.3f}  "
                      f"race {100*d['race_frac']:3.0f}%  mean v {d['mean_speed']:.2f}{qa}")
        if ts:
            print(f"  Theorem-S: {ts['false_feasible_pairs']} false-feasible (c,u) "
                  f"= {100*ts['false_feasible_frac_of_phi']:.1f}% of Phi-feasible escape R_inf")


if __name__ == "__main__":
    main()
