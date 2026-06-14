# Dubins E0 — Verifier-in-the-loop certified-training round (2026-06-14)

Consolidated result of the certified-training round on the learned Q-CBF artifact
`(V_theta, Q_theta, pi_theta)` for the fixed-speed Dubins car, certified with the
sound cell-worst CROWN/IBP verifier (Theorem A: C1 `{V>=0} ⊆ K`, C3 witness gate,
C4 `min_d Q ≤ min_d V(f)`).  All training is verifier-in-the-loop on the SAME
sound interval bounds the verifier checks (IBP-pass ⇒ deployed-CROWN-pass); none
of it enters the trusted computing base, and none of it uses the forbidden
Lipschitz / weight-decay / spectral flattening.

## Metrics

Per-gate pass rates are the cell-worst CROWN pass fractions on a sampled active
set.  The **headline coverage metric is the JOINT certified set** — the cells that
pass **C1 ∧ C3 ∧ C4 together** (on one common sample) — reported as a cell count
and as `rho = certified_volume / Vol(Omega*)`.  Per-gate rates can look healthy
while the joint set is empty, so the joint size is the honest bottom line.  (Now
computed by `run_learned_spec_diagnostic.py`, section `joint_certified_set`.)

## Results table

| config (lever)                     | C1 bad | C3 % | C4 menu % | C4 witn % | JOINT cells | rho |
|------------------------------------|-------:|-----:|----------:|----------:|------------:|----:|
| OFF (no certified lever)           |  3762  |  7.6 |   0.0     |   0.0     |     0       | 0.0 |
| one-sided C4 only                  |  3762  |  0.0 |  99.0     |  86.6     |     0       | 0.0 |
| two-sided C4+C3 (V0.12)            |  3762  | 27   |  21       |   7.4     |     0       | 0.0 |
| two-sided + v_dec_margin 0.25      |  2293  | 33.6 |  16.6     |   3.2     |     0       | 0.0 |
| two-sided + cell-worst V (decrease)| 30079  | 90.8 |   0.3     |   1.5     |     0       | 0.0 |
| two-sided + C1-floor-only V-cert   |   *23* | 45.7 |  11.0     |   2.0     |     0       | 0.0 |

(OFF = matched `cert_*_w=0` baseline.  All pass rates are cell-worst CROWN on
~1.5–2k sampled active cells; JOINT on a common sample.)

## What works (positive signals)

1. **Two-sided Q lever is sound and effective.** Verifier-in-the-loop CROWN-IBP
   eps-schedule training takes a previously-0% gate to high pass on the REAL
   CROWN verifier: one-sided drives C4 menu/witness 0% → 99%/87%; the two-sided
   variant (adding the C3 lower-gate push) is the only config where BOTH gates are
   simultaneously alive, and it resolves the C3/C4 tension as a mechanism (Q is
   squeezed into the band `[gamma*ubV+eps, lb V(f)-m]`).

2. **C1-floor-only V-cert is sound and effective.** The cell-worst C1 floor
   (`ub_IBP V(C) < -m` on every `g<0` cell, full coverage, teacher anchor on safe
   cells only, decrease push OFF) drives **C1 bad 3762 → 23** and sharpens
   `{V>=0}` strictly inside K — the cleanest C1 yet, and it lifts C3 (27 → 46%).

## The walls (why JOINT is still 0)

* **C3/C4 tension (Q side):** C4 wants Q low (`ub Q ≤ lb V(f)`), C3 wants Q high
  (`min_d Q(x,pi,d) ≥ gamma*V+eps`).  The two-sided lever balances them but at a
  level where neither is ~100% on the same cells.
* **Band-vs-slack / cannot widen by training (V side):** the CBF band cannot be
  widened by training — a bigger *pointwise* decrease margin roughens V and
  inflates its cell slack (`v_dec_margin 0.25` backfired); a cell-worst decrease
  push inflates V globally and destroys C1 (the `30079`-bad row).  The residual
  C4 slack is a *verifier* quantity (reducible only by finer cells).
* **C1-vs-C4 tension (V boundary):** pushing `{V>=0}` inside K (C1) lowers the
  boundary successor values C4 leans on, so fixing C1 (3762 → 23) *worsened* C4
  (menu 21 → 11%, witness 7.4 → 2.0%).
* **All-or-nothing strict spec:** the certificate accepts the inner set only if
  EVERY active cell passes C3 ∧ C4, so the joint must reach ~100% on active —
  far from the current per-gate rates.

## Bottom line

Every lever advances its own gate soundly and without flattening, but **no config
yet yields a single jointly-certified cell (JOINT = 0, rho = 0)** — the gates do
not co-pass on the same cells.  The closest state is the C1-floor-only run: C1
essentially solved (23), C3 ~46%, with **C4-witness (~2%) the binding wall**.

Next candidates (not yet run): finer verifier cells to shrink the C4 cell slack;
a softer / band-aware C1 floor to relieve the C1-vs-C4 boundary tension; and/or an
inner sub-region certificate (accept the passing sub-mask) to report a non-zero
`rho` instead of the strict all-or-nothing pass.
