# F1TENTH E2-L — Certifying the **Learned** Filter on the Brakeable Bicycle (**Gate D: FAIL under the cell+box primitive; open under tighter primitives**)

> **Scope correction (2026-06-11).** An earlier version of this write-up concluded
> the learned barrier is "sound to certify but not non-vacuously certifiable here."
> That over-scopes the evidence. **Route B erodes the analytic *ideal* `V` to empty
> too** (116784→0), so its failure characterises the **certification primitive**
> (outward-rounded cell+box one-step reachability), not the learned object — a
> method that cannot close on a known-invariant ideal set cannot prove the learned
> object uncertifiable. **Route A**'s knife-edge failure is a genuine
> learned-vs-analytic structural fact, but is specific to the **brake-only**
> sub-level under the box primitive. Whether a tighter sound primitive
> (direct-composition CROWN / zonotope / contraction) certifies the learned trio
> is the **open experimental question** (P1 in progress — `run_cert_p1.py`).

## P1 result (`run_cert_p1.py`, 2026-06-11) — primitive validated, sub-level knife-edge is a clean learned-vs-ideal fact

The first replacement primitive, **P1 = direct composition** (Route 1, dev doc
§8.5): bound `V_θ(f_brake(x,d))` directly by minimising the **CROWN affine lower
functional** of `V_θ` over the *true* nonlinear successor — **no intermediate
outward-rounded successor box** (`qcbf/certify/direct.py`,
`crown_lower_affine` + `crown_brake_successor_lb`; randomized soundness in
`tests/test_direct.py`).

**Iron rule — IDEAL PASSES.** Under P1's exact (structural) limit the analytic
ideal `V` certifies **86,580 cells** (22.5% of the (p,v) domain), min braking
decrease ≈ `−3e-16` ≈ exactly 0. This **directly refutes the old "even the ideal
fails" claim** — that was a Route-B (cell-rounding) artefact; a tight one-step
primitive certifies the known-invariant ideal non-vacuously.

**Learned trio (3 seeds, 80³ grid, 0.075 m cells):** still **FAIL** (no level
closes), but now cleanly attributed:

| quantity (median) | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| direct-composition brake undershoot | +0.140 | +0.090 | +0.151 |
| box-route undershoot (old Route A) | +0.148 | +0.095 | +0.158 |
| CROWN gap on `V_θ` over the cell | +0.120 | +0.096 | +0.121 |

Two facts decide it: (1) **undershoot ≈ the per-cell CROWN gap on `V_θ`**, so the
binding term is `V_θ`'s own per-cell variation — the *output* side of CROWN — not
the input successor box; direct composition removes only the input-box slack
(the ≈0.006–0.008 box→direct tightening). (2) The sub-level closure is a
**knife-edge**: the analytic contraction is *exactly* 0, so any strictly-positive
black-box undershoot fails. Resolution sweep (seed 0): undershoot
**0.241 (0.125 m) → 0.177 (0.094 m) → 0.140 (0.075 m)** — monotone in cell size
but asymptotic to 0, never reaching it.

**Consequence for the next primitives.** Because the binding term is the
*output-side* CROWN gap (`V_θ`'s per-cell variation), **P3 (zonotope/polytope)
attacks the wrong slack** (it tightens the input set, ≈0.006 here) and will not
close the sub-level either. The live candidate is **P4** — a contraction/Lyapunov
or finite-horizon recursive-feasibility certificate that **drops the
`successor ⊆ level` requirement**, removing the exactly-zero-contraction
knife-edge by construction. (Driver `run_cert_p1.py`; report
`results/f1tenth_e2/p1_report.json`.)

## Step-0 exact ceiling probe (`run_probe_ceiling.py`, 2026-06-11) — "can run" ⇏ "C2 holds"

"The filter runs" only certifies the *visited* states and, more precisely, only
the predicate the filter *tests*. The verifier-independent ceiling is the **exact
pointwise margin on the whole `{Vθ≈0}` in-set boundary shell** (`Vθ(x)∈[0,δ]`,
δ=0.1, 400k samples, no bounds, worst-`d` — exact for the heading-free `V`).
Result over **3 seeds**:

| true pointwise margin (worst over seeds) | min | %<0 of shell |
|---|---:|---:|
| **C3** witness feasibility `min_d Qθ(x,π♭,d) − γVθ` (filter's predicate) | **+0.052** | **0.0%** |
| **C2** decrease `min_d Vθ(f(x,π♭,d)) − γVθ`, γ=0.5 (deployed) | **−0.115** | 1.3–5.9% |
| **C2** decrease, γ=1 (hard invariance) | −0.118 | 6–16% |

**Conclusion: the object can run *because C3 is pointwise clean* (≥+0.05 with
margin ≈ the trained `m`), but its C2 is not.** There exist in-set boundary states
(`Vθ≥0`) whose worst-case braking successor has `Vθ(f) ≤ −0.065 < 0` — a **true,
exact, object-level C2 violation**, not verifier looseness: the trained `{Vθ≥0}`
is *not* robustly forward-invariant under the witness. The gap between clean C3
and holed C2 is exactly the firewall-forbidden `Qθ ≈ Vθ∘f`: the filter trusts the
`Q` predicate while the true value `Vθ∘f` dips; the rollout stays safe only
because the discount + finite horizon + not-visiting-the-pockets mask the dips.

**Verdict (per the dichotomy): negative pockets ⇒ NOT (only) a verifier problem.**
And the fix is *not* "raise the Q-hinge `m`" (already filled, +0.05–0.10) — it is
to add a **direct C2-decrease loss** `[γVθ(x) − min_d Vθ(f(x,π♭,d))]₊` (penalise
the true `Vθ∘f` decrease at worst-`d`) or **CEGIS on the C2-violation set**, same
object family, same plant. Re-probe; once the C2 ceiling is ≥0 on the band, the
residual becomes a *pure* verification problem and the tight P1 primitive (already
validated on the ideal at 86,580) / P4 applies. (Report `ceiling_probe.json`.)

The headline experiment of the project. E2 turned Gate D green with the **analytic**
braking‑distance CBF. E2‑L asks the actual project claim: can the **deployed
learned** trio `V_θ, Q_θ, π♭_φ` be Gate‑D‑certified **directly** — Theorem A's
(C1)–(C3) discharged by CROWN on the *networks themselves*, with **no
approximation assumption** (`V_θ ≈ V_target` never enters the certificate)?

**Result: a sound, attributable FAIL by both available sound routes.** Treating
PASS and FAIL as equally valid, this is the strong negative that *characterises
when a learned barrier is certifiable* — and it respects the soundness firewall
(no faked PASS, no fallback to the analytic set).

## Object construction (`distill.py`) — training is for non‑vacuity only

The analytic CBF is distilled into ReLU MLPs (small, weight‑decayed for
CROWN‑tightness): `V_θ(px,py,v)` 3→32→32→1, `Q_θ(x,u,d)` 8→64→64→1,
`π♭_φ(x)` 4→32→32→2. Targets: `V_target = clearance − D(v) − m`,
`Q_target = V_target∘f`, `π♭_target =` the braking witness. The contraction
margin `m` is the decrease‑hinge margin (`min_d Q_θ(x,π♭,d) ≥ γV_θ + m`), swept
`m ∈ {0, 0.05, 0.10, 0.15}` × 3 seeds. The trained object is faithful
(`MSE(V_θ,V_target) ≈ 0.005`, brake‑agreement ≈ 1.0), then **frozen**; the
certificate is checked on the frozen nets with the true `f`.

## Two sound routes, both verified directly on the networks

**Route A — heading‑free braking‑invariant sub‑level `{V_θ ≥ c}`** (the learned
analogue of the analytic E2). A level `c` is *valid* iff the whole sub‑level
`{ub V_θ ≥ c}` is (C1) collision‑free and (C2) **braking‑closed**: the CROWN
lower bound of `V_θ` over the worst‑heading brake successor is `≥ c`. The
runtime filter races only when a sound check confirms `V_θ(x⁺) ≥ c`, else brakes.

**Route B — heading‑inclusive 4‑D cell‑reachability** (`(px,py,ψ,v)` Tarski
greatest fixed point under the brake map): the only route that can yield a
heading *tube*. Run on **both** the learned candidate `{lb V_θ ≥ c}` **and the
analytic ideal candidate** (the exact box bounds of `V_target`), to separate the
learned object from the lattice geometry.

`Q_θ` is exercised by **C3** (witness feasibility, CROWN on the composed
`Q_θ(x,π♭(x),d) − γV_θ`), reported as a **liveness** property (the brake
fallback is always available, so C3 does not gate safety).

## Result (`python experiments/f1tenth_e2/run_cert_learned.py`)

```
margin m | pass | A closes c | B cert(learn) | B cert(an.) | undershoot | C3 hole
   0.00  |   0% |    none    |       0       |      0      |   +0.15…   |  …
   0.05  |   0% |    none    |       0       |      0      |   +0.15…   |  …
   0.10  |   0% |    none    |       0       |      0      |   +0.15…   |  …
   0.15  |   0% |    none    |       0       |      0      |   +0.15…   |  …
```
(Exact numbers in `results/f1tenth_e2/e2_learned_report.json`; figures in
`results/f1tenth_e2/figures/`.)

## Attribution — *why* it fails (the publishable content)

**Route A is a knife‑edge the learned object cannot stand on.** `clearance(p)` is
a distance function (`‖∇‖ = 1` a.e.), so the worst‑heading clearance loss over a
step is **exactly** `dt·v`, which the discrete braking distance recovers exactly
— the analytic CBF has **exactly‑zero** worst‑case braking contraction and closes
A by this *structural 1‑Lipschitz cancellation* (`verify_braking_decrease`). A
black‑box `V_θ` must instead be **CROWN‑bounded over finite cells**, incurring a
brake‑successor undershoot `ub V_θ − lb V_θ(x⁺) ≈` the **true per‑cell variation
of `V`** (≈ 0.15 at a 0.14 m cell; *not* fit error — `MSE(V_θ) ≈ 0.005`). This
is **strictly positive** and exceeds the zero contraction, so **no level closes**.
It shrinks with finer cells (fig 1) but never reaches 0; and the margin `m` only
moves **C3 / liveness**, never the C2 boundary (fig 3) — m cannot add C2 slack
without assuming `Q_θ ≈ V_θ∘f` (forbidden).

**Route B is independently obstructed by cell+box geometry.** The 4‑D
reachability **erodes to empty even for the analytic IDEAL `V`** (fig 2: certified
cells collapse to 0 in ~7 GFP iterations). The outward‑rounded brake‑successor box
always overlaps neighbour cells, so "successor ⊆ accepted" forces a boundary‑free
set — the same cell+box obstruction that blocks fixed‑speed Gate D
(`docs/dubins_e0_results_and_ablation.md`). Because it kills the *analytic* ideal
too, the empty learned route‑B is **not** a learned‑object limitation.

## The one‑line finding

> Under the **outward-rounded cell+box one-step reachability primitive**, neither
> sound route closes on the learned trio. But that primitive **also erodes the
> analytic ideal `V` to empty** (Route B), so its failure is a property of **the
> primitive**, not of the learned object — it cannot establish that the learned
> barrier is uncertifiable. The brake-only sub-level (Route A) does fail on a
> genuine structural knife-edge the analytic CBF stands on via exact 1-Lipschitz
> cancellation, but this says nothing about a full **direct-composition** C2 over
> the deployed `Φ_rob` filter. **The open question** — and the next experiment — is
> whether a tighter sound primitive (direct-composition CROWN with no intermediate
> successor box (**P1**), zonotope/polytope reachable sets, or a
> contraction/Lyapunov argument) certifies the frozen learned trio non-vacuously.
> The iron rule: each new primitive is first sanity-checked on the analytic ideal
> `V` (which **must** pass, ~86,580 cells); only then is the learned object
> certified, so any residual failure is cleanly attributable to learned-vs-ideal.
