# Technical Report — Post-hoc certification of a deployed learned robust Q-CBF filter
### Full experimental journey: from ρ=0 direct-verification failures, through a rejected "near-cheat" ρ≈0.45, to the claim-aligned ρ≈0.80 certificate

Platform: brakeable F1TENTH 4-state bicycle. Object: the **deployed learned trio
`(V_θ, Q_θ, π♭)`**. Verifier: pure-NumPy IBP + CROWN, full disturbance set `D`,
true dynamics `f`. This report records the **whole arc** — every dead end and what
it taught — so the result is reproducible and the design choices are auditable.

> **Soundness firewall (held throughout).** Training is for non-vacuity only.
> Every safety claim is discharged post hoc on the *frozen* networks. No
> `V_θ≈V*`/`‖Q_θ−Q*‖` assumption ever enters a proof. `unknown ⇒ not certified`.

---

## 0. Problem statement and the core claim

Robust state-action Q-CBF synthesis yields an ideal pair `(V*, Q*)`, but the
object that *runs* is a finite learned network tuple `(V_θ, Q_θ, π♭)`. The project
claim is to **certify that deployed artifact directly**:

```
verify (C1) g≥0, (C2) successor stays in the certified set, (C3) witness feasible
  on the frozen (V_θ,Q_θ,π♭) over full D  ⇒  Theorem A: recursive feasibility,
  robust forward invariance, safety — with NO approximation hypothesis.
```

The honest tension that drives this whole report: a learned `V_θ` distilled from a
**zero-margin** (knife-edge) barrier value does not exactly satisfy the one-step
conditions — it *wobbles*. The journey is about finding a certificate form that is
(a) sound, (b) non-vacuous, and (c) faithful to the deployed object — without
mutilating the object to make verification easy.

---

## 1. Baselines that worked (the references)

| result | status | notes |
|---|---|---|
| Fixed-speed Gate D | **blocked (characterized)** | robust-invariant sets are thin constant-`V` orbits; the interval one-step box cannot close them — a real obstruction, not a bug |
| Brakeable plant + **analytic** CBF `V=clearance−D(v)−m` | **Gate D PASS** | 86,580 cells @ 80³, exact braking decrease `−3e-16≈0`; this is the **Ω\*** reference (ρ denominator), never itself the certified learned object |

The analytic CBF closes because braking has an **exact** 1-Lipschitz cancellation:
clearance lost per step `≤ dt·v` is exactly cancelled by braking distance
recovered `D(v)−D(v⁺)=dt·v`. The entire difficulty below is that a **black-box
`V_θ` cannot reproduce this exact zero-margin cancellation.**

---

## 2. Direct certification of the learned object — the ρ=0 failures

All attempts here target the natural sub-level set `{V_θ ≥ c}` and ask the verifier
to prove it one-step robustly invariant. **All return ρ=0**, and the *reasons* are
the valuable part.

### 2.1 Route A — heading-free braking sub-level `{V_θ ≥ c}`
CROWN-bound `V_θ` over the worst-heading braking successor box; accept `c` if the
whole sub-level is collision-free and `lbV_θ(successor) ≥ c`.

- **Result: never closes (ρ=0).** The brake-successor undershoot
  `ubV_θ − lbV_θ(x⁺) ≈` the **per-cell CROWN gap on `V_θ` ≈ the true per-cell
  variation of `V`** (≈0.15 at 56³, ∝ cell size; *not* fit error — MSE(V_θ)=0.005).
- This strictly-positive gap exceeds the **exactly-zero** analytic contraction. The
  margin `m` (finetune decrease hinge) only tightens C3/liveness; it cannot add C2
  contraction slack without assuming `Q≈V∘f` (forbidden).

### 2.2 Route B — heading-inclusive 4-D cell-reachability GFP
Tarski greatest fixed point under the brake map (the only route to a heading-tube).

- **Result: erodes to empty (ρ=0) — even for the ANALYTIC ideal `V`** (synthetic
  test: 116,784 → 0 in 7 iterations).
- **Diagnosis:** the obstruction is the **outward-rounded cell+box successor
  overlap**, *not* the learned net: every successor box straddles neighbour cells,
  so "successor ⊆ accepted" forces a boundary-free (hence empty) set. This is the
  same cell+box obstruction as fixed-speed Gate D.

### 2.3 P1 — direct-composition bound (remove the intermediate box)
Bound `V_θ(f_brake)` by minimising the CROWN affine lower **functional** over the
*true* successor (no outward-rounded box).

- **Iron rule PASSES:** analytic ideal certifies 86,580 cells (min decrease
  `−3e-16`) — refutes "even the ideal fails" (that was Route-B cell-rounding).
- **Learned: still FAIL (ρ=0).** Undershoot median over 3 seeds `{0.140,0.090,0.151}`
  vs the box route `{0.148,0.095,0.158}` — direct composition tightens only
  **0.006–0.008**, because the binding slack is `V_θ`'s **output-side per-cell
  variation** (median `{0.120,0.096,0.121}`), not the input box.
- **Implication:** zonotope/P3 (attacking the input slack) won't help; the
  remaining slack is intrinsic to the black-box value's per-cell oscillation.

### 2.4 The exact ceiling probe — separating object from verifier
Zero-relaxation pointwise probe (no CROWN, exact `f`, worst `d`) on the boundary
shell `V_θ∈[0,0.1]`, 400k samples, 3 seeds:

```
C3 witness  min_d Q_θ(x,π♭,d) − γV_θ :  min +0.052,  0% negative   (CLEAN — this is why it runs)
C2 decrease V_θ(f(x,π♭,d)) − γV_θ    :  γ=0.5: min −0.115, 1.3–5.9% neg
                                        γ=1  : min −0.118, 6–16% neg
```

**Decisive:** there exist in-set states (`V_θ≥0`) whose worst-case brake successor
has `V_θ(f) ≤ −0.065 < 0`. So `{V_θ≥0}` is **genuinely not** one-step invariant —
a *true object-level* C2 hole, **not** verifier looseness. The clean-C3 / holed-C2
gap is the firewall-forbidden `Q_θ≈V_θ∘f` surfacing: the filter runs (C3) while the
true successor value dips (C2).

---

## 3. Training-repair attempts — partial, then stuck

If the holes are object-level, repair the object (still post-hoc verified). These
are **claim-safe training pressures** (no proof assumption), and they **plateau**.

| attempt | mechanism | result |
|---|---|---|
| **C2-fix hinge** (STEP-1) | `relu(η+γV_θ(x)−min_d V_θ(f(x,π♭,d)))` + one-sided Q-conservatism + anchors | lifts worst-case ~30% (γ=0.5: −0.115→−0.077) but **does not close** |
| **naive CEGIS** (STEP-1) | retrain on counterexamples + jitter | **does not converge** — 4 rounds oscillate −0.074/−0.112/−0.095/−0.116; retrained `V` chases its own moving boundary |
| **inner-level sweep** (STEP-2) | certify `{V_θ≥c}`, c>0 | **REFUTED** — `min h_Bc` is **level-independent** (−0.097 at c=0 → −0.075 at c=0.30), frac<0 ≈ 5% at *every* level |
| **stabilized tail-CEGIS** (STEP-3) | fixed level + trust region + persistent replay | **diverges** −0.070→−0.138→−0.139 |
| **pairwise temporal regression** (STEP-3) | match `V_θ(f)−V_θ(x)` to analytic Δ | **plateaus −0.076** — a *representation limit* of the small MLP (pair-MSE 0.0009 but worst-case unmoved) |

**The binding quantity is the network's one-step error variation
`ε(f)−ε(x) ≈ 0.075–0.10`, which is level-independent.** A black-box approximation
of a *zero-margin* value wobbles ±~0.08 around the exact-zero contraction → ~5–15%
holes that no amount of nudging removes. Sub-level certification of the black-box
is, on this evidence, **impossible at achievable fidelity**.

---

## 4. The "near-cheat" that barely worked — ρ≈0.45, and why it was REJECTED

To force a closure, the value's **representation** was changed (STEP-3 → STEP-7):

```
STRUCTURED  V_θ(p,v) = C_θ(p) − D(v) − m      (D(v) = the ANALYTIC braking distance)
            +  HARD 1-Lipschitz C_θ            (spectral-norm-projected ReLU MLP)
```

This works *mechanically*: replacing the speed axis with the analytic `D(v)` makes
the speed contraction exact, and a hard 1-Lipschitz `C_θ` reproduces the position
cancellation. The arc:

```
black-box        : worst-case contraction floor ≈ 0.076   → ρ=0
structured V (soft Lip): min −0.011, frac<0 0.09%         → ρ=0 (verifier gap)
structured V + HARD 1-Lip C_θ: directional ratio 1.79→0.49, deficit GONE → ρ≈0.45 banked @ c=0.10 (64³)
```

**This was REJECTED as off-claim (STEP-8).** Three independent violations, each
pinned to code (`distill.py`, `run_cert_structured.py`):

1. **Model baked into the OBJECT, not just the verifier.** `D(v)` is the analytic
   dynamics formula; with `V_θ=C_θ(p)−D(v)`, the speed-axis safety is *copied in*,
   not learned ⇒ circular (use the model to design the object, then certify the
   object respects the model). *Legal:* `f` inside the C2 verifier. *Illegal:* `f`
   inside the object's functional form — the exact line between this and the claim.
2. **`Q_θ` is bypassed.** The structured C2 closes on `V_θ(f)` directly and never
   uses `Q_θ`; the state-action Q-CBF — the paper's whole novelty — is not what is
   certified.
3. **Hard 1-Lipschitz has no learning necessity.** It exists solely to reproduce
   the analytic cancellation; it underfits the value (MSE(clearance)≈0.11, shrinking
   the set) and cripples expressivity to suit the verifier. *Operational test:*
   remove `D(v)` and the Lipschitz clamp ⇒ the safety collapses ⇒ the safety came
   from injection, not the learned object.

> **Verdict:** ρ≈0.45 is *not* a certificate of the deployed learned Q-CBF filter;
> it certifies a hand-designed analytic CBF plus a thin learned residual. Good
> number, wrong object. Kept only as a clearly-labelled **analytic-CBF upper-bound
> baseline**, physically separated from any headline.

---

## 5. The claim-aligned certificate — ρ≈0.80 (the result)

**Pivot (RFC):** stop trying to prove `{V_θ≥c}` is one-step invariant (it is not —
§2.4). Certify the **maximal brake-invariant subset** of `{V_θ≥0}` via a finite
**brake-to-stop trajectory bound**, keeping the object a plain ReLU MLP.

### 5.1 Why the subset works where the sub-level cannot
- The γ=1 *non-decrease* holes (needed for a sub-level **core**) are ~15% and
  level-independent → a hole-free core does **not** exist (re-confirmed on the
  well-trained net *and* after C2-fix repair). **Gate 0b.**
- But the actual **exit** holes (`V_θ(f)<0` from `{V_θ≥0}`, needed for the
  **subset**) are far fewer, and the brakeable plant has a free anchor: stopped
  safe states `{v=0, g≥0}` are **fixed points** of braking.

### 5.2 The certificate (`brake_funnel_cert`)
A cell joins `S_brake` iff, over its heading cell and worst `d`, the **whole
braking trajectory** (`a=a_min`, ~18 steps to `v=0`) keeps `lbV_θ ≥ 0 ∧ lb g ≥ 0`
at every step. This is a **finite trajectory composition** (CROWN per step), **not**
a set-closure GFP → no carving erosion; only the heading interval widens (bounded,
since braking kills `v` fast). `v=0` is a fixed point ⇒ **infinite-horizon** safety.
The Q-CBF tie-in: verify the witness is live, `min_d Q_θ(x,π♭,d) ≥ γV_θ+ε`, giving
`S_brake^Q = S_brake ∩ {Q-live}`.

| res | `|S_brake|` | Ω\* | ρ_brake | C3 live | **ρ_brake^Q** | audit cbv |
|---|---|---|---|---|---|---|
| 44³ | 10,624 | 13,252 | 0.802 | 0.971 | **0.779** | 0 |
| 56³ | 23,011 | 28,440 | 0.809 | 0.992 | **0.803** | 0 |

**Resolution-stable ≈0.80, zero certified-but-violated** (extremal + greedy `d`,
min g +0.09…+0.14). **No `D(v)` in the object, no Lipschitz constraint, model only
in the verifier.** This is the main positive result, and it sits well above the
rejected 0.45 while being fully on-claim. (The relational CROWN primitives
`crown_upper_affine` / `crown_relational_decrease_lb` were built and proven sound,
but they target the dead sub-level route; ρ≈0.80 comes from the brake funnel, not
relational cancellation — stated honestly.)

---

## 6. The deployed *racing* filter — a Theorem-S rejection (not a certificate)

The richer min-intervention filter races `u_race=(a_max, steer)` iff
`Φ_θ(x)=[min_d Q_θ(x,u_race,d)≥γV_θ]`, else brakes. Soundness fact: the one-step
closure `race-succ ∈ S_brake` proves `S_brake` deployed-invariant **iff it holds on
all of `S_brake`** (then multi-step racing is covered by induction).

- **Baseline `Q_θ`:** closure fails on 2,209/10,624 cells → carved set is **not** a
  proven invariant; **audit collides — cbv=205, min g −0.30, raced 24%.**
- **Counterexample bank (Experiment C):** N=540 states where `Φ_θ` holds (margin as
  thin as +0.0002) but the racing successor is not brake-safe; 75% of in-set states
  are racing-feasible ⇒ `Q_θ` is broadly **over-permissive**.
- **Conservative-`Q_θ` repair (Experiment D, ablation):** the `c2_fix` bundle made
  racing **worse** (cbv 205→3144, min g −1.36) and distorted `V_θ` (MSE 0.012→0.09).
  Honest negative — a cleaner one-sided `Q_θ ≤ lbV_θ(f)+η` *without* the
  V-distorting anchors is the next repair to try.

This is exactly the framework's **Theorem S / false-feasibility** result: the
learned `Q_θ` predicate admits unsafe racing, so the verifier **correctly rejects**
the naive racing deployment — *demonstrating why post-hoc verification is
necessary* rather than trusting the learned predicate.

---

## 7. What each failure taught (design principles, for the record)

1. **Sub-level vs subset.** A *zero-margin* learned barrier has level-independent
   non-decrease holes; no `{V_θ≥c}` is invariant. The correct object is the maximal
   **subset**, anchored by a structural fixed point (here, stopped states).
2. **Exact probe before every expensive run.** It repeatedly converted
   "compute/verifier problem" into "object problem" (and back). It is what proved
   the sub-level route dead *cheaply*, and what saved the verifier runs.
3. **Tighten the certificate form, never the object.** P1/relational tighten the
   *verifier*; structured-`D(v)`/hard-Lipschitz tighten the *object* and break the
   claim. The legal line: model in the C2 verifier = fine; model in the object's
   functional form = forbidden.
4. **A clean audit is the soundness backstop.** It falsified the racing cert
   (cbv=205) that looked fine on paper — the carved-set multi-step gap.
5. **Negative results are results.** "Sub-level impossible (object holes)" and
   "racing `Q_θ` is over-permissive (Theorem S)" are publishable characterizations,
   not dead ends.

---

## 8. Final status and artifacts

**Headline:** post-hoc verification **accepts** a large fallback-pinned learned
Q-CBF safe set (`S_brake^Q`, ρ≈0.80 of analytic safe volume, cbv=0,
resolution-stable, Q-live 0.97–0.99) and **rejects** the naive racing `Q_θ` filter
(false feasibility, audit-confirmed) — Theorem A and Theorem S on the *deployed
learned object*, with no `D(v)` and no Lipschitz in the object.

| artifact | path |
|---|---|
| Final clean result | `RESULTS_RFC.md` |
| This full journey | `TECHNICAL_REPORT.md` |
| Figures | `figures/{safe_slices, false_feasibility, resolution_and_audit}.png` |
| Data | `data/00_probe* 01_brake_funnel* 02_deployed_racing* 03_false_feasibility*` |
| Logs | `logs/run_{res44_full, res56_confirm, c2fix_ablation, counterexamples_figures}.log` |
| Code | `code/{run_cert_rfc, cex_and_figs, distill}.py` |

**Open next steps:** (i) clean one-sided Q-conservatism (no V-distortion) to try to
certify the racing filter; (ii) a true `S_Q^∞` recursive-feasibility fixed point if
the predicate becomes sound; (iii) the relational bound for a tighter / higher-`v`
certificate. None require re-introducing `D(v)` or hard Lipschitz.

---

### Appendix — number provenance
Fixed-speed/analytic: `project_gate_d_resolution`. Direct-failure arc (Routes A/B,
P1, probe, C2-fix, level sweep, structured, hard-Lip, ρ=0.45 + rejection): memory
`project_e2_learned_certification` STEP-0…STEP-8 and the original drivers
(`run_cert_learned.py`, `run_cert_p1.py`, `run_probe_ceiling.py`,
`run_level_sweep.py`, `run_cert_structured.py`, `run_cert_rho.py`). RFC result
(STEP-9…11): `run_cert_rfc.py`, `cex_and_figs.py`, this folder's data/logs.
