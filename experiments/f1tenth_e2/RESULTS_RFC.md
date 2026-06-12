# E2-L / RFC — Post-hoc certification of a deployed learned robust Q-CBF filter

**Finished evidence block for the toy/example section.** Brakeable F1TENTH model;
the learned object `(V_θ, Q_θ, π♭)` is a plain ReLU MLP tuple — **no analytic
braking-distance `D(v)` in the object, no hard-Lipschitz / spectral-norm
structure.** Training is only a non-vacuity aid; every safety claim below is
discharged **post hoc** by the verifier on the frozen networks with the true
dynamics `f` and the full disturbance set `D`. Driver:
[`run_cert_rfc.py`](run_cert_rfc.py).

## Main claim

> We post-hoc certify a deployed learned robust Q-CBF filter on the brakeable
> F1TENTH model. The verifier constructs a **fallback-pinned invariant subset**
> `S_brake^Q ⊆ {V_θ≥0} ∩ K`, verifies full-`D` safety along the brake-to-stop
> trajectory, and verifies the fallback action is **live under the learned `Q_θ`
> predicate**. The certified set covers **≈ 0.80 of the analytic-reference safe
> volume** with **zero certified-but-violated** audit failures. In contrast, the
> **naive racing deployment** using the baseline `Q_θ` predicate is **rejected by
> the same verifier**: it admits racing actions whose successors leave the
> certified backbone and collide under adversarial `d` — a **false-feasibility**
> diagnostic (Theorem S) showing why post-hoc verification is necessary.

## 1. Set names (used consistently)

| symbol | definition | role |
|---|---|---|
| `Ω*` | analytic reference safe set `{V_analytic ≥ 0}` (86,580 cells @ 80³) | ρ denominator, iron-rule oracle — **not** certified |
| `S_brake` | `{x : ∀t≤H_stop, lbV_θ(x_t) ≥ 0 ∧ lb g(x_t) ≥ 0}` along the brake-to-stop trajectory | `V_θ`-based fallback safety backbone (verified) |
| `S_brake^Q` | `S_brake ∩ {x : min_d Q_θ(x,π♭,d) ≥ γV_θ(x)+ε}` | **main positive Q-CBF certificate** (verified, Q-live) |
| `S_race,naive` | naive racing closure of `S_brake` under the baseline `Q_θ` predicate | **apparent only — rejected by audit** |
| `S_Q^∞` | gfp of `S∩K∩Feas_Q∩Pre_Φ(S)` | reserved for a future *certified* racing fixed point (not achieved) |

## 2. Result table (baseline net, MSE(V)=0.012, MSE(Q)=0.039; seed 0)

| Filter / Set | Runtime rule | Verified? | Coverage | Audit (cbv) | Interpretation |
|---|---|---|---|---|---|
| `S_brake` | brake-to-stop fallback | **yes** | ρ_brake ≈ **0.80–0.81** | **0** | verified fallback backbone |
| `S_brake^Q` | `Q_θ`-live fallback-pinned | **yes** | ρ_brake^Q ≈ **0.78–0.80** | **0** | **main positive Q-CBF certificate** |
| `S_race,naive` | race iff `Q_θ`-feasible | **no** | apparent 0.635 | **205** | false-feasibility diagnostic (Thm S) |
| conservative `Q_θ` (c2fix) | race iff conservative `Q_θ` | **no** | — | 3144 (worse) | repair did **not** close (honest ablation) |

`ρ_brake^Q = ρ_brake · live_frac`, `live_frac = |S_brake^Q|/|S_brake|`.

## 3. Why naive sub-level certification fails (exact probe)

Exact pointwise probe (no bounds), worst `d`, witness brake `u=(a_min, steer)`:

```
h1(x) = V_θ(f(x,u,d)) − V_θ(x)        (γ=1 hard non-decrease)
min h1 ≈ −0.08 … −0.10,  Pr[h1<0] ≈ 15%  at EVERY level c ∈ [0.05, 0.30]
```

The holes are **level-independent** (object property, not verifier looseness;
confirmed on the well-trained net and after c2_fix repair). Therefore **no
`{V_θ ≥ c}` is a robust one-step invariant set** for any `c` → Gate 0b. This
justifies the maximal-invariant-**subset** certificate instead of a sub-level set.

## 4. Certified fallback-pinned Q-CBF backbone (the positive result)

A cell joins `S_brake` iff, over its heading cell and worst `d`, the **whole
braking trajectory** (a=a_min, ~`H_stop`≈18 steps to v=0, a fixed point) keeps
`lbV_θ ≥ 0 ∧ lb g ≥ 0` at every step. This is a finite trajectory composition
(CROWN per step), **not** a set-closure GFP → no carving erosion; only the
heading interval widens (bounded, since braking kills v fast). `v=0` is a fixed
point ⇒ infinite-horizon safety directly.

| res | `|S_brake|` | `Ω*` cells | ρ_brake | C3 live_frac | ρ_brake^Q | audit cbv |
|---|---|---|---|---|---|---|
| 44×44×33 | 10,624 | 13,252 | 0.802 | 0.971 | 0.779 | 0 |
| 56×56×42 | 23,011 | 28,440 | 0.809 | 0.992 | 0.803 | 0 |

**Resolution-stable around 0.80; zero certified-but-violated** under extremal +
greedy-`d` rollouts (min g +0.09…+0.14, min V_θ +0.01…+0.03). This is the main
positive Q-CBF certificate: the deployed fallback-pinned filter (V_θ-defined
invariant set, Q_θ-live witness) is sound on ≈80% of the analytic safe volume.

## 5. Naive racing filter fails post-hoc certification (Theorem-S diagnostic)

Deployed predicate `Φ_θ(x) = {u : min_d Q_θ(x,u,d) ≥ γV_θ(x)+ε}`. The naive
min-intervention filter races `u_race=(a_max, racing steer)` iff `Φ_θ`, else
brakes. **One-pass soundness fact:** `race-succ ∈ S_brake` proves `S_brake` is
deployed-invariant **iff it holds on all of `S_brake`** (then multi-step racing
is covered by induction). Baseline `Q_θ`:

- closure **fails on 2,209 / 10,624** `S_brake` cells → carved set is **not** a
  proven invariant (multi-step racing escapes);
- **audit falsifies the carved set: cbv = 205, min g = −0.30, raced 24%.**

So `u ∈ Φ_θ(x) ⇏ f(x,u,d) ∈ S_brake^Q`: the learned `Q_θ` predicate is
**over-permissive**. The verifier correctly **rejects** the naive racing
deployment. `ρ_naive = 0.635` is **apparent only**, never reported as certified.

## 6. Optional repair: conservative `Q_θ` (ablation — did not close)

The `c2_fix` distillation (one-sided `Q_θ ≤ V_θ∘f` conservatism + C2 hinge +
non-vacuity anchors) was tested as the certification-aware repair. **It did not
close certification:** the racing audit got *worse* (cbv = 3,144, min g = −1.36),
and the C2-hinge/anchors distorted `V_θ` (MSE 0.09, inflating `{V_θ≥0}` past the
analytic set, ρ_brake → a meaningless 1.07). Honest framing:

> Conservative training as bundled in `c2_fix` did not reduce racing
> false-feasibility on this model; a cleaner one-sided `Q_θ ≤ lbV_θ(f)+η`
> objective (without the V-distorting anchors) is the next repair to try. The
> conservative loss is **not a proof assumption** — only the post-hoc verifier
> proves safety; this is an ablation, not the headline.

## 7. What this does / does not claim

- ✅ `S_brake^Q` (fallback-pinned, Q-live) is **certified** invariant + safe,
  ρ ≈ 0.80, cbv = 0, resolution-stable.
- ✅ The naive racing `Q_θ` filter is **rejected** by the same verifier
  (false-feasibility, audit-confirmed) — the Theorem-S necessity argument.
- ❌ `{V_θ ≥ 0}` is **not** certified invariant (level-independent holes).
- ❌ `ρ_naive = 0.635` is **not** certified (apparent closure only).
- ❌ The current positive result does **not** certify the full racing filter.
- ❌ The ρ ≈ 0.80 comes from the **brake-to-stop funnel**, *not* from relational
  cancellation (the relational primitives are built + sound but target the dead
  sub-level route; they are for a future tighter/racing certificate).
- ❌ Training losses do **not** prove safety — only the verifier does.

## 8. Reproduce

```
python experiments/f1tenth_e2/run_cert_rfc.py --res 44        # baseline: S_brake, S_brake^Q, naive racing, audit
python experiments/f1tenth_e2/run_cert_rfc.py --res 56        # resolution confirmation
python experiments/f1tenth_e2/run_cert_rfc.py --res 44 --c2fix# conservative-Q ablation
python experiments/f1tenth_e2/cex_and_figs.py                 # false-feasibility bank + figures
python -m tests.test_direct                                   # primitive soundness (D1–D4)
```

Reports: `results/f1tenth_e2/rfc_phase1_report.json`,
`rfc_phase2_report.json`, `rfc_res56.log`, `rfc_phase2_c2fix.log`,
`cex_bank.npz`, `figs/`.
