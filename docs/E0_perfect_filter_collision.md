# Dubins-E0 — Does the *perfect* (ground-truth HJ) filter ever collide?

*Fixed-speed Dubins car. The ground-truth value `V_HJ` / action value `Q_HJ` are
**grid tables from value iteration — NOT neural networks**, so there is no CROWN
and no fitting error anywhere in this study. We deploy the best HJ-greedy filter
extractable from `V_HJ` and measure its **realized** long-horizon collision rate
under disturbance. Script: `experiments/dubins_e0/run_perfect_filter_collision.py`.
Artifact: `results/.../perfect_filter_collision.json`.*

## 0. Motivation and setup

The cell-worst certificate does not pass at 100% even on the ground-truth value
(`V` cell-pass ~0.7, `Q` ~0.9). One might assume the residual is pure **verifier
conservativeness** over a truly-safe object. This experiment tests that assumption
by measuring whether the perfect filter actually stays safe.

- **Value object.** `V_HJ` = discounted HJ safety value (Fisac/Akametalu backup,
  `V=(1−γ)g+γ·min(g, max_u min_d V(f))`), `γ_teach=0.92`, grid `51×51×48`.
- **Filter (re-evaluated at the continuous state each step).**
  `u*(x)=argmax_u min_d V_HJ(f(x,u,d))`, `V_HJ` trilinearly interpolated, fine
  matched `u`/`d` grids. This is the best filter the GT value defines.
- **Reference set / denominator.** `Ω*={V_HJ≥0}` (`Vol(Ω*)/Vol(domain)=0.439`);
  starts = all `40³` lattice centres in `Ω*∩K` = **28 080 cells**.
- **Disturbance regimes.** `d=0`; constant `±0.3`; random `U[−.3,.3]` (3 seeds);
  and the **matched value-greedy worst-case adversary** `d*=argmin_d V_HJ(f(x,u*,d))`
  (exactly the min-player of the `max_u min_d` game the value was solved for).
- Horizon `H=150` steps (=15 s; all regimes converge by `H≈50`).

---

## 1. Headline: the perfect filter collides — a lot, and broadly under disturbance

**Converged collision probability (% of the 28 080 Ω* starts, H=150):**

| regime | collision % | character |
|--------|------------:|-----------|
| `d=0` nominal              | **14.76** | thin-shell only (deep interior safe) |
| `d=+0.3` constant          | **34.15** | — |
| `d=−0.3` constant          | **34.13** | symmetric with +0.3 |
| random `U[−.3,.3]`         | **15.20** | seeds 15.21 / 15.38 / 15.01 (tight) |
| **matched worst-case adv** | **65.71** | **broad** — even deep interior fails |

Collision fraction vs horizon (converges fast, then flat):

| regime |  H=5 | H=10 | H=20 | H=50 | H=150 |
|--------|-----:|-----:|-----:|-----:|------:|
| d0     | 1.05 | 4.79 | 13.71| 14.76| 14.76 |
| ±0.3   | 1.08 | 6.10 | 16.6 | 34.1 | 34.1  |
| adv    | 1.45 | 9.57 | 28.36| 64.82| 65.71 |

Two immediate facts:
1. **`Ω*={V_HJ≥0}` is NOT a safe invariant set** even for the optimal filter:
   ~15% leaks at `d=0`, two-thirds under the matched adversary.
2. **The perfect filter is `±d`-symmetric** (`+0.3`≡`−0.3`≡34.1%). The strong
   left/right asymmetry seen earlier for the *learned* witness was a property of
   that chiral learned policy, not of the problem — the GT filter has none.

**This is not a deployment/grid artifact.** Re-running the `d=0` sim with deploy
grids `7×7`, `17×7` (=solve grids), `21×21`, `41×41` gives **14.5 / 14.7 / 14.6 /
14.6%** — invariant to control/disturbance resolution. The leak is a property of
the value `V_HJ` itself.

---

## 2. Mechanism: the discounted value **strictly decays** under the matched adversary

At the discounted fixed point, write `F=max_u min_d V(f)`. Wherever the future
binds (`F<g`, i.e. the safe interior away from `V=g`), `V=(1−γ)g+γF`, so along the
**matched greedy saddle** (`u*=argmax_u min_d V(f)`, `d*=argmin_d V(f(x,u*,·))`):

```
V(f(x,u*,d*)) = F = (V − (1−γ)g)/γ
ΔV = V(f) − V = (1−γ)(V − g)/γ  <  0      (since V < g on the interior)
```

So **under the worst-case disturbance the value decreases every step**, by
`(1−γ)/γ·(g−V) ≈ 0.087·(g−V)`. The apparent "margin" `V>0` is robustness against
the **nominal** successor, *not* against the adversary; the adversary drains it at
a fixed rate until `V` crosses 0 and collision follows. This holds for **any**
starting value — higher `V` only delays it.

**Numerical confirmation** (matched-adversary trace from the deepest interior
state, `V₀=1.23`): `V` decreases monotonically every step, and the observed `ΔV`
tracks the predicted `(1−γ)(V−g)/γ` (both ≈ −0.012…−0.02 / step). The deepest,
"safest" state is being bled out.

**Consequence — no level set is robustly invariant.** A `V`-threshold sweep
(collision % of `{V_HJ≥τ}` starts; `ρ=Vol({V≥τ})/Vol(Ω*)`):

| τ    |  ρ   | d=0 coll% | **adv coll%** | rand coll% |
|------|-----:|----------:|--------------:|-----------:|
| 0.00 |1.000 |   14.76   |   **65.71**   |   15.21    |
| 0.10 |0.883 |   10.19   |   **63.54**   |   10.61    |
| 0.20 |0.748 |    3.90   |   **59.94**   |    4.16    |
| 0.30 |0.641 |    0.53   |   **56.37**   |    0.89    |
| 0.50 |0.471 |    0.00   |   **49.13**   |    0.00    |

Raising `τ` **does** recover a nominal/random-safe invariant set
(`{V≥0.3}`: 0.5% at d=0, still 64% of Ω*; `{V≥0.5}`: 0%). But against the matched
adversary, **even `{V≥0.5}` (top 47% of Ω*) still collides 49%** — exactly as the
decay law predicts (a fixed `τ` cannot stop a per-step drain over a long horizon).

**Where it fails.** First-violation type is overwhelmingly the **world boundary**,
not the obstacle (adv: 17 564 world vs 888 obstacle): with world radius 1.8 and a
robust min turn radius 1.43, the adversary nudges the heading until the car can no
longer turn back inside and it **spirals out**. Adversary violation depth is large
(median min-g −2.24), and time-to-collision is gradual (median 24 steps), i.e. a
slow bleed, not an instantaneous boundary leak.

---

## 3. Implications for the certificate and the project

1. **The cert's <100% on the GT is partly REAL, not just slack.** Some of the cells
   the verifier refuses to certify on `V_HJ` are genuinely unsafe (the value
   over-approximates the safe set on the thin `V≈0` shell at d=0). The verifier is
   *correctly* flagging them; it is not purely conservative.

2. **The root cause that the invariance certificate cannot close is the teacher
   itself.** `V_HJ` is a **discounted** (soft) safety value; its superlevel sets are
   **not robustly forward-invariant** (§2). Distilling it into `V_θ/Q_θ/π_θ` and
   then certifying robust invariance was asking the networks to have a property the
   *ground-truth teacher does not have*. This explains, at the value level, why the
   certified robust set collapsed to ≈0 and the robust viability set was ≈1–3%:
   the largest robustly-invariant subset of `{V_HJ≥0}` under `d∈[−0.3,0.3]` is
   genuinely small (≤ the ~34% surviving a constant bias, minus adaptive erosion).

3. **The correct certifiable object is not `{V_HJ≥0}`.** Two sound routes:
   - **Undiscounted viability kernel** (`γ→1`): the true `min(g, max_u min_d V(f))`
     fixed point *is* robustly invariant on its zero level set — but it "collapses
     under interpolation" on this grid (oracle note), the numerical reason γ<1 was
     used. The discount that buys VI stability is exactly what breaks invariance.
   - **Explicit recurrence / T4 shrink** to the greatest fixed point of
     `S↦S∩K∩Pre^∀d(S)` — which *does* verify multi-step robust invariance, and
     whose smallness is now **explained** (it is the true robust kernel, not a
     verifier failure).

**Bottom line.** The perfect filter is far from collision-free: ≈15% at `d=0`
(thin shell) and **≈66% under the matched adversary** (broad), from its own
`{V_HJ≥0}`. The failure is a clean, quantified consequence of the **discounted**
safety value decaying under worst-case disturbance — not verifier looseness, not
NN error, not deployment resolution. The object to certify must be a genuine
(undiscounted / fixed-point) robust invariant set.
