# Design Review — Certified-Robust Q-CBF (Dubins E0)

This document records the load-bearing design decisions in the pipeline, the
soundness argument each one supports, and the places where engineering pragmatism
trades tightness (never soundness) for tractability. It is written for a
reviewer who wants to attack the proof, not admire the plots.

The certified claim, stated once, precisely:

> Let `V_θ, Q_θ, π_θ` be frozen. If the verifier accepts (C1 ∧ C3 ∧ C4 hold on
> every cell that can intersect `{V_θ ≥ 0}`), then for every state `x₀ ∈
> {V_θ ≥ 0}` and every admissible disturbance sequence `d₀, d₁, … ∈ D`, the
> deployed runtime filter built from the **same** `V_θ, Q_θ, π_θ` keeps the
> state in `{V_θ ≥ 0}` and satisfies `g(x_t) ≥ 0` (no collision, in-domain) for
> all `t`.

This rests on **Theorem A**, discharged here by three conditions on the
deployed networks, checked at the deployed decay `γ_deploy`:

- **C1** `g ≥ 0` on every cell that can intersect `{V_θ ≥ 0}` (safety floor).
- **C3** at every such cell the witness `u = π_θ(x)` clears the robust Q-gate
  with margin: `min_d Q_θ(x, π_θ(x), d) ≥ γ_deploy V_θ(x) + ε` (feasibility).
- **C4** every action the loop can apply (the finite menu and the witness)
  satisfies `min_d Q_θ(x,u,d) ≤ min_d V_θ(f(x,u,d))` (gate ⇒ decrease).

The robust one-step decrease `V_θ(f(x,u,d)) ≥ γ_deploy V_θ(x)` is **not** a
separate proof obligation (see §1). The slack `ε` is consumed by **none** of the
conditions in the proof — it is pure margin against the verifier's own looseness.

The exported `accepted` mask is the set of *inner* cells (`lbV ≥ 0`); it is a
conservative inner-volume lower bound on the true invariant set `{V_θ ≥ 0}`, and
a convenient source of audit initial states. It is **all-or-nothing**: the
certificate either passes on every active cell (and the inner cells are
reported) or it certifies nothing. There is no level parameter `c` and no
greatest-fixed-point set carving — `{V_θ ≥ 0}` is the invariant set, by the
decrease argument below.

---

## 1. Why C3 + C4 replace a separate decrease (C2) check

The deployed filter, at each `x`, applies either a finite-menu action `u` whose
**sound CROWN lower bound** of `min_d Q_θ(x,u,d)` is `≥ γ_deploy V_θ(x)`, or the
fallback `clip(π_θ(x))`. Two facts close the loop:

- **Gate ⇒ decrease.** Any applied menu action passes its gate, so its *true*
  `min_d Q_θ ≥ γ_deploy V_θ`. C4 (checked on that exact menu action) gives
  `min_d V_θ(f) ≥ min_d Q_θ`, hence `V_θ(f) ≥ γ_deploy V_θ ≥ 0` for all `d`:
  the successor is again in `{V_θ ≥ 0}`.
- **Feasibility.** If no menu action passes, the fallback is used. C3 proves the
  fallback passes the gate (margin `ε`), so the feasible set is never empty, and
  C4 on the witness action gives the same decrease for the fallback.

So at every `x ∈ {V_θ ≥ 0}` the loop has a feasible action and every action it
can take keeps the state in `{V_θ ≥ 0}`; C1 then gives `g ≥ 0` throughout. The
existential "∃ menu action with `V_θ(f) ≥ γ_deploy V_θ`" (the old C2) is
**implied** by C3 + C4 for the deployed gate, so it is dropped. This is the one
place to be careful: C4 must cover *exactly* the action set the runtime can
apply — the finite menu (at its exact scalar values) and the witness (enclosed
by the full control interval `[-ω, ω]`, which contains any `clip(π_θ(x))`).

Consistency is **directional and therefore sound**: the runtime gate uses a
*lower* bound of `min_d Q_θ` (so it only ever applies actions that truly satisfy
the margin), while C4 uses an *upper* bound of `min_d Q_θ` and a *lower* bound of
`min_d V_θ(f)` (so it only certifies the decrease where it truly holds).

## 2. Two distinct, decoupled knobs (`γ_deploy` ≠ teacher discount `λ`)

There are two roles that must **not** share one number:

- `γ_deploy` — the **deployed per-step CBF decay** (`γ_d = e^{−λ_c·dt}`). It is
  the only decay the certificate and runtime use (gate, C3, C4). `~0.90` is a
  gentle, physical class-K rate at `dt = 0.10`; the earlier `0.5` allowed `V`
  to halve every step (`≈6.9/s`), which is far too aggressive.
- `λ = γ_teach` — the **discount in the teacher's discounted safety backup**
  (§ "anti-spec trap"). It only shapes the (untrusted) labels and the reference
  volume `Ω* = {V ≥ 0}`; it never enters the certificate. `~0.92`.

These are genuinely different objects (a CBF decay vs. an HJ discount), so
collapsing them into one symbol is what produced the earlier breakage. A naïve
"single γ in a CBF value iteration" does not work at all here: `min(g, γF)` with
`γ<1` is identically `≤ 0` (so `Ω* = ∅`), and `min(g, F/γ)` *diverges*. The
teacher therefore uses the discounted backup, and the certificate uses
`γ_deploy` directly on the frozen networks. The γ-consistency that matters —
`runtime gate = C3 = C4 = γ_deploy` — is preserved, and `Ω*` is reported as the
ρ-denominator.

The teacher is an under-approximation, not an exact CBF: it carries a positive
deployed margin `~(1 − γ_deploy)·V` on the safe interior but can be slightly
negative on the `V = 0` boundary shell. That only narrows which cells the
*verifier* can certify; it is never a soundness issue.

## 3. An all-piecewise-linear artifact

Every network is ReLU-MLP and the policy head is the **exact** hardtanh
`clip(y) = ReLU(y+ω) − ReLU(y−ω) − ω`. Consequently `h3` (§4) is exactly
piecewise-linear, and IBP/CROWN bounds are exact relaxations of the true
function — no smooth-activation slack, no surrogate. The price is that a learned
"clip" would have been smoother to train; we accept the kink because it makes the
verifier exact and the compile lossless.

## 4. Compiling `h3` into a single network preserves the x↔u correlation

The C3 quantity is

```
h3(x,d) = Q_θ(x, clip(π_θ(x)), d) − γ_deploy V_θ(x) − ε .
```

A naïve verifier that bounds `π`, then `Q`, then subtracts a bound on `V`, would
treat the `x` feeding `π` and the `x` feeding `Q`/`V` as **independent**
intervals and lose the correlation, massively inflating the bound. Instead
`compile_h3` emits **one** `SeqNet` on input `(px,py,ψ,d)` in which:

- the policy sub-net computes `u = clip(π(x))` and routes it forward,
- the original `x` is carried *losslessly* alongside via `±ReLU` identity pairs,
- `Q`, `γ_deploy·V`, and the `−ε` bias are stacked into the same affine-ReLU
  stream.

CROWN then propagates a *single* linear relaxation through the shared `x`,
keeping the correlation. The compile is verified exact to ~1e-15 in T-tests
(`compile_h3`, `compile_policy`). Requirement: `π` and `V` share hidden depth so
the identity carry stays aligned; `Q`'s input order is `(px,py,ψ,u,d)`.

## 5. Exploiting Dubins structure in the oracle backup

The teacher VI never forms successors per `(state,u,d)` explicitly. Heading
enters the dynamics only as a **uniform shift** `ψ ← ψ + dt·(u+d)`, and the
position update depends only on `ψ`. So the backup is a precomputed **bilinear
xy-stencil** (4 neighbours, fixed weights) composed with a **periodic roll + lerp
in ψ** per `(u,d)`. This turns an `O(N·|u|·|d|)` gather into a handful of
vectorised `roll`/`einsum` calls and is what makes an 81³ oracle tractable on one
CPU. This is a *teacher-side* optimisation only and carries no soundness weight.

## 6. Exact interval primitives (the soundness floor)

Everything the proof ultimately rests on is exact, closed-form interval
arithmetic, unit-tested for enclosure:

- `g_bounds_on_box`: exact min/max of the quadratic obstacle/boundary `g` over an
  axis-aligned state box.
- `cos_interval` / `sin_interval`: exact ranges accounting for the
  `0, π/2, π, 3π/2` critical points inside the heading interval.
- `successor_boxes`: interval image of one dynamics step over a state×u×d box,
  with the **heading wrap split into two boxes that are never merged** (merging
  across the `±π` seam would silently over-cover). T7 checks 800 random boxes
  contain their sampled successors with the wrap-split active.

These feed C4: `min_d V_θ(f)` is lower-bounded by CROWN over the (interval)
successor box; `min_d Q_θ` is upper-bounded over fixed `d`-probes (a finite probe
min is `≥` the true `min_d`, so the upper bound is sound). This separated
successor-box bound is conservative but sound; tightening it (e.g. composing the
affine V-bound through the analytic `f` to share `x`) is a tractability lever for
later, not a soundness question.

## 7. Verifier slack `ε`, float64, and the rounding caveat

`ε = 5e-3` is the C3 witness slack. The verifier runs in float64 and the bounds
are mathematically sound, **but** the implementation does **not** use directed
(outward) rounding on the floating-point bound arithmetic itself. For the E0
result this is immaterial — the certified margins clear `ε` by orders of
magnitude relative to float64 ulp — but a *formally airtight* deployment should
either (a) add interval directed rounding to `bounds.py`, or (b) inflate `ε`
past a conservative global float64 round-off budget. We flag this explicitly
rather than bury it; it is the one place the chain is "sound modulo IEEE
round-off" instead of "sound, period".

## 8. Witness-margin fine-tuning (training-time only)

After `V_θ, Q_θ` are frozen, `π_θ` is fine-tuned to raise the verified
composition margin `m(x) = min_k Q_θ(x,π(x),d_k) − γ_deploy V_θ(x)` toward
`m_target = 0.06` via a hinge loss, with gradients flowing **only through
`dQ/du`** (V, Q stay frozen; a small anchor to the oracle-fallback labels
prevents drift). This is a *training-time* nudge to make C3 pass on more cells —
it changes which cells get certified, **not** whether certification is sound. The
verifier re-checks the *actual* fine-tuned `π` from scratch; nothing about the
fine-tuning is trusted.

## 9. Staged C3 (only ever *adds* certified cells)

C3 is discharged in three escalating stages over the undecided cells:

- **Stage A** — whole-cell IBP with IBP intermediate bounds (cheapest).
- **Stage B** — CROWN with tightened intermediate bounds on what A left undecided.
- **Stage C** — sub-split the cell (2×2×2 in state × 2 in `d`) and re-run on what
  B left undecided (most expensive, tightest).

Each stage is **sound on its own** and only *upgrades* undecided→certified, so
staging can only **enlarge** the certified set relative to running the cheap test
alone — it never certifies a cell the precise test would reject. This is purely a
compute-allocation strategy.

---

## The oracle "anti-spec" trap (a design-review finding worth its own section)

A subtle and instructive failure surfaced while validating the teacher. The
*undiscounted* avoid value iteration

```
V ← min( g , max_u min_d V(f(x,u,d)) )
```

with **multilinear interpolation** of `V` at the off-grid successor
`f(x,u,d)` **collapses to `g_fail` everywhere** at any finite grid resolution —
the positive (safe-invariant) region drains to empty. We verified this is **not**
an implementation bug:

- the vectorised backup matches a brute-force `interp_V` successor evaluation to
  0.0 on random states;
- a hand-built orbiting feedback controller keeps `g ≥ 0.47` for 2000 steps
  against a greedy adversary, so a robustly-invariant set provably **exists**;
- yet the undiscounted interpolated minimax fixed point reports `Vol(Ω*) = 0`.

The mechanism: multilinear interpolation in the minimax backup behaves like
**never-ending state noise** (the interpolant lets the adversary "mix"
neighbouring values every step). Over an infinite horizon this noise wins at any
fixed resolution and the avoid value degenerates.

We measured all the candidate backups directly (smoke grid, `_future_value`):

| teacher backup | result |
|---|---|
| `min(g, F)` (undiscounted, `λ=1`) | `Ω* = 0` (the collapse above) |
| `min(g, γF)`, `γ<1` | `V ≤ 0` everywhere ⇒ `Ω* = 0` (since `V_max = γF_max ≤ γV_max`) |
| `min(g, F/γ)`, `γ<1` | **diverges**, `V → −10⁶` |
| `(1−λ)g + λ·min(g, F)`, `λ=0.92` | **converges, `Ω* ≈ 38%`, resolution-robust** |

**The discounted safety backup `V ← (1−λ)g + λ·min(g, max_u min_d V(f))` is the
fix.** The `(1−λ)g` source term re-injects the true margin every sweep, pinning
`V` near `g` on the safe interior, so the interpolation dissipation can no longer
drain the positive region; the discount `λ<1` makes the map a contraction. `λ`
near 1 = lighter discount = larger `Ω*`; we use `λ = γ_teach = 0.92`. This is a
**teacher-side** choice only: the resulting `V_HJ` is an under-approximation, not
an exact CBF (it can have a slightly negative one-step CBF residual on the `V=0`
shell), which is why the *verifier*, not the oracle, carries soundness.

> Reviewer takeaway: the oracle is a **teacher, not a spec**. If a reader asks
> "is your reachable set exact?", the answer is "the *certified* set is sound by
> construction and audited; the *oracle* set is only a denominator and a label
> source." `γ_teach` is a teacher hyperparameter, not a soundness knob — the
> certificate is checked on the frozen networks at `γ_deploy` regardless.

---

## Reuse / hardware path (F1TENTH)

The only problem-specific soundness primitives are the interval methods in
`dynamics/` (`step`, `g`, `g`-box bounds, `successor_boxes`). Porting to an
F1TENTH bicycle model means subclassing those; the network artifact, compiler,
CROWN/IBP verifier, runtime filter and audit are all reused unchanged. Wider or
structured disturbances are handled by certifying against an augmented
`D_aug ⊇ D` — the single place that needs widening is the disturbance interval
inside `successor_boxes`. Because the verifier is plain NumPy with no solver
dependency, the resulting certificate is bit-reproducible and small enough to
audit by hand, which is the entire point of the exercise.
