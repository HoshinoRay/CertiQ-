# Design Review — Certified-Robust Q-CBF (Dubins E0)

This document records the load-bearing design decisions in the pipeline, the
soundness argument each one supports, and the places where engineering pragmatism
trades tightness (never soundness) for tractability. It is written for a
reviewer who wants to attack the proof, not admire the plots.

The certified claim, stated once, precisely:

> Let `Ω_cert ⊆ X` be the set of lattice cells the verifier accepts. For every
> state `x₀` whose cell is in `Ω_cert`, and for every admissible disturbance
> sequence `d₀, d₁, … ∈ D`, the closed loop formed by the **runtime filter built
> from the same frozen `V_θ, Q_θ, π♭_φ`** keeps the state inside `Ω_cert` and
> satisfies `g(x_t) ≥ 0` (no collision, in-domain) for all `t`.

This rests on **Theorem A**: if (C1) `g ≥ 0` on every accepted cell, (C2) the
robust one-step image of every accepted cell under the filtered dynamics is
covered by accepted cells, and (C3) at every accepted cell there is a *witnessed
feasible action* with margin `Q_θ(x,π(x),d) ≥ γV_θ(x)+ε` for all `d`, then the
accepted set is robustly forward-invariant and safe. The slack `ε` is consumed
by **none** of the three conditions in the proof — it is pure margin against the
verifier's own looseness.

---

## 1. The runtime ↔ certificate handshake (the subtle one)

The certificate certifies a *set*, but safety is a property of the *closed
loop*. These coincide only if the runtime filter never takes an action the
verifier did not account for. The handshake:

- **Runtime** applies a candidate `u` *only if* a **sound CROWN lower bound** of
  `min_d Q_θ(x,u,d)` is `≥ γ V_θ(x)`. Otherwise it falls back to `clip(π♭_φ(x))`.
- **Verifier** `antecedent_skip` declares a `(cell, u-cell)` pair *vacuous* when
  the **CROWN upper bound** of the antecedent `a = min_d Q_θ − γV_θ` is `< 0`
  over the whole cell×u-cell box — i.e. *no* state in that cell could ever pass
  the runtime gate with that `u`. C2 then only has to cover the non-skipped
  pairs.

Consistency is *directional and therefore sound*: the runtime gate uses a
**lower** bound (so it only ever applies actions that truly satisfy the margin),
while the skip test uses an **upper** bound (so it only skips pairs that truly
can never be applied). Any action the runtime can actually apply is therefore
inside the set C2 reasoned about.

The fallback `clip(π♭_φ(x))` is covered through **C3**: where the *true* witness
margin is `≥ ε`, the corresponding u-cell is never skipped, so C2 closure already
includes the fallback's successor. This is why C3 is a *feasibility* witness, not
a performance objective — it exists precisely to keep the fallback inside the
certified envelope.

## 2. Mode B — cell-lattice greatest fixed point; `c` is only an initialiser

C2 is a set-closure condition, solved as a **Tarski greatest fixed point** on the
finite cell lattice (Prop T4): start from a candidate set, delete any cell whose
robust successor range is not fully accepted, repeat to convergence. The
implementation (`c2_fixed_point`) prunes with a 3-D summed-area table so each
sweep is `O(#cells)`.

The level parameter `c` enters **only** as the initial set
`Cand(c) = C1 ∧ C3 ∧ (lbV ≥ max(c,0))`. It is *not* a threshold in the proof and
larger `c` is not "more conservative certification" — it is a different
starting point for the same fixed-point operator. Because every `c` reuses the
same C1/C3/successor precompute, the **c-sweep is nearly free**: only the cheap
pruning loop reruns. We report `ρ(c)` across the sweep to expose the
volume/robustness frontier; Gate D uses the **best** (largest) certified set.

## 3. An all-piecewise-linear artifact

Every network is ReLU-MLP and the policy head is the **exact** hardtanh
`clip(y) = ReLU(y+ω) − ReLU(y−ω) − ω`. Consequently `h3` (below) is exactly
piecewise-linear, and IBP/CROWN bounds are exact relaxations of the true
function — no smooth-activation slack, no surrogate. The price is that a learned
"clip" would have been smoother to train; we accept the kink because it makes the
verifier exact and the compile lossless.

## 4. Compiling `h3` into a single network preserves the x↔u correlation

The C3 quantity is

```
h3(x,d) = Q_θ(x, clip(π♭_φ(x)), d) − γ V_θ(x) − ε .
```

A naïve verifier that bounds `π`, then `Q`, then subtracts a bound on `V`, would
treat the `x` feeding `π` and the `x` feeding `Q`/`V` as **independent**
intervals and lose the correlation, massively inflating the bound. Instead
`compile_h3` emits **one** `SeqNet` on input `(px,py,ψ,d)` in which:

- the policy sub-net computes `u = clip(π(x))` and routes it forward,
- the original `x` is carried *losslessly* alongside via `±ReLU` identity pairs,
- `Q`, `γV`, and the `−ε` bias are stacked into the same affine-ReLU stream.

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

## 7. Sound antecedent skip via fixed `d`-probes

The skip upper bound is `min` over a few fixed disturbance probes `δ_k` of
`CROWN-ub Q(·,·,δ_k) − γ·CROWN-lb V`. Taking the min over probes can only make
the skip test **harder to pass** (smaller upper bound is *not* what we want for
soundness — note we require the ub `< 0`), so using a finite probe set is sound:
a real `min_d` is `≤` any finite-probe `min`, hence if the finite-probe upper
bound is already `< 0`, the true antecedent is too. `ante_d_probes` (1 in pilot,
3 in paper) trades tightness for cost.

## 8. C2 membership by outward-rounded ranges + summed-area table

Interval successor boxes are converted to **conservatively outward-rounded**
integer index ranges on the lattice (a box that touches a cell includes that
cell). Coverage — "is every cell in this range accepted?" — is answered in `O(1)`
by an 8-term inclusion–exclusion on a **3-D prefix-sum** of the acceptance mask.
The heading axis of the prefix sum is **tiled ×2** and zero-padded so a wrapped
range needs no seam special-casing. Outward rounding guarantees the membership
test is *conservative*: it can reject a truly-covered cell, never accept an
uncovered one.

## 9. Verifier slack `ε`, float64, and the rounding caveat

`ε = 5e-3` is the C3 witness slack. The verifier runs in float64 and the bounds
are mathematically sound, **but** the implementation does **not** use directed
(outward) rounding on the floating-point bound arithmetic itself. For the E0
result this is immaterial — the certified margins clear `ε` by orders of
magnitude relative to float64 ulp — but a *formally airtight* deployment should
either (a) add interval directed rounding to `bounds.py`, or (b) inflate `ε`
past a conservative global float64 round-off budget. We flag this explicitly
rather than bury it; it is the one place the chain is "sound modulo IEEE
round-off" instead of "sound, period".

## 10. Witness-margin fine-tuning (training-time only)

After `V_θ, Q_θ` are frozen, `π♭_φ` is fine-tuned to raise the verified
composition margin `m(x) = min_k Q_θ(x,π(x),d_k) − γV_θ(x)` toward
`m_target = 0.06` via a hinge loss, with gradients flowing **only through
`dQ/du`** (V, Q stay frozen; a small anchor to the oracle-fallback labels
prevents drift). This is a *training-time* nudge to make C3 pass on more cells —
it changes which cells get certified, **not** whether certification is sound. The
verifier re-checks the *actual* fine-tuned `π` from scratch; nothing about the
fine-tuning is trusted.

## 11. Staged C3 (only ever *adds* certified cells)

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
ground-truth value iteration

```
V ← min( g , max_u min_d V(f(x,u,d)) )      (undiscounted Bellman–Isaacs avoid)
```

with **multilinear interpolation** of `V` at the (off-grid) successor
`f(x,u,d)` **collapses to `g_fail` everywhere** at *any* finite grid resolution —
the positive (safe-invariant) region drains to empty. We verified this is **not**
an implementation bug:

- the vectorised backup matches a brute-force `interp_V` successor evaluation to
  0.0 on 200 random states;
- a hand-built orbiting feedback controller keeps `g ≥ 0.47` for 2000 steps
  against a greedy adversary, so a robustly-invariant set provably **exists**;
- yet the interpolated minimax fixed point reports `Vol(Ω*) = 0` at 21³, 31³, 41³.

The mechanism: multilinear interpolation in the minimax backup behaves like
**never-ending stochastic state noise** (the interpolant lets the adversary "mix"
neighbouring values every step). Over an infinite horizon this noise wins at any
fixed resolution and the avoid value degenerates. Two standard fixes were tested:

- **Discounted safety backup** (Fisac/Akametalu),
  `V ← (1−λ)g + λ·min(g, max_u min_d V(f))`. Converges, positive region
  ~40% of domain, **resolution-robust**. Cost: the effective horizon `~1/(1−λ)`
  is finite, so a fully-adversarial greedy rollout from deep-interior states can
  still fail — acceptable for a *teacher*, since the certificate, not the oracle,
  carries soundness.
- **Vertex-min backup** (successor value = min over the 8 bracketing grid
  vertices). Monotone, clean semantics (positive region = robust-invariant set
  with one cell of margin), but **over-pessimistic** here: the per-step
  guaranteed heading authority (`dt·ω_max = 0.1` rad) is below the heading cell
  width, so the pessimism outruns the control and the set still empties.

**Decision:** use the **discounted interp backup (`λ = 0.92`)** as the default
teacher (`OracleConfig.backup="interp", discount=0.92`). The oracle's *only* jobs
are (i) generating sensible training labels and fallback actions and (ii) defining
the ρ-denominator `Vol(Ω*)`. Certificate soundness is independent of the oracle
being an exact HJ solution — it comes from C1∧C2∧C3 + Theorem A, double-checked
by the adversarial **audit** that searches for any certified-but-violated state.
This is exactly the kind of separation-of-concerns the architecture was built to
exploit: a *good-enough* learned/▴taught artifact, made *trustworthy* by an
independent sound verifier.

> Reviewer takeaway: the oracle is a **teacher, not a spec**. If a reader asks
> "is your reachable set exact?", the answer is "the *certified* set is sound by
> construction and audited; the *oracle* set is only a denominator and a label
> source." The discount is a teacher hyperparameter, not a soundness knob.

---

## Reuse / hardware path (F1TENTH)

The only problem-specific soundness primitives are the four interval methods in
`dynamics/` (`step`, `g`, `g`-box bounds, `successor_boxes`). Porting to an
F1TENTH bicycle model means subclassing those; the network artifact, compiler,
CROWN/IBP verifier, lattice closure, c-sweep, runtime filter and audit are all
reused unchanged. Wider or structured disturbances are handled by certifying
against an augmented `D_aug ⊇ D` (Corollary-M hook) — the single place that
needs widening is the disturbance interval inside `successor_boxes`. Because the
verifier is plain NumPy with no solver dependency, the resulting certificate is
bit-reproducible and small enough to audit by hand, which is the entire point of
the exercise.
