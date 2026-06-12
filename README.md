# Certified-Robust Q-CBF — Dubins-Car Experiment (E0)

A **sound, post-hoc certification pipeline** for a *frozen, learned* robust
state–action control-barrier filter on a Dubins car under a bounded adversarial
disturbance.  Given three trained networks

| symbol | role | shape (default) |
|--------|------|-----------------|
| `V_θ(x)`            | learned barrier value          | MLP 3→64→64→1 (ReLU) |
| `Q_θ(x,u,d)`        | learned robust state–action value | MLP 5→96→96→1 (ReLU) |
| `π♭_φ(x)`           | learned backup policy (`u∈[−1,1]`) | MLP 3→64→64→1 + hardtanh |

the pipeline produces a **machine-checkable certificate** that the *runtime
filter built from those very networks* renders an explicit subset of state space
**robustly forward-invariant and collision-free for every admissible
disturbance sequence** — not in expectation, not on samples, but for the whole
continuous disturbance set `D = [−0.3, 0.3]`.

The proof object is the trio of conditions **(C1) ∧ (C2) ∧ (C3)** of *Theorem A*;
the verifier discharges them with **interval bound propagation + CROWN linear
relaxation** over a lattice of state cells, entirely in **NumPy** (no autodiff
framework, no solver) so the verifier itself is short enough to audit by hand.

> **What is *certified* vs. *learned*.** The networks are arbitrary learned
> artifacts — we never trust them. Soundness comes only from the verifier's
> bounds and Theorem A. The oracle value iteration is a *teacher* used to label
> training data and to define the ρ-denominator `Vol(Ω*)`; **it is never part of
> any soundness claim.**

---

## 1. Install

Pure NumPy. Matplotlib only for the figures.

```bash
python -m venv .venv && source .venv/bin/activate
pip install "numpy>=1.26" matplotlib
```

No network access, GPU, or compiled extension is required at any stage.
Tested on Python 3.12 / NumPy 2.4.

## 2. Thirty-second sanity check

Run the unit suite — it proves the *verifier primitives* are sound
(every CROWN/IBP bound encloses the true network, the compiled `h3` network
reproduces `Q(x,clip(π(x)),d)−γV(x)−ε` to ~1e-15, the interval trig / `g`-bounds
enclose their functions, successor boxes contain the true image incl. heading
wrap):

```bash
python tests/test_soundness.py
# ... all soundness tests passed
```

Then a fast end-to-end smoke run of the *whole* pipeline on a deliberately tiny
grid (finishes in ~1 min; it is a plumbing check, **not** a Gate-D demo — the
12³ certification lattice is far too coarse to retain cells through the C2
closure, so it reports Gate D: FAIL by design):

```bash
python experiments/dubins_e0/run_all.py --config experiments/dubins_e0/config_smoke.json
```

## 3. The real experiment

Two profiles use a certification lattice fine enough to retain cells through the
C2 closure and are intended to demonstrate **Gate D** (a non-empty certified set
with **zero** certified-but-violated states in the adversarial audit):

```bash
# Pilot  (~20-40 min on 1 CPU): 51³ oracle, 40³ lattice, 8 u-cells
python experiments/dubins_e0/run_all.py --config experiments/dubins_e0/config_pilot.json

# Paper  (hours): 81³ oracle, 56³ lattice, ante_d_probes=3, larger c-sweep
python experiments/dubins_e0/run_all.py --config experiments/dubins_e0/config_paper.json
```

`run_all.py` chains the six stages and each stage can also be run on its own
(they communicate only through artifact files in `results/<name>/`, so any stage
can be re-run without repeating the expensive ones):

```bash
python experiments/dubins_e0/run_oracle.py  --config <cfg>   # V* teacher + Ω* volume
python experiments/dubins_e0/run_train.py   --config <cfg>   # fit V_θ, Q_θ, π♭_φ; witness fine-tune
python experiments/dubins_e0/run_certify.py --config <cfg>   # compile h3, C1∧C2∧C3, c-sweep, GATE D
python experiments/dubins_e0/run_audit.py   --config <cfg>   # adversarial falsification audit
python experiments/dubins_e0/run_figures.py --config <cfg>   # all paper figures
```

### What you get in `results/<name>/`

```
oracle.npz / oracle_report.json     V*, residual history, Ω* fraction
v.npz q.npz pi.npz                  frozen network weights (.npz)
train_report.json                   MSEs + witness-margin diagnostics on Ω*
certificate.npz / cert_report.json  accepted mask, per-cell lbV/ubV, C1/C3 flags,
                                    skip mask, c-sweep table, GATE D verdict, ρ
audit_report.json                   iid / extremal / greedy-adversary results;
                                    certified_but_violated count (must be 0)
manifest.json                       per-stage wall time + SHA-256 of every artifact,
                                    all keyed to the config hash (tamper-evident)
figures/  (*.png)                   certified slices, c-sweep ρ-curve, margins,
                                    fixed-point convergence, oracle, audit rollouts
```

The headline number is

```
ρ(c) = Vol(Ω_cert(c)) / Vol(Ω*)
```

reported for every `c` in the sweep; **Gate D passes iff the best certified set
is non-empty and the audit finds zero violations.**

---

## 4. Repository layout

```
qcbf/
  config.py              frozen dataclasses for every knob; JSON (de)serialise;
                         config-hash; all stages assert the hash in manifest.json
  dynamics/dubins.py     DubinsModel (step/g/in_domain), exact interval arithmetic:
                         cos/sin intervals, quadratic g-box bounds, successor
                         boxes with exact heading-wrap splitting
  oracle/value_iteration ground-truth Bellman–Isaacs VI (the *teacher*); fast
                         bilinear-xy + heading-roll backup; Ω* volume by MC
  nets/mlp.py            tiny hand-written MLP (forward/backward/Adam, .npz I/O),
                         exact hardtanh policy head, witness-margin fine-tuning
                         (frozen Q,V; gradient flows through dQ/du only)
  verify/
    bounds.py            SeqNet, IBP, CROWN backward (batched), CROWN∩IBP
    compiler.py          compile h3 = Q(x,clip(π(x)),d) − γV(x) − ε into ONE
                         affine-ReLU SeqNet that preserves the x↔u correlation
    conditions.py        check_c1 (exact), per-cell V bounds, staged C3
                         (IBP→CROWN→sub-split), antecedent skip test
  certify/
    cells.py             CellLattice; outward-rounded successor index ranges;
                         3-D summed-area acceptance test (heading axis tiled ×2)
    refine.py            precompute_certificate + c2_fixed_point (Tarski greatest
                         fixed point via prefix-sum pruning)
    csearch.py           Ω* MC volume + c-sweep + Gate-D verdict
  runtime/filter.py      CertifiedFilter: min-intervention action with a *sound*
                         CROWN feasibility check; clip(π) fallback
  audit/falsify.py       adversarial rollouts (iid / extremal / greedy-on-V),
                         checks g≥0 AND cell-membership every step
  plots/figures.py       all paper figures (matplotlib, Agg backend)

experiments/dubins_e0/   run_*.py drivers + common.py + the three config JSONs
tests/test_soundness.py  verifier-primitive soundness suite (T1–T8)
```

See **`DESIGN_REVIEW.md`** for the eleven load-bearing design decisions
(runtime↔certificate handshake, why `c` is only an initialiser, the all-PWL
artifact, the single-network `h3` compile, the oracle "anti-spec" trap and its
fix, the float-rounding caveat, and the reuse path to F1TENTH hardware).

## 5. Reusing this for your own system

The certification machinery is dynamics-agnostic. To port to a new plant
(e.g. an F1TENTH bicycle model):

1. Subclass the dynamics: provide `step`, `g`, `in_domain`, exact `g`-box
   bounds, and `successor_boxes` (interval image of one step over a state×u×d
   box). These are the *only* problem-specific soundness primitives.
2. Keep `V_θ, Q_θ, π♭_φ` as ReLU MLPs with a hardtanh policy head so the whole
   `h3` artifact stays piecewise-linear and the CROWN/IBP verifier applies
   unchanged.
3. For richer disturbance models, certify against an augmented set `D_aug ⊇ D`
   (the Corollary-M hook): nothing in Theorem A or the verifier assumes `D` is
   an interval other than inside `successor_boxes`, which is the single place to
   widen.

Because the verifier is plain NumPy, the certificate is reproducible bit-for-bit
and small enough to inspect line-by-line — which is the point.

**Worked example (F1TENTH kinematic bicycle).** This reuse path is implemented:
`qcbf/dynamics/bicycle.py` (+ `tests/test_bicycle.py`) provides the four interval
primitives for a fixed-speed bicycle with steering disturbance, and
`experiments/f1tenth_e1/run_cert.py` runs the full certifier on it. The only
generalisations the rest of the code needed were a `dyn.control_max` property, a
`model.heading_rate(u,d)` hook, and an injectable `successor_boxes_fn`; the
oracle is now the plant-agnostic `GridOracle`. See
`experiments/f1tenth_e1/README.md` and
`docs/dubins_e0_results_and_ablation.md` §6.

**Gate-D PASS (variable-speed safe racing).** The fixed-speed E0/E1 cases
*characterise* a real limit — their robust-invariant sets are thin constant-V
orbits that the interval one-step box cannot close. The realistic fix is a
**brakeable** 4-state bicycle with an analytic **braking-distance CBF**
(`qcbf/dynamics/bicycle_accel.py`, `tests/test_bicycle_accel.py`,
`experiments/f1tenth_e2/`): because the car can stop, `{V≥0}` is robustly
invariant by construction and **Gate D passes** (86 580-cell certified set,
zero certified-but-violated states under an adversarial audit, while the car
races). This is the positive headline result; see `experiments/f1tenth_e2/README.md`.

**Learned-object characterisation (E2-L).** Certifying the **deployed learned**
trio `V_θ, Q_θ, π♭_φ` *directly* (CROWN on the networks, no `V_θ≈V_target`
assumption) is a **sound FAIL by both routes**, and that is the point:
(A) the heading-free braking sub-level never closes because the brake-successor
undershoot `≈` the CROWN gap on `V_θ` `≈` the true per-cell variation of `V`
(>0), whereas the analytic CBF closes via an *exact 1-Lipschitz cancellation* a
black-box net cannot reproduce; (B) the heading-inclusive 4-D cell-reachability
erodes to empty **even for the analytic ideal `V`** (the cell+box obstruction).

**Scope of (B) (corrected).** Because route B erodes the *analytic ideal `V`* to
empty too (116784→0), its failure characterises the **certification primitive**
(outward-rounded cell+box one-step reachability), **not** the learned object: a
primitive that cannot close even on a known-invariant ideal set cannot be used to
conclude that the learned object is uncertifiable. The correct scoped statement
is: *outward-rounded cell+box one-step reachability cannot non-vacuously certify
any bounded robust-invariant set here; whether a tighter sound primitive
(direct-composition CROWN / zonotope / contraction) closes on the learned object
is an open experimental question* (in progress — see `run_cert_p1.py`). Route A's
knife-edge failure is a genuine learned-vs-analytic structural fact, but is
specific to the brake-only sub-level. Driver:
`experiments/f1tenth_e2/run_cert_learned.py`, write-up
`experiments/f1tenth_e2/README_LEARNED.md`.
