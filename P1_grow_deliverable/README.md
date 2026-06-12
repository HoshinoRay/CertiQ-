# P1 — Grow-from-seed certified invariant-set expansion (self-contained deliverable)

Everything from the P1 experiment in one place: the **method**
([`TECHNICAL.md`](TECHNICAL.md)), the **results report**
([`RESULTS.md`](RESULTS.md)), the **figures** ([`figures/`](figures/)), and the
**data / logs / code** used to produce them. Experiment P1 of the next-step
framework `../grow_from_seed_certified_expansion.md` §9.

## One-paragraph summary

A frozen learned robust Q-CBF trio `(V_θ,Q_θ,π♭)` is grown from its brake seed
`S_brake` (ρ≈0.80) by a **verified least-fixed-point operator** `G_V` that adds
heading-specific cells able to *drive* into the certified set in one sound step.
**The load-bearing result is soundness:** every iterate is a real robust invariant
set — `cbv ≡ 0` under three independent audits (Enc MC, grown one-step
containment, layered rollout) at res 36/44/56/64/80. The certified coverage `ρ_∞`
is **resolution-monotone** (0.818→**0.880**) and the grow contribution `Δρ`
**crosses the +0.05 GO bar at res 80**. A capture-basin calibration shows the
remaining residual is **recoverable conservatism** (the sound set recovers 86 % of
the reachable basin within the anchor; capture-gap of Ω\* ≈ 0) and that the true
capture basin is **≈ 1.9× Ω\*** (the brake-anchored denominator understates the
safe set). The deployed certified shield is **safe (cbv 0)** while the naive `Φ_θ`
filter is **unsafe (cbv 303)** — Theorem-S: **24 %** of `Φ_θ`-feasible actions
escape the certified set. The same `Q_θ` is over-permissive as a *predicate* yet
**top-5 = 100 %** as a *proposal*.

## Figures

| file | what it shows |
|---|---|
| `figures/fig1_resolution_sweep.png` | `ρ_∞` monotone↑ to 0.88; `Δρ` crosses +0.05 (GO) at res 80; ideal by res 56 |
| `figures/fig2_grow_curve_layers.png` | anytime `ρ(R_k)` growth + per-layer onion histogram |
| `figures/fig3_failure_decomposition.png` | proven 4-way blockage: `v0` (learned anchor) vs `frontier` (heading Enc) |
| `figures/fig4_capture_calibration.png` | recovery 0.86 within anchor, capture-gap ≈ 0, **Viab ≈ 1.9× Ω\*** |
| `figures/fig5_deployed_threeway.png` | shield cbv 0 (safe) vs naive cbv 303 (unsafe) vs brake; progress |
| `figures/fig6_q_duality.png` | `Q_θ`: 24 % over-permissive predicate, top-5 = 100 % proposal |
| `figures/fig7_certified_slices.png` | `R_∞` (px,py) slices: heading-fraction certified, vs Ω\*/obstacle/wall |
| `figures/fig8_onion_layers.png` | lfp layer `ℓ` at which each (px,py) joins `R_∞` |

## Layout

```
README.md         this index
TECHNICAL.md      method: operator, sound primitives, Theorem A, A_ver, calibration
RESULTS.md        full results report (tables, calibration, deployment, verdict)
figures/          fig1..fig8 (.png)
data/             p1_grow_full.json (calib+deploy+Q), p1_grow_grid_res56.npz (slices)
logs/             per-resolution run logs (res 36/44/56/64/80, npsi 32, --full)
code/             snapshot of the scripts (engine, driver, reference, analysis, figures)
```

## Reproduce

```bash
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 44 --full   # grow + calib + deploy
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 80          # GO point
python experiments/f1tenth_e2/run_cert_p1_grow.py --res 56 --dump   # grid for slices
python experiments/f1tenth_e2/analyze_p1_grow.py                    # table + fits + calib
python experiments/f1tenth_e2/make_p1_figures.py                    # all figures
python -m tests.test_bicycle_accel                                  # primitive soundness
```
