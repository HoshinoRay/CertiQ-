# F1TENTH E1 — Certified-Robust Q-CBF on the Kinematic Bicycle

The same post-hoc certifier as Dubins E0, ported to an F1TENTH-scale **kinematic
bicycle** to demonstrate the reuse path (README §5): *only the dynamics change.*

```
State    x = (px, py, psi)            Control  u = delta (steering, |delta|<=delta_max)
Disturb  d (steering disturbance)
px+ = px + dt v cos(psi)              (shared with Dubins -> oracle stencil reused)
py+ = py + dt v sin(psi)
psi+= wrap(psi + dt (v/L) tan(delta+d))   (bicycle yaw rate; one new tan interval)
```

## What is reused vs. new

| reused unchanged | new (plant-specific) |
|---|---|
| CROWN/IBP verifier, `h3` compile, lattice C2 closure, c-sweep, runtime filter, audit, MLPs, training | `qcbf/dynamics/bicycle.py`: `step/g/successor_boxes` + `tan_interval` |

The certifier was made plant-agnostic with two hooks — `dyn.control_max` and
`model.heading_rate(u,d)` — plus an injectable `successor_boxes_fn`. The grid
oracle is `GridOracle(dyn, cfg, model=BicycleModel(dyn))`.

## Run

```bash
python tests/test_bicycle.py                                   # primitive soundness (T-bike)
python experiments/f1tenth_e1/run_cert.py --scale smoke        # ~35 s end-to-end plumbing
python experiments/f1tenth_e1/run_cert.py --scale pilot        # Gate-D attempt (56^3, sharpened)
```

Configs are built in code (`make_cfg`) so no JSON plant-dispatch is needed;
`results/f1tenth_e1_<scale>/f1tenth_report.json` holds the funnel, Gate-D
verdict, ρ and the adversarial-audit counts.

## Why this should certify where Dubins struggled

The Dubins ablation (`docs/dubins_e0_results_and_ablation.md`) showed Gate D
hinges on **skip selectivity**, which needs (i) a **sharpened value**
(`TrainConfig.value_sharpen`, carried over here) and (ii) enough robust-set
*thickness*. The F1TENTH steering authority `(v/L)·tan(δ_max) ≈ 2.56 rad/s` is
large relative to the `~0.6 rad/s` disturbance, so the robust invariant set is
fatter than the Dubins `R ≥ 1.43` annulus — more room for the interval-C2
closure to retain cells.
