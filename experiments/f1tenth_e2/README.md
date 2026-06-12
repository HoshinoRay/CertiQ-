# F1TENTH E2 — Certified Safe Racing on the Brakeable Bicycle (**Gate D: PASS**)

The experiment that turns Gate D green. E0/E1 showed the certifier is *sound* but
that a **fixed-speed** car cannot pass Gate D: its maximal robust-invariant sets
are thin constant-V orbits, and the interval one-step box undershoots their
boundary with no contraction margin to absorb it
(`docs/dubins_e0_results_and_ablation.md` §4–5). E2 removes the unrealistic
assumption — the car can **brake to a stop** — and the wall dissolves.

## Plant + value (`qcbf/dynamics/bicycle_accel.py`)

```
x = (px, py, psi, v),  v in [0, v_max]      u = (a, delta)      d = (d_a, d_delta)
V(px,py,v) = clearance(p) - D(v) - margin
```

`D(v)` is the **exact discrete braking distance** at the guaranteed deceleration
`b = |a_min| - d_a_max`. The CBF is invariant by a one-line structural argument:
under one braking step the clearance lost (`<= dt*v`, Euclidean 1-Lipschitz) is
cancelled **exactly** by the braking distance recovered (`D(v) - D(v - dt b) =
dt*v`), so `V(x+) - V(x) >= 0` for worst-case heading *and* disturbance. That is
the contraction margin a fixed-speed car structurally lacks. The interval
primitives (`brake_cbf_bounds`, `braking_successor`, `dist_bounds`) are
machine-verified in `tests/test_bicycle_accel.py`.

## What the certificate checks (no skip, no cascade, no learned value)

1. `{V >= 0}` is **non-empty** on a grid (machine-enumerated).
2. The braking decrease `V(x+) - V(x) >= 0` on a grid (coupled displacement
   bound) — so `{V >= 0}` is robustly forward-invariant under braking.
3. `V >= 0  =>  clearance >= margin > 0  =>  g > 0`, so the certified set is
   collision-free by construction.
The deployed filter is min-intervention: apply the racing action iff a sound
worst-disturbance check gives `V(x+) >= buffer` (the car can still brake from
`x+`); otherwise brake. The adversarial audit (extremal + greedy) closes the loop.

## Result (`python experiments/f1tenth_e2/run_cert.py`, ~1 s)

```
certified set {V>=0}: 86 580 / 384 000 cells (22.5% of domain), up to 2.30 m/s
braking decrease: min V(x+)-V(x) = -0.0000  (>= 0, invariant)
audit:extremal  min g +0.118  min V +0.0002  collisions 0  brakes 79%
audit:greedy    min g +0.161  min V +0.0002  collisions 0  brakes 83%
GATE D: PASS   (certified-but-violated = 0)
```

The car **races** (accelerates ~20% of steps, brakes near the obstacle), keeps
`V >= 0` under both adversaries, and never collides (`g >= +0.12`). This is the
positive demonstration that completes the story: a sound dynamics-agnostic
certifier + a sharp characterisation of where cell+box reachability fails
(fixed-speed) + a certified safe-racing result on the realistic brakeable plant.
