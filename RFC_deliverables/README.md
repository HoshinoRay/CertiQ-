# RFC deliverables — post-hoc certification of a deployed learned robust Q-CBF filter

E2-L / brakeable F1TENTH. The learned object `(V_θ, Q_θ, π♭)` is a plain ReLU MLP
tuple — **no analytic `D(v)`, no hard-Lipschitz**. All safety claims are
discharged **post hoc** by the verifier on the frozen networks + true `f` +
full `D`. Start with **`RESULTS_RFC.md`** (the full write-up), then the figures.

## Headline numbers

| set | verified? | coverage of Ω* | audit cbv | meaning |
|---|---|---|---|---|
| **S_brake^Q** (Q-live fallback-pinned) | **yes** | **≈ 0.80** (res56: 0.803) | **0** | **main Q-CBF certificate** |
| S_brake (V_θ brake backbone) | yes | 0.80–0.81 | 0 | safety backbone |
| naive racing (baseline Q_θ) | **no** | 0.635 *apparent* | **205** | false-feasibility (Theorem S) |
| conservative Q_θ (c2fix) | no | — | 3144 | repair failed (ablation) |

Resolution-stable (res44 → 0.779, res56 → 0.803); Q-liveness 0.97–0.99;
false-feasibility bank N=540 (Q_θ says "race OK" but successor not brake-safe).

## Folder map

```
RESULTS_RFC.md      ← final clean result: table, claims, what-not-to-claim, reproduce
TECHNICAL_REPORT.md ← FULL JOURNEY: every ρ=0 direct-verification failure, the
                       rejected "near-cheat" ρ≈0.45 (and why), → the ρ≈0.80 result
figures/
  safe_slices.png            Ω* (gray) vs S_brake (blue) vs S_brake^Q (green), v=0.5/1.5/2.0
  false_feasibility.png      racing states Q_θ admits but whose successor collides
  resolution_and_audit.png   ρ vs grid; certified-but-violated bar (0 / 205 / 3144)
data/
  00_probe_baseline.json     exact γ=1 non-decrease probe → level-independent holes (Gate 0b)
  00_probe_c2fix.json        same after conservative training (still Gate 0b)
  01_brake_funnel_phase1.json   S_brake / S_brake^Q result + audit (the positive certificate)
  02_deployed_racing_phase2.json racing closure result + audit (rejected)
  03_false_feasibility_stats.json / .npz   Experiment C counterexample bank
logs/
  run_res44_full.log         full pipeline @ res 44 (probe → funnel → racing → audit)
  run_res56_confirm.log      resolution confirmation @ res 56
  run_c2fix_ablation.log     conservative-Q ablation (failed)
  run_counterexamples_figures.log   bank + figures
code/
  run_cert_rfc.py            the certifier (probe, brake funnel, deployed cert, audits)
  cex_and_figs.py            counterexample bank + figures
  distill.py                 trains the frozen learned trio (non-vacuity only)
```

## One-line claim

Post-hoc verification **accepts** a large fallback-pinned learned Q-CBF safe set
(≈80% of analytic safe volume, zero violations) and **rejects** the naive racing
Q_θ filter (false feasibility) — Theorem A and Theorem S on the *deployed learned
object*.

## Reproduce

```
python experiments/f1tenth_e2/run_cert_rfc.py --res 44     # full pipeline
python experiments/f1tenth_e2/run_cert_rfc.py --res 56     # resolution confirm
python experiments/f1tenth_e2/run_cert_rfc.py --res 44 --c2fix   # conservative-Q ablation
python experiments/f1tenth_e2/cex_and_figs.py              # bank + figures
python -m tests.test_direct                                # primitive soundness D1–D4
```
