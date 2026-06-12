"""Experiment configuration.

Every stage of the pipeline (oracle -> training -> certification -> audit)
is driven by a single frozen `ExperimentConfig`.  Configs serialize to JSON;
every artifact written to disk embeds the config hash so that a certificate
can always be traced back to the exact dynamics, networks and verifier
settings that produced it.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------- #
# Sub-configs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DubinsConfig:
    """Fixed-speed Dubins car with additive angular-rate disturbance."""
    dt: float = 0.10
    v: float = 1.0
    omega_max: float = 1.0
    d_max: float = 0.30
    # safety geometry
    obs_center: tuple[float, float] = (0.0, 0.0)
    obs_radius: float = 0.45
    world_radius: float = 1.80
    # state domain (position); heading is always [-pi, pi) periodic
    p_lo: float = -2.0
    p_hi: float = 2.0
    # value assigned to successors that leave the position domain
    g_fail: float = -1.0

    @property
    def control_max(self) -> float:
        """Generic control-input bound used by the plant-agnostic verifier."""
        return self.omega_max


@dataclass(frozen=True)
class OracleConfig:
    """Grid Bellman-Isaacs value iteration (ground truth, *not* a proof object)."""
    n_px: int = 51
    n_py: int = 51
    n_psi: int = 41          # periodic, cell-centered
    n_u: int = 17
    n_d: int = 7
    tol: float = 1e-5
    max_iters: int = 400
    # Successor-value evaluation inside the Bellman-Isaacs backup:
    #   "vertex_min": V(f(x,u,d)) lower-bounded by the min over the 8 grid
    #       vertices bracketing f(x,u,d).  Makes the (undiscounted) backup
    #       monotone decreasing from V0 = g, so VI converges and the positive
    #       region of the fixed point is a robustly invariant set *with one
    #       grid cell of margin* -- a pessimistic but reliable teacher.
    #   "interp": plain multilinear interpolation.  NOT recommended for the
    #       undiscounted infinite-horizon game: interpolation acts as
    #       never-ending stochastic state noise, and the avoid value
    #       degenerates to g_fail at any finite resolution.
    backup: str = "interp"
    # Discount for the Fisac-style discounted safety backup
    #   V <- (1-discount)*g + discount*min(g, max_u min_d V(f))
    # 1.0 = undiscounted.  The undiscounted minimax backup under multilinear
    # interpolation degenerates to g_fail at any finite resolution (interpolation
    # acts as never-ending stochastic state noise), so a discount < 1 is used to
    # obtain a convergent teacher with a stable positive region.  The oracle is a
    # *teacher* for label generation and the rho-denominator only -- certificate
    # soundness rests on C1/C2/C3 + Theorem A + the empirical audit, never on the
    # oracle being an exact HJ solution.
    discount: float = 0.92


@dataclass(frozen=True)
class NetConfig:
    """All-piecewise-linear artifact: ReLU hidden layers, hardtanh policy head.

    Piecewise linearity is a deliberate verification choice: every activation
    admits an exact-piece CROWN relaxation, so no transcendental-activation
    bounds enter the trusted computing base.
    """
    v_hidden: tuple[int, ...] = (64, 64)
    q_hidden: tuple[int, ...] = (96, 96)
    pi_hidden: tuple[int, ...] = (64, 64)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    n_q_samples: int = 250_000
    batch: int = 1024
    lr: float = 1e-3
    epochs_v: int = 60
    epochs_q: int = 40
    epochs_pi: int = 60
    # witness-margin fine-tuning of the policy head (frozen V, Q)
    epochs_margin: int = 40
    margin_target: float = 0.06
    margin_d_grid: int = 5
    gamma: float = 0.5       # beta(r) = gamma * r (runtime decay)
    # Optional monotone value sharpening of the regression targets:
    #   V_theta, Q_theta are fit to  phi(V*) , phi(V* o f)  with
    #   phi(v) = tanh(value_sharpen * v)  (0 disables -> plain V* regression).
    # phi is monotone with phi(0)=0, so {V_theta>=0} = {V*>=0} and the oracle
    # fallback argmax is unchanged; sharpening only steepens the gate
    # Q >= gamma V near the safe-set boundary, which makes the (sound)
    # antecedent-skip test selective -- the load-bearing fix for C2 closure
    # when V* is too flat (see DESIGN_REVIEW: flat-value -> skip~0 -> C2 empty).
    value_sharpen: float = 0.0


@dataclass(frozen=True)
class CertConfig:
    """Cell-lattice certificate (Mode B / Prop T4 fixed point)."""
    n_cells_px: int = 40
    n_cells_py: int = 40
    n_cells_psi: int = 40
    n_u_cells: int = 8          # partition of U = [-omega_max, omega_max]
    # tightness knobs (all sound: sub-splitting only tightens bounds)
    c3_state_subsplit: int = 2  # split each state cell 2x2x2 for the C3 bound
    c3_d_subsplit: int = 2      # split the d-box for the C3 bound
    ante_d_probes: int = 3      # fixed d-probes for the sound antecedent upper bound
    c2_u_subsplit: int = 2      # split each control cell for successor boxes
    c2_d_subsplit: int = 2      # split the d-box for successor boxes
    eps_margin: float = 5e-3    # verifier slack epsilon (never an approximation constant)
    c_sweep: tuple[float, ...] = (0.00, 0.02, 0.05, 0.10, 0.15, 0.20)
    tighten_intermediate: bool = True   # backward-CROWN intermediate bounds
    chunk: int = 512            # batched-verifier chunk size


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 1
    n_rollouts: int = 400       # per disturbance mode
    horizon: int = 200
    n_u_candidates: int = 21    # runtime filter action grid
    adversary_d_grid: int = 21  # greedy adversary disturbance grid


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "dubins_e0_pilot"
    out_dir: str = "results/dubins_e0_pilot"
    dynamics: DubinsConfig = field(default_factory=DubinsConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    nets: NetConfig = field(default_factory=NetConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    cert: CertConfig = field(default_factory=CertConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)

    # ----------------------------------------------------------------- IO
    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @staticmethod
    def load(path: str | Path) -> "ExperimentConfig":
        raw = json.loads(Path(path).read_text())
        return ExperimentConfig(
            name=raw["name"],
            out_dir=raw["out_dir"],
            dynamics=DubinsConfig(**{**raw["dynamics"],
                                     "obs_center": tuple(raw["dynamics"]["obs_center"])}),
            oracle=OracleConfig(**raw["oracle"]),
            nets=NetConfig(**{k: tuple(v) for k, v in raw["nets"].items()}),
            train=TrainConfig(**raw["train"]),
            cert=CertConfig(**{**raw["cert"], "c_sweep": tuple(raw["cert"]["c_sweep"])}),
            audit=AuditConfig(**raw["audit"]),
        )
