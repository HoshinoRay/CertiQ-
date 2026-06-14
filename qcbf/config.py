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
        """Dubins angular-rate control bound."""
        return self.omega_max


@dataclass(frozen=True)
class OracleConfig:
    """Grid CBVF/HJ value iteration for the Dubins safety game."""
    n_px: int = 51
    n_py: int = 51
    n_psi: int = 41          # periodic, cell-centered
    n_u: int = 17
    n_d: int = 7
    tol: float = 1e-5
    max_iters: int = 400


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
    # ---- two decoupled, distinct knobs ------------------------------------ #
    # gamma_deploy: the deployed per-step CBF decay (gamma_d = exp(-lambda*dt)).
    #   It is the ONLY decay the certificate checks; runtime gate, C3 and C4 all
    #   use it.  ~0.90 is a gentle, physical class-K rate at dt=0.10.
    # gamma_teach: the *discount* lambda in the teacher's discounted safety
    #   backup  V <- (1-lambda) g + lambda * min(g, max_u min_d V(f)).  It only
    #   shapes the (untrusted) labels and the reference volume Omega* = {V>=0};
    #   it never enters the certificate.  ~0.92 keeps Omega* non-empty and
    #   resolution-robust (DESIGN_REVIEW "anti-spec trap").
    gamma_deploy: float = 0.90  # beta(r) = gamma_deploy * r  (deployed/cert decay)
    gamma_teach: float = 0.92   # teacher discount lambda (discounted safety VI)
    # ---- certificate-aligned training pressures --------------------------- #
    # Differentiable hinges that push the LEARNED artifact toward C1/C4 (never
    # proof assumptions; the verifier still checks the frozen networks).  These
    # are exactly the form reusable as RL/adversarial losses later.
    # C1 safety floor: penalize relu(V_theta(x) - g(x) + c1_floor_margin), i.e.
    #   push V_theta <= g - margin so {V_theta>=0} stays inside K.
    c1_floor_w: float = 1.0
    c1_floor_margin: float = 0.08    # moat ~ one cell width so straddling cells
                                     # have V_theta < 0 throughout (C1)
    # C4 one-sided consistency (trained on Q with V frozen): penalize
    #   relu(Q_theta(x,u,d) - V_theta(f(x,u,d)) + c4_oneside_margin), pushing
    #   Q_theta <= V_theta(f) - margin so the gate never over-states the successor.
    c4_oneside_w: float = 1.0
    c4_oneside_margin: float = 0.03    # Q just below V(f): small, to leave C3 room
    # CBF decrease margin on V_theta: push max_u min_d V(f(x,u,d)) >= gamma V + m
    # so the witnessed one-step decrease clears the verifier's interval slack.
    v_dec_w: float = 1.0
    v_dec_margin: float = 0.12
    v_dec_n_u: int = 7                 # control menu for the decrease backup
    v_dec_n_d: int = 3                 # disturbance grid for the decrease backup
    # teacher-fit weight: the MSE-to-teacher is only an ANCHOR/regularizer, not
    # the objective (the CBF-condition losses are).  Down-weight it so it does
    # not fight the floor/decrease shaping.  (Plain MSE is itself not obviously
    # optimal -- a margin/sign-aware fit is a candidate refinement.)
    teacher_fit_w: float = 0.5
    # verifier-in-the-loop certified C4 training (EXPERIMENTAL, default OFF):
    # acts on the cell-worst bound  ub Q_theta(C,u,D) <= lb V_theta(f) - margin.
    # Uses a CROWN-IBP eps-schedule: ramp the certified box from a point (eps=0,
    # exact) to the full cell (eps=1, the deployed condition) over the first
    # cert_eps_warmup_frac of the cert epochs, then hold at 1.  This avoids the
    # one-shot IBP collapse (pure full-width IBP is ~11 loose and crushes Q / C3).
    # Soundness unchanged: IBP looser than CROWN, so the eps=1 condition trained
    # here implies the deployed CROWN verifier's C4.  See DEVELOPMENT_LOG.
    cert_c4_w: float = 0.0
    cert_c4_margin: float = 0.02
    cert_n_cells: int = 4096
    cert_d_subsplit: int = 2
    cert_eps_start: float = 0.0        # box half-width fraction at epoch 0
    cert_eps_warmup_frac: float = 0.5  # fraction of cert epochs to ramp eps 0->1
    # two-sided C3 up-pressure: push min_d lb_IBP Q(C,u*,D) >= gamma*ubV + eps for
    # the best valid menu action, so the C4 down-push cannot crush Q below the
    # gate (one-sided C4 certifies C4 ~99% but kills C3).  0 = one-sided C4 only.
    cert_c3_w: float = 0.0
    # cell-worst certified training of V (the V analog of the Q lever; runs after
    # the pointwise train_v_cbf, before Q).  Shrinks V's CELL-WORST slack -- the
    # binding barrier for C1 (ub_IBP V<-m on g<0 cells) and the witness band
    # (min_d lb_IBP V(f) >= gamma*ub_IBP V(C) + m).  Sound (IBP => CROWN); a
    # teacher anchor keeps {V>=0} from collapsing.  0 = OFF.
    cert_v_w: float = 0.0              # C1-floor push weight (ub_IBP V < -m, g<0)
    cert_v_c1_margin: float = 0.10
    # decrease/band push weight.  DEFAULT 0: the decrease push inflates V globally
    # and destroys C1 (no Fisac clamp), so V-cert is C1-floor-ONLY unless this is
    # explicitly raised.  cert_v_w handles the safety floor; the band is left to
    # the teacher's natural margin (widen it later with finer verifier cells).
    cert_v_dec_w: float = 0.0
    cert_v_dec_margin: float = 0.10
    cert_v_anchor_w: float = 1.0       # teacher anchor, applied to SAFE cells only


@dataclass(frozen=True)
class CertConfig:
    """Cell-lattice verifier for the strict deployed Q-CBF specification."""
    n_cells_px: int = 40
    n_cells_py: int = 40
    n_cells_psi: int = 40
    n_u_cells: int = 8          # partition of U = [-omega_max, omega_max]
    # tightness knobs (all sound: sub-splitting only tightens bounds)
    c3_state_subsplit: int = 2  # split each state cell 2x2x2 for the C3 bound
    c3_d_subsplit: int = 2      # split the d-box for the C3 bound
    ante_d_probes: int = 3      # fixed d-probes for the sound min_d Q upper bound
    c2_d_subsplit: int = 2      # split the d-box for the C4 successor boxes
    c4_psi_subsplit: int = 1    # split each cell's heading for the C4 bounds
                                # (sound cos/sin tightening; measured a minor
                                # lever, so default off -- raise if trig becomes
                                # the binding slack after training)
    eps_margin: float = 5e-3    # verifier slack epsilon (never an approximation constant)
    tighten_intermediate: bool = True   # backward-CROWN intermediate bounds
    chunk: int = 512            # batched-verifier chunk size


@dataclass(frozen=True)
class AuditConfig:
    seed: int = 1
    n_rollouts: int = 400       # per disturbance mode
    horizon: int = 200
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
            cert=CertConfig(**raw["cert"]),
            audit=AuditConfig(**raw["audit"]),
        )
