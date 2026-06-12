"""Stage 4 -- falsification audit of the certified closed loop."""
from common import load_cfg, out_path, save_json, update_manifest

import numpy as np

from qcbf.audit.falsify import run_audit
from qcbf.certify.cells import CellLattice
from qcbf.dynamics.dubins import DubinsModel
from qcbf.nets.mlp import MLP
from qcbf.runtime.filter import CertifiedFilter
from qcbf.verify.bounds import SeqNet


def main() -> None:
    cfg = load_cfg()
    v = MLP.load(str(out_path(cfg, "v.npz")))
    q = MLP.load(str(out_path(cfg, "q.npz")))
    pi = MLP.load(str(out_path(cfg, "pi.npz")))
    accepted = np.load(out_path(cfg, "certificate.npz"))["accepted"]
    if not accepted.any():
        print("[audit] no certified cells -- nothing to audit (Gate D failed)")
        save_json(cfg, "audit_report.json", {"pass": False,
                                             "reason": "empty certificate"})
        return

    lat = CellLattice.build(cfg.dynamics, cfg.cert)
    filt = CertifiedFilter(lat, accepted, v, q, pi, SeqNet.from_mlp(q),
                           cfg.train.gamma, cfg.dynamics.omega_max,
                           cfg.dynamics.d_max, cfg.audit.n_u_candidates)
    model = DubinsModel(cfg.dynamics)
    print(f"[audit] {cfg.audit.n_rollouts} rollouts x {cfg.audit.horizon} "
          f"steps per mode, from {int(accepted.sum())} certified cells")
    report = run_audit(cfg, model, filt, verbose=True)
    print(f"[audit] certified-but-violated = "
          f"{report['certified_but_violated']}, invariance violations = "
          f"{report['invariance_violations']}  -> "
          f"{'PASS' if report['pass'] else 'FAIL'}")
    save_json(cfg, "audit_report.json", report)
    update_manifest(cfg, "audit", {
        "pass": report["pass"],
        "certified_but_violated": report["certified_but_violated"],
        "invariance_violations": report["invariance_violations"]})


if __name__ == "__main__":
    main()
