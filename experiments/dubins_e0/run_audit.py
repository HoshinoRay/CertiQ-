"""Stage 4 -- falsification audit of the certified closed loop."""
from common import file_hash, load_cfg, out_path, save_json, update_manifest

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

    # Refuse a certificate mask that does not belong to the CURRENT config and
    # the CURRENT frozen weights -- otherwise we would audit new networks against
    # a stale acceptance set and report meaningless numbers.
    cert = np.load(out_path(cfg, "certificate.npz"))
    required = ("config_hash", "v_hash", "q_hash", "pi_hash", "strict_spec_pass")
    missing = [k for k in required if k not in cert.files]
    if missing:
        raise SystemExit(f"[audit] certificate.npz missing provenance {missing}; "
                         "re-run run_certify.py")
    expect = {"config_hash": cfg.hash(),
              "v_hash": file_hash(out_path(cfg, "v.npz")),
              "q_hash": file_hash(out_path(cfg, "q.npz")),
              "pi_hash": file_hash(out_path(cfg, "pi.npz"))}
    for k, want in expect.items():
        got = str(cert[k])
        if got != want:
            raise SystemExit(
                f"[audit] certificate {k} mismatch ({got} != {want}); the mask was "
                "certified against a different config/weights -- re-run run_certify.py")
    if not bool(cert["strict_spec_pass"]):
        print("[audit] strict spec did not pass -- nothing to audit")
        save_json(cfg, "audit_report.json", {"pass": False,
                                             "reason": "strict spec failed"})
        return
    accepted = cert["accepted"]
    if not accepted.any():
        print("[audit] no certified cells -- nothing to audit (empty certificate)")
        save_json(cfg, "audit_report.json", {"pass": False,
                                             "reason": "empty certificate"})
        return

    lat = CellLattice.build(cfg.dynamics, cfg.cert)
    filt = CertifiedFilter(lat, accepted, v, q, pi, SeqNet.from_mlp(q),
                           cfg.train.gamma_deploy, cfg.dynamics.omega_max,
                           cfg.dynamics.d_max, cfg.cert.n_u_cells)
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
