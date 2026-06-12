"""Stage 5 -- paper figures."""
import json
import types

from common import load_cfg, out_path, update_manifest

import numpy as np

from qcbf.certify.cells import CellLattice
from qcbf.nets.mlp import MLP
from qcbf.oracle.value_iteration import DubinsOracle
from qcbf.plots import figures
from qcbf.runtime.filter import CertifiedFilter
from qcbf.verify.bounds import SeqNet


def main() -> None:
    cfg = load_cfg()
    lat = CellLattice.build(cfg.dynamics, cfg.cert)
    cert = np.load(out_path(cfg, "certificate.npz"))
    accepted = cert["accepted"]
    oracle = DubinsOracle(cfg.dynamics, cfg.oracle)
    oz = np.load(out_path(cfg, "oracle.npz"))
    sol = {"V": oz["V"], "history": oz["history"], "iters": int(oz["iters"]),
           "wall_s": float(oz["wall_s"])}
    sweep = {"entries": json.loads(
        out_path(cfg, "cert_report.json").read_text())["sweep"]}

    pre = types.SimpleNamespace(c3_lb=cert["c3_lb"], lbV=cert["lbV"])
    v = MLP.load(str(out_path(cfg, "v.npz")))
    q = MLP.load(str(out_path(cfg, "q.npz")))
    pi = MLP.load(str(out_path(cfg, "pi.npz")))
    filt = CertifiedFilter(lat, accepted, v, q, pi, SeqNet.from_mlp(q),
                           cfg.train.gamma, cfg.dynamics.omega_max,
                           cfg.dynamics.d_max, cfg.audit.n_u_candidates)

    out = str(out_path(cfg, ""))
    paths = figures.make_all(cfg, lat, pre, accepted, sweep, oracle, oz["V"],
                             sol, filt, out.rstrip("/"))
    for p in paths:
        print(f"[figures] wrote {p}")
    update_manifest(cfg, "figures", {"n_figures": len(paths)})


if __name__ == "__main__":
    main()
