"""Run the full E0 pipeline: oracle -> train -> certify -> audit -> figures.

Each stage runs in its own subprocess (memory isolation); artifacts and the
manifest accumulate under cfg.out_dir.

    python experiments/dubins_e0/run_all.py --config experiments/dubins_e0/config_pilot.json
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["run_oracle.py", "run_train.py", "run_certify.py",
          "run_audit.py", "run_figures.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(HERE / "config_pilot.json"))
    args = ap.parse_args()
    t0 = time.time()
    for stage in STAGES:
        print(f"\n========== {stage} ==========", flush=True)
        rc = subprocess.call([sys.executable, str(HERE / stage),
                              "--config", args.config])
        if rc != 0:
            print(f"stage {stage} failed (rc={rc})")
            sys.exit(rc)
    print(f"\npipeline complete in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
