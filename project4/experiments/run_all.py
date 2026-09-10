"""Regenerate project4's committed assets: simulation metrics + report figures.

    python -m project4.experiments.run_all

Outputs land in project4/assets/ and are committed, so the report can be served
without running anything.
"""

import json
import os
import time

from . import simulate
from .. import plots


def main():
    os.makedirs(simulate.OUT_DIR, exist_ok=True)
    t0 = time.time()
    metrics = simulate.run()
    with open(os.path.join(simulate.OUT_DIR, "sim_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"simulation done in {time.time() - t0:.1f}s")

    paths = plots.build_all(metrics, simulate.OUT_DIR)
    for name, path in paths.items():
        print(f"  {name}: {os.path.relpath(path)} ({os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    main()
