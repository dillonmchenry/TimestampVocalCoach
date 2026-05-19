"""Internal helper: execute every code cell in notebooks/sprint1_demo.ipynb in
order, without Jupyter, so we can verify the demo runs end-to-end. Not meant
for normal users."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "sprint1_demo.ipynb"


def main() -> int:
    import matplotlib
    matplotlib.use("Agg")

    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    print(f"Notebook OK; {len(nb['cells'])} cells")

    ns = {"__name__": "__demo__", "__file__": str(NB_PATH)}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        try:
            exec(compile(src, f"<cell {i}>", "exec"), ns)
        except Exception as exc:
            print(f"  cell {i}: ERROR {type(exc).__name__}: {exc}")
            raise
        print(f"  cell {i}: OK")
    print("All code cells executed without errors.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
