#!/usr/bin/env python3
"""7.3 照度梯度：200/50/10/2 lux × rgb/ir/fusion。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.eval_nine_groups import _wait_operator
from src.experiments.camera_live import DualEvalCapture
from src.experiments.pipeline import EvalConfig, run_trial

LUX_LEVELS = [200, 50, 10, 2]
MODES = ["rgb", "ir", "fusion"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--out", type=Path, default=_REPO / "experiments" / "records" / "lux_sweep.csv")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    capture = DualEvalCapture()
    rows = []
    try:
        for lux in LUX_LEVELS:
            input(f"请将照度调至约 {lux} lux，操作员 1.5m 就位后 Enter…")
            for mode in MODES:
                _wait_operator(0, args.frames)
                r = run_trial(
                    capture,
                    EvalConfig(mode=mode),
                    f"lux_{lux}",
                    0,
                    frames=args.frames,
                    env_lux=float(lux),
                )
                rows.append(
                    {
                        "lux": lux,
                        "mode": mode,
                        "detection_rate": r.detection_rate,
                        "center_error_px": r.center_error_px,
                        "fps": r.fps,
                    }
                )
                print(f"  lux={lux} {mode}: det={r.detection_rate:.3f}")
    finally:
        capture.release()

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lux", "mode", "detection_rate", "center_error_px", "fps"])
        w.writeheader()
        w.writerows(rows)
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
