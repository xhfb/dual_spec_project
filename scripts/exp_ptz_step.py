#!/usr/bin/env python3
"""G6 云台 PD 阶跃/跟随实验。"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

RGB_CENTER = (320.0, 240.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--ptz", action="store_true")
    parser.add_argument("--out", type=Path, default=_REPO / "experiments" / "records" / "ptz_step.csv")
    args = parser.parse_args()

    from src.experiments.camera_live import DualEvalCapture
    from src.experiments.pipeline import EvalConfig, EvalPipeline

    gimbal = None
    if args.ptz:
        from src.ptz.gimbal_tracker import GimbalTracker

        gimbal = GimbalTracker(config_path=_REPO / "config" / "ptz.yaml")
        if not gimbal.initialize():
            print(f"云台初始化失败: {gimbal.status}", file=sys.stderr)
            return
    else:
        print("警告: 未加 --ptz，仅记录像素误差")

    capture = DualEvalCapture()
    pipe = EvalPipeline(EvalConfig(mode="fusion"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.perf_counter()
    try:
        for i in range(args.frames):
            bundle = capture.read()
            primary, err, _, _ = pipe.process_frame(bundle.rgb, bundle.temp)
            cx, cy = RGB_CENTER
            if primary is not None:
                cx, cy = primary.center
            if gimbal is not None:
                gimbal.update(cx if primary else None, cy if primary else None)
            rows.append(
                {
                    "frame": i,
                    "t_s": time.perf_counter() - t0,
                    "error_px": err if primary else float("nan"),
                    "detected": int(primary is not None),
                }
            )
            if i % 30 == 0:
                print(f"frame {i} err={err:.1f}px det={primary is not None}")
    finally:
        capture.release()
        if gimbal is not None:
            gimbal.stop()

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame", "t_s", "error_px", "detected"])
        w.writeheader()
        w.writerows(rows)
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
