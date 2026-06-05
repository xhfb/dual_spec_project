#!/usr/bin/env python3
"""G7 红外测温精度：IR 峰值 vs 标准温度计。"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--out", type=Path, default=_REPO / "experiments" / "records" / "temp_accuracy.csv")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from src.experiments.camera_live import DualEvalCapture

    capture = DualEvalCapture()
    rows = []
    try:
        for i in range(args.samples):
            input(f"[{i+1}/{args.samples}] 对准目标，Enter 采 IR 帧…")
            bundle = capture.read()
            if bundle.temp is None:
                print("  跳过: 无温度矩阵")
                continue
            peak_raw = float(np.max(bundle.temp))
            ref = input("  标准温度计读数 (℃): ").strip()
            ref_c = float(ref) if ref else float("nan")
            rows.append({"sample": i, "ir_peak_raw": peak_raw, "reference_c": ref_c})
    finally:
        capture.release()

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample", "ir_peak_raw", "reference_c"])
        w.writeheader()
        w.writerows(rows)
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
