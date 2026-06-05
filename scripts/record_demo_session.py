#!/usr/bin/env python3
"""演示会话录制：PNG + JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.experiments.camera_live import DualEvalCapture
from src.experiments.pipeline import EvalConfig, EvalPipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="S1_normal")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument(
        "--out",
        type=Path,
        default=_REPO.parent.parent / "06-testing" / "test-records",
    )
    args = parser.parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_dir = args.out / f"session_{args.scenario}_{ts}"
    session_dir.mkdir(parents=True, exist_ok=True)

    capture = DualEvalCapture()
    pipe = EvalPipeline(EvalConfig(mode="fusion"))
    try:
        with open(session_dir / "states.jsonl", "w", encoding="utf-8") as jf:
            for i in range(args.frames):
                bundle = capture.read()
                primary, err, tid, iou = pipe.process_frame(bundle.rgb, bundle.temp)
                jf.write(
                    json.dumps(
                        {
                            "frame": i,
                            "detected": primary is not None,
                            "center_error_px": err,
                            "fusion_match_iou": iou,
                            "track_id": tid,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                if bundle.rgb is not None and i % args.interval == 0:
                    cv2.imwrite(str(session_dir / f"rgb_{i:04d}.png"), bundle.rgb)
    finally:
        capture.release()

    (session_dir / "meta.json").write_text(
        json.dumps({"scenario": args.scenario, "frames": args.frames, "started_at": ts}, indent=2),
        encoding="utf-8",
    )
    print(f"会话已保存: {session_dir}")


if __name__ == "__main__":
    main()
