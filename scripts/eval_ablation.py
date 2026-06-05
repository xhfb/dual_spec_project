#!/usr/bin/env python3
"""消融实验 A1/A2/A3（S2/S3 恶劣场景）。

对照组:
  baseline   — 全功能
  A1         — --no-registration
  A2         — --fixed-weights 0.5 0.5
  A3         — --no-tracker
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.experiments.camera_live import DualEvalCapture
from src.experiments.pipeline import EvalConfig, run_trial

ABLATIONS = {
    "baseline": EvalConfig(mode="fusion"),
    "A1_no_registration": EvalConfig(mode="fusion", no_registration=True),
    "A2_fixed_weights": EvalConfig(mode="fusion", fixed_weights=(0.5, 0.5)),
    "A3_no_tracker": EvalConfig(mode="fusion", no_tracker=True),
}
DEFAULT_SCENARIOS = ["S2_low_light", "S3_backlight"]


def main() -> None:
    parser = argparse.ArgumentParser(description="消融实验 runner")
    parser.add_argument("--scenario", choices=DEFAULT_SCENARIOS + ["S1_normal"])
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--out", type=Path, default=_REPO / "experiments" / "records")
    args = parser.parse_args()

    scenarios = DEFAULT_SCENARIOS if args.all_scenarios else [args.scenario or "S2_low_light"]
    args.out.mkdir(parents=True, exist_ok=True)
    capture = DualEvalCapture()
    records = []

    try:
        for scenario in scenarios:
            lux_raw = input(f"[{scenario}] 照度 lux: ").strip()
            lux = float(lux_raw) if lux_raw else float("nan")
            for ab_name, cfg in ABLATIONS.items():
                for trial in range(args.trials):
                    input(f"[{scenario}/{ab_name} trial {trial}] 人员就位后 Enter…")
                    r = run_trial(capture, cfg, scenario, trial, args.frames, env_lux=lux, notes=ab_name)
                    records.append({**asdict(r), "ablation": ab_name})
                    print(
                        f"  det={r.detection_rate:.3f} iou={r.fusion_match_iou_mean:.3f} "
                        f"id_sw={r.id_switch_count} rec_ms={r.recovery_ms}"
                    )
    finally:
        capture.release()

    out_path = args.out / "ablation_results.json"
    payload = {
        "meta": {"created_at": datetime.now(timezone.utc).isoformat(), "frames": args.frames},
        "records": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"已写入 {out_path}")


if __name__ == "__main__":
    main()
