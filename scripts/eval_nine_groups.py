#!/usr/bin/env python3
"""T9 九组对比实验：S1/S2/S3 × RGB/IR/Fusion。

用法（板端）::

    python3 scripts/eval_nine_groups.py --all --trials 10 --frames 90
    python3 scripts/eval_nine_groups.py --scenario S2_low_light --mode fusion --trials 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.experiments.camera_live import DualEvalCapture
from src.experiments.pipeline import EvalConfig, aggregate_trials, run_trial

SCENARIOS = ["S1_normal", "S2_low_light", "S3_backlight"]
MODES = ["rgb", "ir", "fusion"]
SCENARIO_HINTS = {
    "S1_normal": "正常光照 >200 lux，操作员站 1.5m 画面中央",
    "S2_low_light": "低照度 <10 lux，操作员站 1.5m 画面中央",
    "S3_backlight": "强背光/逆光，RGB 局部过曝，操作员站 1.5m",
}


def _prompt_env(scenario: str) -> tuple[float, str]:
    print(f"\n=== 场景 {scenario} ===")
    print(SCENARIO_HINTS.get(scenario, ""))
    lux_raw = input("照度 lux（数字，未知回车跳过）: ").strip()
    lux = float(lux_raw) if lux_raw else float("nan")
    notes = input("备注（可选）: ").strip()
    return lux, notes


def _wait_operator(trial: int, frames: int) -> None:
    print(f"\n[Trial {trial}] 请操作员站入 1.5m 标定距离、画面中央区域。")
    print(f"按 Enter 开始采集 {frames} 帧…")
    input()


def main() -> None:
    parser = argparse.ArgumentParser(description="九组对比实验 runner")
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--all", action="store_true", help="跑完全部 9 组")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--out", type=Path, default=_REPO / "experiments" / "records")
    parser.add_argument("--dry-run", action="store_true", help="仅打印计划不采数")
    args = parser.parse_args()

    if not args.all and (not args.scenario or not args.mode):
        parser.error("请指定 --scenario + --mode，或使用 --all")

    scenarios = SCENARIOS if args.all else [args.scenario]
    modes = MODES if args.all else [args.mode]

    plan = [(s, m) for s in scenarios for m in modes]
    if args.dry_run:
        print("计划运行:", plan, f"trials={args.trials} frames={args.frames}")
        return

    args.out.mkdir(parents=True, exist_ok=True)
    capture = DualEvalCapture()
    all_trials = []
    env_records = []

    try:
        for scenario in scenarios:
            lux, notes = _prompt_env(scenario)
            env_records.append({"scenario": scenario, "lux": lux, "notes": notes})
            for mode in modes:
                cfg = EvalConfig(mode=mode)
                for trial in range(args.trials):
                    _wait_operator(trial, args.frames)
                    result = run_trial(
                        capture,
                        cfg,
                        scenario,
                        trial,
                        frames=args.frames,
                        env_lux=lux,
                        notes=notes,
                    )
                    all_trials.append(result)
                    print(
                        f"  {scenario}/{mode} trial={trial}: "
                        f"det={result.detection_rate:.3f} fps={result.fps:.1f} "
                        f"err={result.center_error_px:.1f}px lost={result.lost_count}"
                    )
    finally:
        capture.release()

    summary = aggregate_trials(all_trials)
    payload = {
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trials_per_group": args.trials,
            "frames_per_trial": args.frames,
            "protocol": "presence_based",
            "distance_m": 1.5,
        },
        "environment": env_records,
        "trials": [asdict(t) for t in all_trials],
        "summary": summary,
    }

    json_path = args.out / "benchmark_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = args.out / "experiment_log.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "scenario",
                "mode",
                "trial_id",
                "detection_rate",
                "center_error_px",
                "fps",
                "lost_count",
                "recovery_ms",
                "fusion_match_iou_mean",
                "env_lux",
            ]
        )
        for t in all_trials:
            w.writerow(
                [
                    t.scenario,
                    t.mode,
                    t.trial_id,
                    f"{t.detection_rate:.4f}",
                    f"{t.center_error_px:.2f}" if t.center_error_px == t.center_error_px else "",
                    f"{t.fps:.2f}",
                    t.lost_count,
                    f"{t.recovery_ms:.1f}" if t.recovery_ms == t.recovery_ms else "",
                    f"{t.fusion_match_iou_mean:.3f}" if t.fusion_match_iou_mean == t.fusion_match_iou_mean else "",
                    t.env_lux,
                ]
            )

    print(f"\n已写入:\n  {json_path}\n  {csv_path}")
    print("请复制到知识库: 06-testing/metrics/benchmark_results.json")


if __name__ == "__main__":
    main()
