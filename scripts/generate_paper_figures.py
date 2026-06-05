#!/usr/bin/env python3
"""从 benchmark JSON 生成论文图 5-1 / 表 5-1。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

try:
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except ImportError as exc:
    raise SystemExit("需要 matplotlib: pip install matplotlib") from exc


SCENARIO_LABELS = {
    "S1_normal": "S1 正常光",
    "S2_low_light": "S2 低照度",
    "S3_backlight": "S3 逆光",
}
MODE_LABELS = {"rgb": "RGB", "ir": "IR", "fusion": "Fusion"}


def _load_summary(benchmark_path: Path) -> dict:
    with open(benchmark_path, encoding="utf-8") as f:
        data = json.load(f)
    if "summary" in data and data["summary"]:
        return data["summary"]
    from src.experiments.pipeline import TrialResult, aggregate_trials

    trials = [
        TrialResult(**{k: t[k] for k in t if k != "frame_metrics"})
        for t in data.get("trials", [])
    ]
    return aggregate_trials(trials)


def plot_detection_rates(summary: dict, out_dir: Path) -> Path:
    scenarios = ["S1_normal", "S2_low_light", "S3_backlight"]
    modes = ["rgb", "ir", "fusion"]
    x = range(len(scenarios))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mode in enumerate(modes):
        vals, errs = [], []
        for s in scenarios:
            row = summary.get(f"{s}/{mode}", {})
            vals.append(row.get("detection_rate_mean", 0) * 100)
            errs.append(row.get("detection_rate_std", 0) * 100)
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width, yerr=errs, capsize=3, label=MODE_LABELS[mode])
    ax.set_ylabel("检出率 (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title("九组实验检出率对比（图 5-1）")
    fig.tight_layout()
    out_path = out_dir / "fig5-1_detection_rate.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_center_error(summary: dict, out_dir: Path) -> Path:
    scenarios = ["S1_normal", "S2_low_light", "S3_backlight"]
    modes = ["rgb", "ir", "fusion"]
    x = range(len(scenarios))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, mode in enumerate(modes):
        vals = []
        for s in scenarios:
            v = summary.get(f"{s}/{mode}", {}).get("center_error_px_mean")
            vals.append(v if v is not None else 0)
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], vals, width, label=MODE_LABELS[mode])
    ax.set_ylabel("中心误差 (px)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in scenarios])
    ax.legend()
    ax.set_title("九组实验中心误差对比")
    fig.tight_layout()
    out_path = out_dir / "fig5-1_center_error.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def export_table_md(summary: dict, out_dir: Path) -> Path:
    lines = [
        "# 表 5-1 测试指标汇总（自动生成）",
        "",
        "| 场景 | 模式 | 检出率 | 中心误差(px) | FPS | 丢失帧均值 |",
        "|------|------|--------|-------------|-----|-----------|",
    ]
    for key, row in sorted(summary.items()):
        scenario, mode = key.split("/", 1)
        lines.append(
            f"| {SCENARIO_LABELS.get(scenario, scenario)} | {MODE_LABELS.get(mode, mode)} | "
            f"{row['detection_rate_mean']:.3f} | "
            f"{row.get('center_error_px_mean') or 0:.1f} | "
            f"{row['fps_mean']:.1f} | {row['lost_count_mean']:.1f} |"
        )
    out_path = out_dir / "table_5_1.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=_REPO.parent.parent / "06-testing" / "metrics")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = _load_summary(args.benchmark)
    print("已生成:")
    print(" ", plot_detection_rates(summary, args.out))
    print(" ", plot_center_error(summary, args.out))
    print(" ", export_table_md(summary, args.out))


if __name__ == "__main__":
    main()
