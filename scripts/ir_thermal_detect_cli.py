#!/usr/bin/env python3
"""IR 热目标检测离线调试 CLI。

读入 roll 对齐后的温度矩阵（.npy），输出检测框并生成叠加可视化图。

用法：
  # 合成数据快速验通路
  python3 scripts/ir_thermal_detect_cli.py --synthetic

  # 标定归档温度矩阵
  python3 scripts/ir_thermal_detect_cli.py path/to/pair_01_ir_temp.npy -o out/

  # 原始矩阵，自动 roll 对齐
  python3 scripts/ir_thermal_detect_cli.py raw_temp.npy --align-roll -o out/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.detection.ir_thermal import (  # noqa: E402
    IR_HEIGHT,
    IR_WIDTH,
    IRThermalDetector,
    align_temp_matrix,
    load_config,
)


def make_synthetic_temp() -> np.ndarray:
    """生成含 2 个热斑的合成温度矩阵，用于无硬件验通路。"""
    base = np.full((IR_HEIGHT, IR_WIDTH), 3200, dtype=np.uint16)
    rng = np.random.default_rng(42)
    base += rng.integers(0, 80, size=base.shape, dtype=np.uint16)

    def blob(cx: int, cy: int, bw: int, bh: int, peak: int) -> None:
        y0, y1 = max(0, cy - bh // 2), min(IR_HEIGHT, cy + bh // 2)
        x0, x1 = max(0, cx - bw // 2), min(IR_WIDTH, cx + bw // 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dy = (yy - cy) / max(bh / 2, 1)
        dx = (xx - cx) / max(bw / 2, 1)
        dist = np.sqrt(dx * dx + dy * dy)
        blob_val = (peak * np.clip(1.0 - dist, 0, 1)).astype(np.uint16)
        base[y0:y1, x0:x1] = np.maximum(base[y0:y1, x0:x1], blob_val)

    blob(80, 60, 36, 72, 5200)
    blob(180, 130, 28, 56, 4800)
    return base


def temp_to_vis(temp: np.ndarray) -> np.ndarray:
    """温度矩阵 → 伪彩 BGR（仅可视化）。"""
    lo, hi = np.percentile(temp, [2, 98])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((temp.astype(np.float32) - lo) / (hi - lo), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def draw_overlay(
    vis_bgr: np.ndarray,
    boxes: list,
    mask: Optional[np.ndarray],
    stats: dict,
) -> np.ndarray:
    out = vis_bgr.copy()
    if mask is not None:
        tint = np.zeros_like(out)
        tint[:, :, 2] = (mask * 80).astype(np.uint8)
        out = cv2.addWeighted(out, 1.0, tint, 0.35, 0)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 1)
        label = f"#{i} {box.score:.2f}"
        cv2.putText(
            out, label, (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1, cv2.LINE_AA,
        )

    thr = stats.get("threshold")
    ndet = stats.get("num_detections", len(boxes))
    if thr is not None:
        cv2.putText(
            out,
            f"thr={thr:.0f} det={ndet}",
            (4, IR_HEIGHT - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )
    return out


def run_one(
    temp_path: Optional[Path],
    *,
    synthetic: bool,
    align_roll: bool,
    config_path: Optional[Path],
    out_dir: Path,
    show_mask: bool,
) -> dict:
    if synthetic:
        temp = make_synthetic_temp()
        stem = "synthetic"
    else:
        assert temp_path is not None
        temp = np.load(str(temp_path))
        stem = temp_path.stem

    cfg = load_config(config_path)
    detector = IRThermalDetector(cfg)

    t0 = time.perf_counter()
    boxes = detector.detect(temp, align_roll=align_roll)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    work = temp
    if align_roll:
        work = align_temp_matrix(temp, cfg.roll_x, cfg.roll_y)

    vis = temp_to_vis(work)
    mask = detector.last_mask if show_mask else None
    overlay = draw_overlay(vis, boxes, mask, detector.last_stats)

    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = out_dir / f"{stem}_ir_detect.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    if show_mask and mask is not None:
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), (mask * 255).astype(np.uint8))

    result = {
        "input": str(temp_path) if temp_path else "synthetic",
        "shape": list(temp.shape),
        "align_roll": align_roll,
        "elapsed_ms": round(elapsed_ms, 3),
        "stats": detector.last_stats,
        "detections": [b.to_dict() for b in boxes],
        "overlay": str(overlay_path),
    }

    json_path = out_dir / f"{stem}_ir_detect.json"
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="IR 热目标检测离线调试")
    parser.add_argument("temp_npy", nargs="?", help="温度矩阵 .npy，形状 (192,256) uint16")
    parser.add_argument("-o", "--output", type=Path, default=REPO_ROOT / "output" / "ir_thermal")
    parser.add_argument("-c", "--config", type=Path, default=REPO_ROOT / "config" / "ir_thermal.yaml")
    parser.add_argument("--synthetic", action="store_true", help="使用合成热斑数据验通路")
    parser.add_argument("--align-roll", action="store_true", help="检测前对输入做 roll 对齐")
    parser.add_argument("--no-mask", action="store_true", help="叠加图不绘制 mask")
    args = parser.parse_args(argv)

    if not args.synthetic and not args.temp_npy:
        parser.error("请提供 temp_npy 或使用 --synthetic")

    paths: List[Optional[Path]] = [None] if args.synthetic else []
    if args.temp_npy:
        p = Path(args.temp_npy)
        if p.is_dir():
            paths = sorted(p.glob("*_ir_temp.npy"))
            if not paths:
                paths = sorted(p.glob("*.npy"))
        else:
            paths = [p]

    print(f"config: {args.config}")
    print(f"output: {args.output}")

    for tp in paths:
        res = run_one(
            tp,
            synthetic=args.synthetic,
            align_roll=args.align_roll,
            config_path=args.config,
            out_dir=args.output,
            show_mask=not args.no_mask,
        )
        print(json.dumps(res, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
