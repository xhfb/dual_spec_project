"""主目标选择：从融合框中打分选出当前帧最优候选。"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from src.detection.types import DetectionBox


def score_detection(
    box: DetectionBox,
    frame_w: float,
    frame_h: float,
    *,
    w_confidence: float = 0.50,
    w_area: float = 0.20,
    w_center: float = 0.25,
    w_fused_bonus: float = 0.05,
) -> float:
    """可解释主目标得分，越大越优先。"""
    cx, cy = box.center
    fcx, fcy = frame_w / 2.0, frame_h / 2.0
    dist = math.hypot(cx - fcx, cy - fcy)
    max_dist = math.hypot(fcx, fcy) + 1e-6
    center_term = 1.0 - min(dist / max_dist, 1.0)

    area_norm = (box.width * box.height) / max(frame_w * frame_h, 1.0)
    area_term = min(area_norm * 8.0, 1.0)  # 人体约占画面 5~15%

    bonus = w_fused_bonus if box.source == "fused" else 0.0
    return (
        w_confidence * box.score
        + w_area * area_term
        + w_center * center_term
        + bonus
    )


def select_best_index(
    boxes: List[DetectionBox],
    frame_w: float,
    frame_h: float,
    **weights,
) -> Tuple[Optional[int], float]:
    """返回 (最佳索引, 得分)；无框时 (-1, 0)。"""
    if not boxes:
        return None, 0.0
    scores = [score_detection(b, frame_w, frame_h, **weights) for b in boxes]
    best_i = int(max(range(len(scores)), key=lambda i: scores[i]))
    return best_i, scores[best_i]
