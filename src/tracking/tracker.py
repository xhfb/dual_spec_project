"""IoU 多目标跟踪 + 主目标锁定。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

from src.detection.types import DetectionBox
from src.fusion.registration import compute_iou
from src.tracking.selector import score_detection
from src.tracking.types import Track, TrackingResult

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "tracking.yaml"


@dataclass
class TrackingConfig:
    frame_width: int = 640
    frame_height: int = 480
    w_confidence: float = 0.50
    w_area: float = 0.20
    w_center: float = 0.25
    w_fused_bonus: float = 0.05
    match_iou: float = 0.30
    max_age: int = 12
    min_hits: int = 2
    ema_alpha: float = 0.55
    primary_lock_frames: int = 8
    primary_switch_margin: float = 0.12

    @classmethod
    def from_dict(cls, data: dict) -> "TrackingConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def load_tracking_config(path: Union[str, Path, None] = None) -> TrackingConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return TrackingConfig()
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        parsed = yaml.safe_load(text) or {}
        data = parsed.get("tracking", parsed)
    else:
        data = json.loads(text).get("tracking", {})
    return TrackingConfig.from_dict(data)


def _smooth_box(prev: DetectionBox, det: DetectionBox, alpha: float) -> DetectionBox:
    a, b = alpha, 1.0 - alpha
    return DetectionBox(
        x1=a * det.x1 + b * prev.x1,
        y1=a * det.y1 + b * prev.y1,
        x2=a * det.x2 + b * prev.x2,
        y2=a * det.y2 + b * prev.y2,
        score=a * det.score + b * prev.score,
        label=det.label or prev.label,
        class_id=det.class_id,
        source=det.source or prev.source,
    )


def _score_weights(cfg: TrackingConfig) -> dict:
    return {
        "w_confidence": cfg.w_confidence,
        "w_area": cfg.w_area,
        "w_center": cfg.w_center,
        "w_fused_bonus": cfg.w_fused_bonus,
    }


class TargetTracker:
    """融合框跟踪器：维护轨迹并输出主目标。"""

    def __init__(self, config: Optional[TrackingConfig] = None) -> None:
        self.config = config or TrackingConfig()
        self._tracks: List[Track] = []
        self._next_id = 1
        self._primary_id: Optional[int] = None
        self._primary_lock: int = 0

    def reset(self) -> None:
        self._tracks.clear()
        self._primary_id = None
        self._primary_lock = 0

    def _match_detections(
        self,
        detections: List[DetectionBox],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """返回 (track_idx, det_idx) 匹配、未匹配 track、未匹配 det。"""
        cfg = self.config
        if not self._tracks or not detections:
            return [], list(range(len(self._tracks))), list(range(len(detections)))

        pairs: List[Tuple[int, int, float]] = []
        for ti, tr in enumerate(self._tracks):
            for di, det in enumerate(detections):
                iou = compute_iou(tr.box, det)
                if iou >= cfg.match_iou:
                    pairs.append((ti, di, iou))
        pairs.sort(key=lambda x: x[2], reverse=True)

        used_t: set = set()
        used_d: set = set()
        matches: List[Tuple[int, int]] = []
        for ti, di, _ in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            matches.append((ti, di))

        unmatched_t = [i for i in range(len(self._tracks)) if i not in used_t]
        unmatched_d = [i for i in range(len(detections)) if i not in used_d]
        return matches, unmatched_t, unmatched_d

    def _update_primary(self, detections: List[DetectionBox]) -> None:
        cfg = self.config
        weights = _score_weights(cfg)
        fw, fh = float(cfg.frame_width), float(cfg.frame_height)

        det_scores = [
            score_detection(d, fw, fh, **weights) for d in detections
        ]

        # 当前主目标仍活跃：续锁
        if self._primary_id is not None:
            primary_track = next(
                (t for t in self._tracks if t.track_id == self._primary_id), None
            )
            if primary_track and primary_track.time_since_update == 0:
                self._primary_lock = cfg.primary_lock_frames
                return

        if self._primary_lock > 0:
            self._primary_lock -= 1
            if self._primary_id is not None:
                still = any(t.track_id == self._primary_id for t in self._tracks)
                if still:
                    return

        # 在已确认轨迹中选最高分
        candidates = [
            t for t in self._tracks
            if t.hits >= cfg.min_hits and t.time_since_update == 0
        ]
        if not candidates:
            candidates = [t for t in self._tracks if t.time_since_update == 0]
        if not candidates:
            self._primary_id = None
            return

        for t in candidates:
            t.primary_score = score_detection(t.box, fw, fh, **weights)

        best = max(candidates, key=lambda t: t.primary_score)
        if self._primary_id is None:
            self._primary_id = best.track_id
            self._primary_lock = cfg.primary_lock_frames
            return

        current = next((t for t in candidates if t.track_id == self._primary_id), None)
        if current is None:
            self._primary_id = best.track_id
            self._primary_lock = cfg.primary_lock_frames
            return

        if (
            best.track_id != self._primary_id
            and best.primary_score >= current.primary_score + cfg.primary_switch_margin
        ):
            self._primary_id = best.track_id
            self._primary_lock = cfg.primary_lock_frames

    def update(self, detections: List[DetectionBox]) -> TrackingResult:
        """输入本帧融合框，返回轨迹与主目标。"""
        cfg = self.config
        matches, unmatched_t, unmatched_d = self._match_detections(detections)

        for ti, di in matches:
            tr = self._tracks[ti]
            tr.box = _smooth_box(tr.box, detections[di], cfg.ema_alpha)
            tr.hits += 1
            tr.age += 1
            tr.time_since_update = 0

        for ti in unmatched_t:
            tr = self._tracks[ti]
            tr.age += 1
            tr.time_since_update += 1

        for di in unmatched_d:
            det = detections[di]
            self._tracks.append(Track(
                track_id=self._next_id,
                box=DetectionBox(
                    det.x1, det.y1, det.x2, det.y2,
                    score=det.score, label=det.label,
                    class_id=det.class_id, source=det.source,
                ),
                hits=1,
                age=1,
            ))
            self._next_id += 1

        self._tracks = [
            t for t in self._tracks
            if t.time_since_update <= cfg.max_age
        ]

        self._update_primary(detections)

        for t in self._tracks:
            t.is_primary = t.track_id == self._primary_id

        primary = next((t for t in self._tracks if t.is_primary), None)
        return TrackingResult(
            tracks=list(self._tracks),
            primary=primary,
            primary_id=self._primary_id,
        )
