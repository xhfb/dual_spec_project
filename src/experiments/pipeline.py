"""论文评测流水线 — 对接仓库真实 API（dual_fusion_web / late_fusion / tracker）。"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.detection.ir_thermal import IRThermalDetector, load_config as load_ir_config
from src.detection.types import DetectionBox
from src.fusion.late_fusion import FusionConfig, FusionResult, fuse_detections, load_fusion_config
from src.fusion.registration import compute_iou, load_homography, project_box
from src.tracking.tracker import TargetTracker, load_tracking_config
from scripts.dual_fusion_web import CONFIG as WEB_CONFIG, _yolo_to_boxes
from utils.Yolov11_infer import YOLOv11Detector

PERSON_CLASS_ID = 0
RGB_CENTER = (320.0, 240.0)
SCORE_THRESH = {"rgb": 0.25, "ir": 0.15, "fused": 0.20}


@dataclass
class EvalConfig:
    mode: str  # rgb | ir | fusion
    no_registration: bool = False
    fixed_weights: Optional[Tuple[float, float]] = None
    no_tracker: bool = False


@dataclass
class TrialResult:
    scenario: str
    mode: str
    trial_id: int
    frames: int
    detection_rate: float
    center_error_px: float
    fps: float
    lost_count: int
    recovery_ms: float
    fusion_match_iou_mean: float
    id_switch_count: int = 0
    env_lux: float = float("nan")
    notes: str = ""
    frame_metrics: List[Dict[str, Any]] = field(default_factory=list)


class EvalPipeline:
    """懒加载检测/融合/跟踪组件。"""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self._yolo: Optional[YOLOv11Detector] = None
        self._ir: Optional[IRThermalDetector] = None
        self._H: Optional[np.ndarray] = None
        self._fusion_cfg: Optional[FusionConfig] = None
        self._tracker: Optional[TargetTracker] = None

    @property
    def yolo(self) -> YOLOv11Detector:
        if self._yolo is None:
            self._yolo = YOLOv11Detector(
                model_path=str(WEB_CONFIG.model_path),
                conf_thresh=WEB_CONFIG.yolo_conf,
                nms_thresh=WEB_CONFIG.yolo_nms,
                cls_num=80,
            )
        return self._yolo

    @property
    def ir_det(self) -> IRThermalDetector:
        if self._ir is None:
            self._ir = IRThermalDetector(load_ir_config(WEB_CONFIG.ir_config))
        return self._ir

    @property
    def H(self) -> np.ndarray:
        if self._H is None:
            self._H = load_homography()
        return self._H

    @property
    def fusion_cfg(self) -> FusionConfig:
        if self._fusion_cfg is None:
            fc = load_fusion_config(WEB_CONFIG.fusion_config)
            if self.cfg.fixed_weights is not None:
                from dataclasses import replace

                wr, wi = self.cfg.fixed_weights
                fc = replace(
                    fc,
                    w_rgb_normal=wr,
                    w_ir_normal=wi,
                    w_rgb_lowlight=wr,
                    w_ir_lowlight=wi,
                )
            self._fusion_cfg = fc
        return self._fusion_cfg

    @property
    def tracker(self) -> TargetTracker:
        if self._tracker is None:
            self._tracker = TargetTracker(load_tracking_config(WEB_CONFIG.tracking_config))
        return self._tracker

    def detect_rgb(self, rgb: np.ndarray) -> List[DetectionBox]:
        boxes, scores, classes = self.yolo.detect(rgb)
        mask = classes == PERSON_CLASS_ID
        return _yolo_to_boxes(boxes[mask], scores[mask])

    def detect_ir(self, temp: np.ndarray) -> List[DetectionBox]:
        return self.ir_det.detect(temp)

    def fuse(self, rgb_boxes: List[DetectionBox], ir_boxes: List[DetectionBox]) -> FusionResult:
        if self.cfg.no_registration:
            return _fuse_without_registration(rgb_boxes, ir_boxes, self.fusion_cfg)
        return fuse_detections(rgb_boxes, ir_boxes, self.H, self.fusion_cfg)

    def boxes_for_mode(
        self,
        rgb_boxes: List[DetectionBox],
        ir_boxes: List[DetectionBox],
    ) -> Tuple[List[DetectionBox], float]:
        if self.cfg.mode == "rgb":
            return [b for b in rgb_boxes if _passes(b, "rgb")], float("nan")
        if self.cfg.mode == "ir":
            projected = [project_box(b, self.H) for b in ir_boxes if _passes(b, "ir")]
            return projected, float("nan")
        result = self.fuse(rgb_boxes, ir_boxes)
        ious = [m.iou for m in result.matches if m.rgb_index is not None and m.ir_index is not None]
        mean_iou = float(np.mean(ious)) if ious else float("nan")
        return [b for b in result.fused_boxes if _passes(b, "fused")], mean_iou

    def primary_from_boxes(self, boxes: List[DetectionBox]) -> Optional[DetectionBox]:
        candidates = [b for b in boxes if _passes(b, self.cfg.mode if self.cfg.mode != "fusion" else "fused")]
        if not candidates:
            return None
        return max(candidates, key=lambda b: b.score * b.width * b.height)

    def process_frame(
        self,
        rgb: np.ndarray,
        temp: np.ndarray,
    ) -> Tuple[Optional[DetectionBox], float, Optional[int], float]:
        rgb_boxes = self.detect_rgb(rgb)
        ir_boxes = self.detect_ir(temp)
        mode_boxes, mean_iou = self.boxes_for_mode(rgb_boxes, ir_boxes)

        primary_box: Optional[DetectionBox] = None
        track_id: Optional[int] = None

        if self.cfg.no_tracker or self.cfg.mode != "fusion":
            primary_box = self.primary_from_boxes(mode_boxes)
        else:
            tr = self.tracker.update(mode_boxes)
            if tr.primary is not None:
                primary_box = tr.primary.box
                track_id = tr.primary_id

        if primary_box is None:
            return None, float("nan"), track_id, mean_iou

        cx, cy = primary_box.center
        err = math.hypot(cx - RGB_CENTER[0], cy - RGB_CENTER[1])
        return primary_box, err, track_id, mean_iou


def _passes(box: DetectionBox, mode: str) -> bool:
    if box.class_id != PERSON_CLASS_ID:
        return False
    thr = SCORE_THRESH.get(mode, 0.2)
    return box.score >= thr


def _fuse_without_registration(
    rgb_boxes: List[DetectionBox],
    ir_boxes: List[DetectionBox],
    cfg: FusionConfig,
) -> FusionResult:
    """消融 A1：IR 框不投影，直接在 RGB 坐标系做 IoU（故意错误对齐）。"""
    from src.fusion.late_fusion import FusionMatch, _fuse_pair, _greedy_match, _scene_weights

    ir_as_rgb = [
        DetectionBox(b.x1, b.y1, b.x2, b.y2, score=b.score, label="person", class_id=0, source="ir")
        for b in ir_boxes
    ]
    w_rgb, w_ir, scene = _scene_weights(rgb_boxes, cfg)
    matched = _greedy_match(rgb_boxes, ir_as_rgb, cfg.match_iou_thresh)
    matches: List[FusionMatch] = []
    fused: List[DetectionBox] = []
    matched_r, matched_i = set(), set()
    for ri, ii, iou in matched:
        rb, ib = rgb_boxes[ri], ir_as_rgb[ii]
        fb = _fuse_pair(rb, ib, w_rgb, w_ir, fuse_coord=cfg.fuse_coord)
        matches.append(FusionMatch(ri, ii, iou, w_rgb, w_ir, fb, rb, ib))
        fused.append(fb)
        matched_r.add(ri)
        matched_i.add(ii)
    for ri, rb in enumerate(rgb_boxes):
        if ri not in matched_r and rb.score >= cfg.rgb_alone_min_score:
            alone = DetectionBox(rb.x1, rb.y1, rb.x2, rb.y2, score=rb.score, label="person", class_id=0, source="rgb")
            fused.append(alone)
    for ii, ib in enumerate(ir_as_rgb):
        if ii not in matched_i and ib.score >= cfg.ir_alone_min_score:
            fused.append(DetectionBox(ib.x1, ib.y1, ib.x2, ib.y2, score=ib.score, label="person", class_id=0, source="ir"))
    return FusionResult(fused_boxes=fused, matches=matches, ir_projected=ir_as_rgb, scene=scene, w_rgb=w_rgb, w_ir=w_ir)


def run_trial(
    capture,
    cfg: EvalConfig,
    scenario: str,
    trial_id: int,
    frames: int = 90,
    env_lux: float = float("nan"),
    notes: str = "",
    on_frame: Optional[
        Callable[
            [int, int, np.ndarray, np.ndarray, Optional[DetectionBox], Dict[str, Any], "EvalPipeline"],
            None,
        ]
    ] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    pipe: Optional["EvalPipeline"] = None,
) -> TrialResult:
    if pipe is None:
        pipe = EvalPipeline(cfg)
    detected = 0
    errors: List[float] = []
    fusion_ious: List[float] = []
    lost = 0
    id_switches = 0
    last_tid: Optional[int] = None
    recovery_ms = float("nan")
    lost_since: Optional[float] = None
    frame_metrics: List[Dict[str, Any]] = []

    start = time.perf_counter()
    for fi in range(frames):
        if should_stop and should_stop():
            break
        bundle = capture.read()
        primary, err, tid, mean_iou = pipe.process_frame(bundle.rgb, bundle.temp)
        fm = {
            "detected": primary is not None,
            "center_error_px": err,
            "fusion_match_iou": mean_iou,
            "track_id": tid,
        }
        frame_metrics.append(fm)
        if on_frame is not None:
            on_frame(fi, frames, bundle.rgb, bundle.temp, primary, fm, pipe)

        if primary is not None:
            detected += 1
            if not math.isnan(err):
                errors.append(err)
            if not math.isnan(mean_iou):
                fusion_ious.append(mean_iou)
            if tid is not None and last_tid is not None and tid != last_tid:
                id_switches += 1
            if tid is not None:
                last_tid = tid
            if lost_since is not None:
                recovery_ms = (time.perf_counter() - lost_since) * 1000.0
                lost_since = None
        else:
            lost += 1
            if lost_since is None:
                lost_since = time.perf_counter()

    duration = max(1e-6, time.perf_counter() - start)
    n_done = len(frame_metrics)
    return TrialResult(
        scenario=scenario,
        mode=cfg.mode,
        trial_id=trial_id,
        frames=n_done,
        detection_rate=detected / max(1, n_done),
        center_error_px=float(np.mean(errors)) if errors else float("nan"),
        fps=n_done / duration,
        lost_count=lost,
        recovery_ms=recovery_ms,
        fusion_match_iou_mean=float(np.mean(fusion_ious)) if fusion_ious else float("nan"),
        id_switch_count=id_switches,
        env_lux=env_lux,
        notes=notes,
        frame_metrics=frame_metrics,
    )


def aggregate_trials(trials: List[TrialResult]) -> Dict[str, Any]:
    from collections import defaultdict

    groups: Dict[Tuple[str, str], List[TrialResult]] = defaultdict(list)
    for t in trials:
        groups[(t.scenario, t.mode)].append(t)

    summary: Dict[str, Any] = {}
    for (scenario, mode), items in groups.items():
        def arr(attr: str) -> List[float]:
            return [getattr(x, attr) for x in items if not math.isnan(getattr(x, attr))]

        summary[f"{scenario}/{mode}"] = {
            "n": len(items),
            "detection_rate_mean": float(np.mean([x.detection_rate for x in items])),
            "detection_rate_std": float(np.std([x.detection_rate for x in items])),
            "center_error_px_mean": float(np.mean(arr("center_error_px"))) if arr("center_error_px") else None,
            "fps_mean": float(np.mean([x.fps for x in items])),
            "lost_count_mean": float(np.mean([x.lost_count for x in items])),
            "fusion_match_iou_mean": float(np.mean(arr("fusion_match_iou_mean"))) if arr("fusion_match_iou_mean") else None,
        }
    return summary
