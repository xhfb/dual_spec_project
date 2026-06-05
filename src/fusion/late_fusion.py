"""Late Fusion：IR 框投影 + RGB/IR 框级 IoU 关联 + 场景自适应加权融合。"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    from src.detection.types import DetectionBox
    from src.fusion.registration import compute_iou, project_box
except Exception:  # pragma: no cover
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.detection.types import DetectionBox  # type: ignore
    from src.fusion.registration import compute_iou, project_box  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "fusion.yaml"


@dataclass
class FusionConfig:
    match_iou_thresh: float = 0.25
    w_rgb_normal: float = 0.70
    w_ir_normal: float = 0.30
    w_rgb_lowlight: float = 0.25
    w_ir_lowlight: float = 0.75
    rgb_conf_high: float = 0.45
    rgb_conf_low: float = 0.20
    rgb_alone_min_score: float = 0.35
    ir_alone_min_score: float = 0.40
    fuse_coord: bool = True
    max_fused: int = 5

    @classmethod
    def from_dict(cls, data: dict) -> "FusionConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def load_fusion_config(path: Union[str, Path, None] = None) -> FusionConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return FusionConfig()
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        parsed = yaml.safe_load(text) or {}
        data = parsed.get("fusion", parsed)
    else:
        data = json.loads(text).get("fusion", {})
    return FusionConfig.from_dict(data)


@dataclass
class FusionMatch:
    """单条融合关联记录。"""

    rgb_index: Optional[int]
    ir_index: Optional[int]
    iou: float
    w_rgb: float
    w_ir: float
    fused: DetectionBox
    rgb_box: Optional[DetectionBox] = None
    ir_proj_box: Optional[DetectionBox] = None

    def to_dict(self) -> dict:
        return {
            "rgb_index": self.rgb_index,
            "ir_index": self.ir_index,
            "iou": round(self.iou, 4),
            "w_rgb": round(self.w_rgb, 3),
            "w_ir": round(self.w_ir, 3),
            "fused": self.fused.to_dict(),
            "rgb_box": self.rgb_box.to_dict() if self.rgb_box else None,
            "ir_proj_box": self.ir_proj_box.to_dict() if self.ir_proj_box else None,
        }


@dataclass
class FusionResult:
    fused_boxes: List[DetectionBox] = field(default_factory=list)
    matches: List[FusionMatch] = field(default_factory=list)
    ir_projected: List[DetectionBox] = field(default_factory=list)
    scene: str = "normal"
    w_rgb: float = 0.7
    w_ir: float = 0.3

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "w_rgb": self.w_rgb,
            "w_ir": self.w_ir,
            "fused_count": len(self.fused_boxes),
            "matches": [m.to_dict() for m in self.matches],
            "fused_boxes": [b.to_dict() for b in self.fused_boxes],
            "ir_projected": [b.to_dict() for b in self.ir_projected],
        }


def _scene_weights(
    rgb_boxes: List[DetectionBox],
    cfg: FusionConfig,
) -> Tuple[float, float, str]:
    if not rgb_boxes:
        return cfg.w_rgb_lowlight, cfg.w_ir_lowlight, "lowlight"
    top_rgb = max(b.score for b in rgb_boxes)
    if top_rgb >= cfg.rgb_conf_high:
        return cfg.w_rgb_normal, cfg.w_ir_normal, "normal"
    if top_rgb <= cfg.rgb_conf_low:
        return cfg.w_rgb_lowlight, cfg.w_ir_lowlight, "lowlight"
    t = (top_rgb - cfg.rgb_conf_low) / max(cfg.rgb_conf_high - cfg.rgb_conf_low, 1e-6)
    w_rgb = cfg.w_rgb_lowlight + t * (cfg.w_rgb_normal - cfg.w_rgb_lowlight)
    w_ir = cfg.w_ir_lowlight + t * (cfg.w_ir_normal - cfg.w_ir_lowlight)
    return w_rgb, w_ir, "mixed"


def _fuse_pair(
    rgb_box: DetectionBox,
    ir_proj: DetectionBox,
    w_rgb: float,
    w_ir: float,
    *,
    fuse_coord: bool,
) -> DetectionBox:
    ws = w_rgb + w_ir
    if ws <= 0:
        ws = 1.0
    wr, wi = w_rgb / ws, w_ir / ws
    if fuse_coord:
        x1 = wr * rgb_box.x1 + wi * ir_proj.x1
        y1 = wr * rgb_box.y1 + wi * ir_proj.y1
        x2 = wr * rgb_box.x2 + wi * ir_proj.x2
        y2 = wr * rgb_box.y2 + wi * ir_proj.y2
    else:
        x1, y1, x2, y2 = rgb_box.x1, rgb_box.y1, rgb_box.x2, rgb_box.y2
    score = wr * rgb_box.score + wi * ir_proj.score
    return DetectionBox(
        x1=x1, y1=y1, x2=x2, y2=y2,
        score=float(np.clip(score, 0, 1)),
        label="person",
        class_id=0,
        source="fused",
    )


def _greedy_match(
    rgb_boxes: List[DetectionBox],
    ir_proj: List[DetectionBox],
    iou_thresh: float,
) -> List[Tuple[int, int, float]]:
    """返回 (rgb_idx, ir_idx, iou) 列表，一对一贪心最大 IoU。"""
    pairs: List[Tuple[int, int, float]] = []
    for ri, rb in enumerate(rgb_boxes):
        for ii, ib in enumerate(ir_proj):
            iou = compute_iou(rb, ib)
            if iou >= iou_thresh:
                pairs.append((ri, ii, iou))
    pairs.sort(key=lambda x: x[2], reverse=True)
    used_r: set = set()
    used_i: set = set()
    out: List[Tuple[int, int, float]] = []
    for ri, ii, iou in pairs:
        if ri in used_r or ii in used_i:
            continue
        used_r.add(ri)
        used_i.add(ii)
        out.append((ri, ii, iou))
    return out


def fuse_detections(
    rgb_boxes: List[DetectionBox],
    ir_boxes: List[DetectionBox],
    H: np.ndarray,
    config: Union[FusionConfig, str, Path, None] = None,
) -> FusionResult:
    """Late Fusion 主入口。

    Args:
        rgb_boxes: RGB 平面 YOLO 人体框。
        ir_boxes: IR 坐标系热检测框。
        H: IR→RGB 单应矩阵。
        config: 融合配置。
    """
    if isinstance(config, FusionConfig):
        cfg = config
    elif config is not None:
        cfg = load_fusion_config(config)
    else:
        cfg = FusionConfig()

    ir_projected = [project_box(b, H) for b in ir_boxes]
    for b in ir_projected:
        b.source = "ir_proj"

    w_rgb, w_ir, scene = _scene_weights(rgb_boxes, cfg)
    matched = _greedy_match(rgb_boxes, ir_projected, cfg.match_iou_thresh)

    matches: List[FusionMatch] = []
    fused: List[DetectionBox] = []
    matched_r = set()
    matched_i = set()

    for ri, ii, iou in matched:
        rb = rgb_boxes[ri]
        ib = ir_projected[ii]
        fb = _fuse_pair(rb, ib, w_rgb, w_ir, fuse_coord=cfg.fuse_coord)
        matches.append(FusionMatch(
            rgb_index=ri, ir_index=ii, iou=iou,
            w_rgb=w_rgb, w_ir=w_ir, fused=fb,
            rgb_box=rb, ir_proj_box=ib,
        ))
        fused.append(fb)
        matched_r.add(ri)
        matched_i.add(ii)

    for ri, rb in enumerate(rgb_boxes):
        if ri not in matched_r and rb.score >= cfg.rgb_alone_min_score:
            alone = DetectionBox(
                rb.x1, rb.y1, rb.x2, rb.y2,
                score=rb.score, label="person", class_id=0, source="rgb",
            )
            fused.append(alone)
            matches.append(FusionMatch(
                rgb_index=ri, ir_index=None, iou=0.0,
                w_rgb=1.0, w_ir=0.0, fused=alone, rgb_box=rb,
            ))

    for ii, ib in enumerate(ir_projected):
        if ii not in matched_i and ib.score >= cfg.ir_alone_min_score:
            alone = DetectionBox(
                ib.x1, ib.y1, ib.x2, ib.y2,
                score=ib.score, label="person", class_id=0, source="ir",
            )
            fused.append(alone)
            matches.append(FusionMatch(
                rgb_index=None, ir_index=ii, iou=0.0,
                w_rgb=0.0, w_ir=1.0, fused=alone, ir_proj_box=ib,
            ))

    fused.sort(key=lambda b: b.score * b.width * b.height, reverse=True)
    if cfg.max_fused > 0:
        fused = fused[: cfg.max_fused]

    return FusionResult(
        fused_boxes=fused,
        matches=matches,
        ir_projected=ir_projected,
        scene=scene,
        w_rgb=w_rgb,
        w_ir=w_ir,
    )
