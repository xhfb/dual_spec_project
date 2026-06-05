"""IR 路热目标检测：温度矩阵 → 检测框（免训练）。

算法：自适应温度阈值 → 二值化 → 形态学 → 连通域 → 几何/温差滤波 → bbox。

坐标约定：输出框处于 256×192、roll 已对齐的 IR 像素坐标系，
与 ``config/homography_meta.json`` 标定一致。不做 IR→RGB 投影。
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

try:
    from src.detection.types import DetectionBox
except Exception:  # pragma: no cover
    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.detection.types import DetectionBox  # type: ignore

IR_HEIGHT = 192
IR_WIDTH = 256
IR_SHAPE = (IR_HEIGHT, IR_WIDTH)

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "ir_thermal.yaml"
DEFAULT_META_PATH = _REPO_ROOT / "config" / "homography_meta.json"
DEFAULT_DEBUG_JSON = _REPO_ROOT / "uvc_ubuntu" / "thermal_cam_debug.json"


@dataclass
class IRThermalConfig:
    """IR 热检测可调参数。"""

    threshold_method: str = "otsu"
    percentile: float = 99.0
    mean_std_k: float = 2.5
    bg_percentile: float = 60.0
    fixed_delta: float = 800.0

    otsu_fallback: str = "percentile"
    thr_clamp_low_percentile: float = 50.0
    thr_clamp_high_percentile: float = 99.5

    morph_open: int = 3
    morph_close: int = 5

    min_area: int = 80
    max_area: int = 12000
    min_aspect: float = 0.25
    max_aspect: float = 4.0
    min_temp_delta_ratio: float = 0.15

    reject_border: bool = True
    border_margin: int = 2
    border_max_area: int = 500

    crop_top: int = 8
    crop_bottom: int = 8
    edge_mask_margin: int = 0

    max_detections: int = 5
    nms_iou: float = 0.5
    class_id: int = 0

    roll_x: int = -164
    roll_y: int = -2

    @classmethod
    def from_dict(cls, data: dict) -> "IRThermalConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def load_config(path: Union[str, Path, None] = None) -> IRThermalConfig:
    """从 YAML/JSON 加载 ``ir_thermal`` 配置段。"""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return IRThermalConfig()

    text = cfg_path.read_text(encoding="utf-8")
    data: dict
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        parsed = yaml.safe_load(text) or {}
        data = parsed.get("ir_thermal", parsed)
    else:
        parsed = json.loads(text)
        data = parsed.get("ir_thermal", parsed)

    return IRThermalConfig.from_dict(data)


def load_roll_from_meta(
    meta_path: Union[str, Path, None] = None,
    debug_json: Union[str, Path, None] = None,
) -> Tuple[int, int]:
    """读取标定/调试配置中的 roll 偏移，与 web_temp_viewer 一致。"""
    roll_x, roll_y = -164, -2

    meta_p = Path(meta_path) if meta_path else DEFAULT_META_PATH
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            xy = meta.get("ir_roll_xy")
            if xy and len(xy) == 2:
                roll_x, roll_y = int(xy[0]), int(xy[1])
                return roll_x, roll_y
        except Exception:
            pass

    dbg_p = Path(debug_json) if debug_json else DEFAULT_DEBUG_JSON
    if dbg_p.exists():
        try:
            cfg = json.loads(dbg_p.read_text(encoding="utf-8"))
            align = cfg.get("align") or {}
            if "roll_x" in align:
                roll_x = int(align["roll_x"])
            if "roll_y" in align:
                roll_y = int(align["roll_y"])
        except Exception:
            pass

    return roll_x, roll_y


def apply_row_crop(
    image: np.ndarray,
    top: int,
    bottom: int,
    fill,
) -> np.ndarray:
    """将上下各 ``top``/``bottom`` 行置为 ``fill``（保持 256×192 坐标不变）。"""
    if top <= 0 and bottom <= 0:
        return image
    out = image.copy()
    if top > 0:
        out[:top, ...] = fill
    if bottom > 0:
        out[-bottom:, ...] = fill
    return out


def align_temp_matrix(
    temp_matrix: np.ndarray,
    roll_x: int = 0,
    roll_y: int = 0,
) -> np.ndarray:
    """对温度矩阵做与标定一致的 np.roll 对齐。"""
    out = np.asarray(temp_matrix, dtype=np.uint16)
    if roll_y:
        out = np.roll(out, roll_y, axis=0)
    if roll_x:
        out = np.roll(out, roll_x, axis=1)
    return out


def _validate_temp_matrix(temp_matrix: np.ndarray) -> Optional[np.ndarray]:
    if temp_matrix is None:
        return None
    arr = np.asarray(temp_matrix)
    if arr.size == 0:
        return None
    if arr.shape != IR_SHAPE:
        return None
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.uint16)
    else:
        arr = arr.astype(np.uint16, copy=False)
    return arr


def _otsu_threshold(values: np.ndarray) -> float:
    """在帧内动态范围上计算 Otsu 阈值（16-bit 友好）。"""
    flat = values.ravel().astype(np.float64)
    vmin, vmax = float(flat.min()), float(flat.max())
    if vmax <= vmin:
        return vmin

    hist, _ = np.histogram(flat, bins=256, range=(vmin, vmax))
    total = flat.size
    sum_total = np.dot(np.arange(256), hist)

    sum_b = 0.0
    w_b = 0.0
    max_var = -1.0
    threshold_bin = 0

    for i in range(256):
        w_b += hist[i]
        if w_b == 0:
            continue
        w_f = total - w_b
        if w_f == 0:
            break
        sum_b += i * hist[i]
        m_b = sum_b / w_b
        m_f = (sum_total - sum_b) / w_f
        var_between = w_b * w_f * (m_b - m_f) ** 2
        if var_between > max_var:
            max_var = var_between
            threshold_bin = i

    frac = threshold_bin / 255.0
    return vmin + frac * (vmax - vmin)


def _frame_stats(flat: np.ndarray, cfg: IRThermalConfig) -> dict:
    """一次遍历求帧内常用分位数与均值方差。"""
    pct_list = sorted(
        {
            50.0,
            95.0,
            float(cfg.percentile),
            float(cfg.bg_percentile),
            float(cfg.thr_clamp_low_percentile),
            float(cfg.thr_clamp_high_percentile),
        }
    )
    pct_vals = np.percentile(flat, pct_list)

    def p(q: float) -> float:
        idx = pct_list.index(q)
        return float(pct_vals[idx])

    return {
        "p50": p(50.0),
        "p95": p(95.0),
        "p_thr": p(float(cfg.percentile)),
        "p_bg": p(float(cfg.bg_percentile)),
        "thr_lo": p(float(cfg.thr_clamp_low_percentile)),
        "thr_hi": p(float(cfg.thr_clamp_high_percentile)),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
        "t_min": float(flat.min()),
        "t_max": float(flat.max()),
    }


def _compute_threshold(
    temp: np.ndarray,
    cfg: IRThermalConfig,
    frame: dict,
) -> float:
    """按配置策略求自适应阈值，含 Otsu 回退与 clamp。"""
    method = cfg.threshold_method.lower()
    thr: float

    if method == "percentile":
        thr = frame["p_thr"]
    elif method == "mean_std":
        thr = frame["mean"] + cfg.mean_std_k * frame["std"]
    elif method == "fixed_offset":
        thr = frame["p_bg"] + cfg.fixed_delta
    else:  # otsu (default)
        thr = _otsu_threshold(temp)
        edge_eps = max(1.0, (frame["thr_hi"] - frame["thr_lo"]) * 0.02)
        otsu_failed = thr <= frame["thr_lo"] + edge_eps or thr >= frame["thr_hi"] - edge_eps
        if otsu_failed:
            fb = cfg.otsu_fallback.lower()
            if fb == "mean_std":
                thr = frame["mean"] + cfg.mean_std_k * frame["std"]
            else:
                thr = frame["p_thr"]

    thr = float(np.clip(thr, frame["thr_lo"], frame["thr_hi"]))
    frame["threshold"] = thr
    frame["temp_range"] = frame["t_max"] - frame["t_min"]
    return thr


def _morph_steps(
    mask: np.ndarray, cfg: IRThermalConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 (二值 mask, 开运算后, 闭运算后)。"""
    binary = mask.astype(np.uint8)
    after_open = binary
    if cfg.morph_open > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_open, cfg.morph_open)
        )
        after_open = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    after_close = after_open
    if cfg.morph_close > 1:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_close, cfg.morph_close)
        )
        after_close = cv2.morphologyEx(after_open, cv2.MORPH_CLOSE, k)
    return binary, after_open, after_close


def _morph_mask(mask: np.ndarray, cfg: IRThermalConfig) -> np.ndarray:
    _, _, after_close = _morph_steps(mask, cfg)
    return after_close


def _labels_to_bgr(labels: np.ndarray, num_labels: int) -> np.ndarray:
    """连通域标签图 → 伪彩 BGR（0=黑，各域不同色）。"""
    if num_labels <= 1:
        return np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)
    vis = (labels.astype(np.uint32) * 9973) % 255
    vis = vis.astype(np.uint8)
    bgr = cv2.applyColorMap(vis, cv2.COLORMAP_TURBO)
    bgr[labels == 0] = 0
    return bgr


def _box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _nms_boxes(
    boxes: List[DetectionBox],
    iou_thresh: float,
) -> List[DetectionBox]:
    if len(boxes) <= 1:
        return boxes
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: List[DetectionBox] = []
    for cand in sorted_boxes:
        cand_xy = (int(cand.x1), int(cand.y1), int(cand.x2), int(cand.y2))
        suppress = False
        for k in kept:
            k_xy = (int(k.x1), int(k.y1), int(k.x2), int(k.y2))
            if _box_iou(cand_xy, k_xy) > iou_thresh:
                suppress = True
                break
        if not suppress:
            kept.append(cand)
    return kept


def _touches_border(
    x1: int, y1: int, x2: int, y2: int,
    w: int, h: int, margin: int,
) -> bool:
    return (
        x1 <= margin
        or y1 <= margin
        or x2 >= w - 1 - margin
        or y2 >= h - 1 - margin
    )


@dataclass
class IRThermalDetector:
    """IR 热目标检测器。"""

    config: IRThermalConfig = field(default_factory=IRThermalConfig)
    last_mask: Optional[np.ndarray] = field(default=None, init=False)
    last_mask_binary: Optional[np.ndarray] = field(default=None, init=False)
    last_mask_open: Optional[np.ndarray] = field(default=None, init=False)
    last_labels: Optional[np.ndarray] = field(default=None, init=False)
    last_labels_vis: Optional[np.ndarray] = field(default=None, init=False)
    last_stats: dict = field(default_factory=dict, init=False)

    @classmethod
    def from_config_path(cls, path: Union[str, Path, None] = None) -> "IRThermalDetector":
        return cls(load_config(path))

    def detect(
        self,
        temp_matrix: np.ndarray,
        ir_vis: Optional[np.ndarray] = None,  # noqa: ARG002 — 仅调试预留
        *,
        align_roll: bool = False,
    ) -> List[DetectionBox]:
        """从 roll 对齐后的温度矩阵检出热目标框。

        Args:
            temp_matrix: (192, 256) uint16 温度/辐射矩阵。
            ir_vis: 可选可视化图，本函数不使用，供上层 debug 叠加。
            align_roll: 为 True 时按配置 roll 再检测（输入为原始未 roll 矩阵时用）。

        Returns:
            0~N 个 ``DetectionBox``（source="ir"）；异常输入返回 []。
        """
        _ = ir_vis
        cfg = self.config

        arr = _validate_temp_matrix(temp_matrix)
        if arr is None:
            self.last_mask = None
            self.last_mask_binary = None
            self.last_mask_open = None
            self.last_labels = None
            self.last_labels_vis = None
            self.last_stats = {"error": "invalid_input"}
            return []

        if align_roll:
            arr = align_temp_matrix(arr, cfg.roll_x, cfg.roll_y)

        work = arr.copy()
        if cfg.crop_top > 0 or cfg.crop_bottom > 0:
            bg_val = int(np.percentile(work, 10))
            work = apply_row_crop(work, cfg.crop_top, cfg.crop_bottom, bg_val)
        if cfg.edge_mask_margin > 0:
            m = cfg.edge_mask_margin
            edge_val = int(np.percentile(work, 10))
            work[:, :m] = edge_val
            work[:, -m:] = edge_val

        flat = work.ravel().astype(np.float64)
        frame = _frame_stats(flat, cfg)
        thr = _compute_threshold(work, cfg, frame)
        frame["threshold_method"] = cfg.threshold_method
        mask_binary, mask_open, mask = _morph_steps((work > thr).astype(np.uint8), cfg)

        self.last_mask_binary = mask_binary
        self.last_mask_open = mask_open
        self.last_mask = mask

        if mask.sum() == 0:
            self.last_labels = None
            self.last_labels_vis = None
            self.last_stats = {
                **frame,
                "num_components": 0,
                "num_detections": 0,
                "filter_rejected": 0,
            }
            return []

        num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        self.last_labels = labels
        self.last_labels_vis = _labels_to_bgr(labels, num_labels)

        t_min = frame["t_min"]
        t_max = frame["t_max"]
        t_bg = frame["p50"]
        temp_span = max(frame["p95"] - frame["p50"], 1.0)

        candidates: List[DetectionBox] = []
        rejected = 0

        for label_id in range(1, num_labels):
            x = int(cc_stats[label_id, cv2.CC_STAT_LEFT])
            y = int(cc_stats[label_id, cv2.CC_STAT_TOP])
            w = int(cc_stats[label_id, cv2.CC_STAT_WIDTH])
            h = int(cc_stats[label_id, cv2.CC_STAT_HEIGHT])
            area = int(cc_stats[label_id, cv2.CC_STAT_AREA])

            if area < cfg.min_area or area > cfg.max_area:
                rejected += 1
                continue

            x1, y1 = x, y
            x2, y2 = x + w, y + h
            aspect = w / max(h, 1)
            if aspect < cfg.min_aspect or aspect > cfg.max_aspect:
                rejected += 1
                continue

            if cfg.reject_border and area <= cfg.border_max_area:
                if _touches_border(x1, y1, x2, y2, IR_WIDTH, IR_HEIGHT, cfg.border_margin):
                    rejected += 1
                    continue

            region = work[labels == label_id]
            t_peak = float(region.max())
            delta_ratio = (t_peak - t_bg) / temp_span
            if delta_ratio < cfg.min_temp_delta_ratio:
                rejected += 1
                continue

            conf = (t_peak - t_bg) / (t_max - t_min + 1e-6)
            conf = float(np.clip(conf, 0.0, 1.0))

            candidates.append(
                DetectionBox(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    score=conf,
                    label="person",
                    class_id=cfg.class_id,
                    source="ir",
                )
            )

        candidates.sort(key=lambda b: b.score * b.width * b.height, reverse=True)
        candidates = _nms_boxes(candidates, cfg.nms_iou)
        if cfg.max_detections > 0:
            candidates = candidates[: cfg.max_detections]

        self.last_stats = {
            **frame,
            "num_components": num_labels - 1,
            "num_detections": len(candidates),
            "filter_rejected": rejected,
            "t_bg": t_bg,
            "temp_span": temp_span,
            "morph_open": cfg.morph_open,
            "morph_close": cfg.morph_close,
            "min_area": cfg.min_area,
            "max_area": cfg.max_area,
            "min_aspect": cfg.min_aspect,
            "max_aspect": cfg.max_aspect,
            "min_temp_delta_ratio": cfg.min_temp_delta_ratio,
            "nms_iou": cfg.nms_iou,
            "max_detections": cfg.max_detections,
            "crop_top": cfg.crop_top,
            "crop_bottom": cfg.crop_bottom,
        }
        return candidates


def detect(
    temp_matrix: np.ndarray,
    ir_vis: Optional[np.ndarray] = None,
    *,
    config: Union[IRThermalConfig, str, Path, None] = None,
    align_roll: bool = False,
) -> List[DetectionBox]:
    """模块级便捷接口。"""
    if isinstance(config, IRThermalConfig):
        det = IRThermalDetector(config)
    elif config is not None:
        det = IRThermalDetector.from_config_path(config)
    else:
        det = IRThermalDetector()
    return det.detect(temp_matrix, ir_vis, align_roll=align_roll)
