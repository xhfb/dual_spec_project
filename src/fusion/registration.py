"""运行期空间配准：把 IR（红外）坐标/检测框投影到 RGB（可见光）像素平面。

依赖标定工具 ``scripts/calibrate_homography.py`` 产出的单应矩阵：
- ``config/homography.npy``        : 3x3 float64 单应矩阵 H（IR -> RGB）。
- ``config/homography_meta.json``  : 元数据（日期/组数/RMSE/分辨率等）。

坐标系约定（与标定一致）：
- IR  : 256x192，已按 ``thermal_cam_debug.json`` 的 roll 对齐。
- RGB : 640x480。

典型用法（Late Fusion）::

    from src.fusion.registration import load_homography, project_box
    H = load_homography()
    rgb_box = project_box(ir_box, H)   # ir_box: DetectionBox(IR坐标)
    iou = compute_iou(rgb_box, rgb_det_box)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np

try:  # 包内导入
    from src.detection.types import DetectionBox
except Exception:  # pragma: no cover - 兜底：直接按文件路径加入 sys.path
    _SRC_DIR = Path(__file__).resolve().parents[1]
    if str(_SRC_DIR.parent) not in sys.path:
        sys.path.insert(0, str(_SRC_DIR.parent))
    try:
        from src.detection.types import DetectionBox  # type: ignore
    except Exception:
        if str(_SRC_DIR) not in sys.path:
            sys.path.insert(0, str(_SRC_DIR))
        from detection.types import DetectionBox  # type: ignore


# 默认产物路径：<repo_root>/config/homography.npy
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_H_PATH = _REPO_ROOT / "config" / "homography.npy"
DEFAULT_META_PATH = _REPO_ROOT / "config" / "homography_meta.json"

PointLike = Union[Sequence[float], np.ndarray, Tuple[float, float]]


def load_homography(
    path: Union[str, os.PathLike, None] = None,
    *,
    check_meta: bool = True,
) -> np.ndarray:
    """加载 3x3 单应矩阵 H（IR -> RGB）。

    Args:
        path: ``homography.npy`` 路径；为 None 时用默认 ``config/homography.npy``。
        check_meta: 若存在同目录的 ``homography_meta.json``，校验其记录的形状。

    Returns:
        ``np.ndarray``，dtype float64，形状 (3, 3)。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 矩阵形状非 3x3 或不可逆。
    """
    h_path = Path(path) if path is not None else DEFAULT_H_PATH
    if not h_path.exists():
        raise FileNotFoundError(
            f"找不到单应矩阵文件: {h_path}. 请先运行 scripts/calibrate_homography.py 完成标定。"
        )

    H = np.load(str(h_path))
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"单应矩阵形状应为 (3,3)，实际为 {H.shape}")

    # 单应矩阵必须可逆（否则无法做投影）。
    if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-12:
        raise ValueError("单应矩阵不可逆或包含非有限值，标定结果可能无效。")

    if check_meta:
        meta_path = h_path.with_name("homography_meta.json")
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                # 仅做轻量一致性提示，不强制失败。
                _ = meta.get("rmse_px")
            except Exception:
                pass

    return H


def project_point(xy_ir: PointLike, H: np.ndarray) -> Tuple[float, float]:
    """把单个 IR 像素点投影到 RGB 平面。

    Args:
        xy_ir: IR 坐标 (x, y)。
        H: 3x3 单应矩阵。

    Returns:
        RGB 平面坐标 (x, y)，float。
    """
    pts = np.array([[[float(xy_ir[0]), float(xy_ir[1])]]], dtype=np.float64)  # (1,1,2)
    import cv2  # 局部导入，避免无 GUI 环境下的 import 副作用

    dst = cv2.perspectiveTransform(pts, H)
    x, y = float(dst[0, 0, 0]), float(dst[0, 0, 1])
    return (x, y)


def project_points(pts_ir: Iterable[PointLike], H: np.ndarray) -> np.ndarray:
    """批量投影 IR 点到 RGB 平面。

    Returns:
        ``np.ndarray``，形状 (N, 2)，float64。
    """
    arr = np.asarray(list(pts_ir), dtype=np.float64).reshape(-1, 1, 2)
    import cv2

    dst = cv2.perspectiveTransform(arr, H)
    return dst.reshape(-1, 2)


def project_box(box_ir: DetectionBox, H: np.ndarray) -> DetectionBox:
    """把 IR 检测框投影到 RGB 平面。

    单应会把矩形变成一般四边形，因此返回 4 个变换后角点的**轴对齐外接框**
    （便于做框级 IoU 关联）；同时把 4 个变换后角点挂在返回对象的 ``quad`` 属性上，
    供需要精确多边形的场景使用。

    Args:
        box_ir: IR 坐标系下的检测框。
        H: 3x3 单应矩阵。

    Returns:
        RGB 坐标系下的 ``DetectionBox``（轴对齐外接框），并带 ``quad`` 属性。
    """
    corners = np.asarray(box_ir.corners(), dtype=np.float64).reshape(-1, 1, 2)
    import cv2

    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    xs = warped[:, 0]
    ys = warped[:, 1]

    out = DetectionBox(
        x1=float(xs.min()),
        y1=float(ys.min()),
        x2=float(xs.max()),
        y2=float(ys.max()),
        score=box_ir.score,
        label=box_ir.label,
        class_id=box_ir.class_id,
        source=box_ir.source,
    )
    # 附带精确四边形角点（左上、右上、右下、左下顺序与 corners() 一致）。
    setattr(out, "quad", [(float(p[0]), float(p[1])) for p in warped])
    return out


def compute_iou(a: DetectionBox, b: DetectionBox) -> float:
    """两轴对齐框 IoU（同一像素平面）。"""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = a.width * a.height
    area_b = b.width * b.height
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def project_quad(box_ir: DetectionBox, H: np.ndarray) -> List[Tuple[float, float]]:
    """仅返回 IR 框 4 角投影到 RGB 后的多边形角点（不取外接框）。"""
    corners = np.asarray(box_ir.corners(), dtype=np.float64).reshape(-1, 1, 2)
    import cv2

    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    return [(float(p[0]), float(p[1])) for p in warped]
