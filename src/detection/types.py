"""检测框数据类型（最小占位）。

RGB 路由 YOLO 出框；IR 路由 ``src/detection/ir_thermal.py`` 热目标检测出框。
本模块提供统一 ``DetectionBox``，供 Late Fusion / 配准（``src/fusion/registration.py``）使用。

坐标约定：``DetectionBox`` 默认承载 IR 原分辨率（256x192，已做 roll 对齐）下的
像素坐标，采用左上-右下角点 ``(x1, y1, x2, y2)`` 表示。经
``registration.project_box`` 投影后，返回的框处于 RGB 像素平面（640x480）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DetectionBox:
    """轴对齐检测框。

    Attributes:
        x1, y1: 左上角像素坐标。
        x2, y2: 右下角像素坐标。
        score: 置信度（IR 路由温差对比度计算；RGB 路为检测器 score）。
        label: 类别标签，默认空串。
        class_id: 类别 ID（COCO 0=person），便于与 RGB YOLO 融合。
        source: 检测来源，如 ``"ir"`` / ``"rgb"`` / ``"fused"``。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = ""
    class_id: int = 0
    source: str = ""

    def __post_init__(self) -> None:
        # 归一化角点，保证 x1<=x2、y1<=y2，避免上游给出反向坐标。
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def corners(self) -> List[Tuple[float, float]]:
        """返回 4 个角点，顺序为左上、右上、右下、左下。"""
        return [
            (self.x1, self.y1),
            (self.x2, self.y1),
            (self.x2, self.y2),
            (self.x1, self.y2),
        ]

    def as_xyxy(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def as_xywh(self) -> Tuple[float, float, float, float]:
        """返回 (左上x, 左上y, 宽, 高)。"""
        return (self.x1, self.y1, self.width, self.height)

    @property
    def confidence(self) -> float:
        """与 ``score`` 同义，便于融合模块统一读取。"""
        return self.score

    def to_dict(self) -> dict:
        """序列化为 JSON 友好字典。"""
        return {
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "confidence": self.score,
            "class_id": self.class_id,
            "source": self.source,
            "label": self.label,
        }

    @classmethod
    def from_xywh(
        cls,
        x: float,
        y: float,
        w: float,
        h: float,
        score: float = 1.0,
        label: str = "",
    ) -> "DetectionBox":
        """从 (左上x, 左上y, 宽, 高) 构造。"""
        return cls(x, y, x + w, y + h, score=score, label=label)

    @classmethod
    def from_cxcywh(
        cls,
        cx: float,
        cy: float,
        w: float,
        h: float,
        score: float = 1.0,
        label: str = "",
    ) -> "DetectionBox":
        """从 (中心x, 中心y, 宽, 高) 构造。"""
        return cls(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0, score=score, label=label)
