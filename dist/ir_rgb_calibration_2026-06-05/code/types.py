"""检测框数据类型（最小占位）。

当前 IR/RGB 均未接入 YOLO 等检测器，本模块仅提供一个轻量 ``DetectionBox``，
作为 Late Fusion / 配准（``src/fusion/registration.py``）使用的统一框结构。
后续真正接入检测器时，可在保持字段名兼容的前提下扩展本类型。

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
        score: 置信度，默认 1.0（无检测器时填占位值）。
        label: 类别标签，默认空串。
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    label: str = ""

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
