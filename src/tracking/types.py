"""跟踪数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.detection.types import DetectionBox


@dataclass
class Track:
    """单条轨迹。"""

    track_id: int
    box: DetectionBox
    hits: int = 1
    age: int = 1
    time_since_update: int = 0
    primary_score: float = 0.0
    is_primary: bool = False

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "box": self.box.to_dict(),
            "hits": self.hits,
            "age": self.age,
            "time_since_update": self.time_since_update,
            "primary_score": round(self.primary_score, 4),
            "is_primary": self.is_primary,
        }


@dataclass
class TrackingResult:
    """一帧跟踪输出。"""

    tracks: List[Track] = field(default_factory=list)
    primary: Optional[Track] = None
    primary_id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "track_count": len(self.tracks),
            "primary_id": self.primary_id,
            "primary": self.primary.to_dict() if self.primary else None,
            "tracks": [t.to_dict() for t in self.tracks],
        }
