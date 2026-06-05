"""跟踪模块：主目标选择 + 多帧 IoU 跟踪。"""

from src.tracking.selector import score_detection, select_best_index
from src.tracking.tracker import TargetTracker, TrackingConfig, load_tracking_config
from src.tracking.types import Track, TrackingResult

__all__ = [
    "Track",
    "TrackingConfig",
    "TrackingResult",
    "TargetTracker",
    "load_tracking_config",
    "score_detection",
    "select_best_index",
]
