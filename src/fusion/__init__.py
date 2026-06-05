"""融合模块：配准投影与 Late Fusion。"""

from src.fusion.late_fusion import (
    FusionConfig,
    FusionMatch,
    FusionResult,
    fuse_detections,
    load_fusion_config,
)
from src.fusion.registration import (
    compute_iou,
    load_homography,
    project_box,
    project_point,
    project_points,
)

__all__ = [
    "FusionConfig",
    "FusionMatch",
    "FusionResult",
    "compute_iou",
    "fuse_detections",
    "load_fusion_config",
    "load_homography",
    "project_box",
    "project_point",
    "project_points",
]
