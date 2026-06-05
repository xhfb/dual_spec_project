"""检测模块：类型定义与 IR/RGB 检测器入口。"""

from src.detection.ir_thermal import (
    IRThermalConfig,
    IRThermalDetector,
    align_temp_matrix,
    detect as detect_ir_thermal,
    load_config as load_ir_thermal_config,
    load_roll_from_meta,
)
from src.detection.types import DetectionBox

__all__ = [
    "DetectionBox",
    "IRThermalConfig",
    "IRThermalDetector",
    "align_temp_matrix",
    "detect_ir_thermal",
    "load_ir_thermal_config",
    "load_roll_from_meta",
]
