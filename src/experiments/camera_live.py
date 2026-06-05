"""Live 双路采集（同步版），逻辑摘自 ``scripts/dual_fusion_web.py``。"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.detection.ir_thermal import (  # noqa: E402
    IR_HEIGHT,
    IR_WIDTH,
    align_temp_matrix,
    apply_row_crop,
    load_roll_from_meta,
)
from scripts import dual_fusion_web as dfw  # noqa: E402

LEN_BYTES = IR_WIDTH * IR_HEIGHT * 2


@dataclass
class FrameBundle:
    rgb: Optional[np.ndarray] = None
    temp: Optional[np.ndarray] = None
    timestamp: float = 0.0


class DualEvalCapture:
    """评测用同步 RGB + IR 温度矩阵采集。"""

    def __init__(self) -> None:
        self._rgb_cap: Optional[cv2.VideoCapture] = None
        self._ir_proc: Optional[subprocess.Popen] = None
        self._roll_x, self._roll_y = load_roll_from_meta()
        self._ir_cfg = None

    def _ensure_rgb(self) -> None:
        if self._rgb_cap is not None and self._rgb_cap.isOpened():
            return
        if self._rgb_cap is not None:
            self._rgb_cap.release()
        self._rgb_cap = dfw._open_rgb()
        if self._rgb_cap is None:
            raise RuntimeError("RGB 相机打开失败，请检查 /dev/video_rgb 与 GStreamer")

    def _ensure_ir(self) -> None:
        if self._ir_proc is not None and self._ir_proc.poll() is None:
            return
        demo = dfw.CONFIG.uvc_dir / "uvc_demo"
        if not demo.exists():
            raise RuntimeError(f"IR uvc_demo 未编译: {demo}")
        cmd = [str(demo), "web"] if os.geteuid() == 0 else ["sudo", "-n", str(demo), "web"]
        self._ir_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(dfw.CONFIG.uvc_dir),
        )

    def read(self) -> FrameBundle:
        import time

        self._ensure_rgb()
        self._ensure_ir()
        assert self._rgb_cap is not None
        ok, rgb = self._rgb_cap.read()
        if not ok or rgb is None:
            raise RuntimeError("RGB 读帧失败")

        assert self._ir_proc is not None and self._ir_proc.stdout is not None
        fb = LEN_BYTES * 2
        buf = self._ir_proc.stdout.read(fb)
        if not buf or len(buf) != fb:
            raise RuntimeError("IR 读帧失败，检查 uvc_demo 进程")

        temp = dfw._select_endian(buf[:LEN_BYTES])
        temp_a = align_temp_matrix(temp, self._roll_x, self._roll_y)
        if self._ir_cfg is None:
            from src.detection.ir_thermal import load_config

            self._ir_cfg = load_config(dfw.CONFIG.ir_config)
        cfg = self._ir_cfg
        if cfg.crop_top or cfg.crop_bottom:
            bg = int(np.percentile(temp_a, 10))
            temp_a = apply_row_crop(temp_a, cfg.crop_top, cfg.crop_bottom, bg)

        return FrameBundle(rgb=rgb, temp=temp_a, timestamp=time.monotonic())

    def release(self) -> None:
        if self._rgb_cap is not None:
            try:
                self._rgb_cap.release()
            except Exception:
                pass
            self._rgb_cap = None
        if self._ir_proc is not None:
            try:
                self._ir_proc.terminate()
            except Exception:
                pass
            self._ir_proc = None


# 兼容旧名
LiveCameraCapture = DualEvalCapture
