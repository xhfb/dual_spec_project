"""云台跟踪：启动校准 + 主目标 PD 速度控制。"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from src.ptz.pd_control import PDConfig, PDController, load_pd_config

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "ptz.yaml"

logger = logging.getLogger(__name__)


@dataclass
class PtzConfig:
    serial_port: str = "/dev/ttyS1"
    yaw_id: int = 2
    pitch_id: int = 1
    yaw_ratio: float = 8.0
    pitch_ratio: float = 4.0
    baudrate: int = 115200
    default_speed: int = 300
    default_acceleration: int = 50

    calibrate_on_start: bool = True
    pitch_calibrate_speed: int = 30
    pitch_calibrate_direction: str = "CW"
    pitch_back_angle: float = -60.0
    yaw_calibrate_speed: int = 30
    yaw_first_direction: str = "CW"
    yaw_stall_interval: float = 1.1
    yaw_second_stall_detect_delay: float = 2.0
    yaw_second_stall_min_travel_motor_deg: float = 30.0

    yaw_limits: Optional[Tuple[float, float]] = None
    pitch_limits: Optional[Tuple[float, float]] = None

    control_hz: float = 15.0
    lost_stop_frames: int = 5
    return_home_timeout_s: float = 10.0
    return_home_speed: int = 50

    @classmethod
    def from_dict(cls, data: dict) -> "PtzConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        out = {k: v for k, v in data.items() if k in known}
        for key in ("yaw_limits", "pitch_limits"):
            val = out.get(key)
            if isinstance(val, list) and len(val) == 2:
                out[key] = (float(val[0]), float(val[1]))
        return cls(**out)


def load_ptz_config(path: Union[str, Path, None] = None) -> PtzConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return PtzConfig()
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = (yaml.safe_load(text) or {}).get("ptz", {})
    else:
        data = json.loads(text).get("ptz", {})
    return PtzConfig.from_dict(data)


def _parse_direction(name: str):
    from emm_stepper import Direction

    return Direction.CW if str(name).upper() == "CW" else Direction.CCW


class GimbalTracker:
    """封装 Gimbal 驱动与 PD 跟踪。"""

    def __init__(
        self,
        ptz_config: Optional[PtzConfig] = None,
        pd_config: Optional[PDConfig] = None,
        *,
        config_path: Union[str, Path, None] = None,
    ) -> None:
        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if ptz_config is None:
            ptz_config = load_ptz_config(path)
        if pd_config is None:
            pd_config = load_pd_config(path)

        self.ptz_cfg = ptz_config
        self.pd = PDController(pd_config)
        self._gimbal = None
        self._calibrated = False
        self._lost_frames = 0
        self._last_target_mono: Optional[float] = None
        self._homing = False
        self._home_triggered = False
        self.status: str = "idle"

    def _import_gimbal(self):
        if str(_REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT))
        from utils.Ptz_X42S import Gimbal  # noqa: WPS433

        return Gimbal

    def initialize(self) -> bool:
        """打开串口并在启动时执行双轴校准。"""
        Gimbal = self._import_gimbal()
        cfg = self.ptz_cfg
        try:
            self._gimbal = Gimbal(
                serial_port=cfg.serial_port,
                yaw_id=cfg.yaw_id,
                pitch_id=cfg.pitch_id,
                yaw_ratio=cfg.yaw_ratio,
                pitch_ratio=cfg.pitch_ratio,
                baudrate=cfg.baudrate,
                default_speed=cfg.default_speed,
                default_acceleration=cfg.default_acceleration,
                yaw_limits=cfg.yaw_limits,
                pitch_limits=cfg.pitch_limits,
            )
        except Exception as e:
            self.status = f"open failed: {e}"
            logger.error("云台打开失败: %s", e)
            return False

        if not cfg.calibrate_on_start:
            self._calibrated = True
            self.status = "ok (skip calibrate)"
            return True

        self.status = "calibrating..."
        logger.info("开始云台双轴校准")
        pitch_ok = self._gimbal.calibrate_pitch(
            calibrate_speed=cfg.pitch_calibrate_speed,
            calibrate_direction=_parse_direction(cfg.pitch_calibrate_direction),
            back_angle=cfg.pitch_back_angle,
        )
        yaw_ok = self._gimbal.calibrate_yaw(
            calibrate_speed=cfg.yaw_calibrate_speed,
            first_direction=_parse_direction(cfg.yaw_first_direction),
            stall_interval=cfg.yaw_stall_interval,
            second_stall_detect_delay=cfg.yaw_second_stall_detect_delay,
            second_stall_min_travel_motor_deg=cfg.yaw_second_stall_min_travel_motor_deg,
        )
        ok = pitch_ok and yaw_ok
        self._calibrated = ok
        self.status = "ok" if ok else "calibrate failed"
        if ok:
            self._gimbal.stop_all()
            self.pd.reset()
        return ok

    def stop(self) -> None:
        if self._gimbal is not None:
            try:
                self._gimbal.set_speed(0, 0)
                self._gimbal.stop_all()
            except Exception as e:
                logger.warning("云台停止异常: %s", e)
        self.pd.reset()
        self._lost_frames = 0
        self._homing = False
        self._home_triggered = False

    def _return_home(self) -> None:
        cfg = self.ptz_cfg
        assert self._gimbal is not None
        self._gimbal.set_speed(0, 0)
        self._gimbal.set_position(
            pitch_angle=0.0,
            yaw_angle=0.0,
            speed=cfg.return_home_speed,
        )
        self._homing = True
        logger.info("无目标超过 %.1fs，云台回中位", cfg.return_home_timeout_s)

    def update(
        self,
        target_cx: Optional[float],
        target_cy: Optional[float],
    ) -> dict:
        """根据主目标中心更新云台速度；无目标时减速停止。"""
        if self._gimbal is None:
            return {"status": self.status, "active": False}

        now = time.monotonic()
        cfg = self.ptz_cfg

        if target_cx is None or target_cy is None:
            self._lost_frames += 1
            if self._lost_frames >= cfg.lost_stop_frames:
                self._gimbal.set_speed(0, 0)
                self.pd.reset()

            lost_s = 0.0
            if self._last_target_mono is not None:
                lost_s = now - self._last_target_mono
                if (
                    not self._home_triggered
                    and lost_s >= cfg.return_home_timeout_s
                ):
                    self._home_triggered = True
                    self._return_home()

            if self._homing:
                pitch_ok, yaw_ok = self._gimbal.is_position_reached()
                if pitch_ok and yaw_ok:
                    self._homing = False
                pos = self._gimbal.get_position()
                return {
                    "status": self.status,
                    "active": False,
                    "homing": True,
                    "lost_seconds": round(lost_s, 1),
                    "lost_frames": self._lost_frames,
                    "position_pitch": round(pos[0], 2),
                    "position_yaw": round(pos[1], 2),
                    "pd": self.pd.last_state.to_dict(),
                }

            return {
                "status": self.status,
                "active": False,
                "homing": False,
                "lost_seconds": round(lost_s, 1),
                "lost_frames": self._lost_frames,
                "pd": self.pd.last_state.to_dict(),
            }

        self._lost_frames = 0
        self._last_target_mono = now
        self._homing = False
        self._home_triggered = False

        yaw_spd, pitch_spd = self.pd.compute(target_cx, target_cy)
        self._gimbal.set_speed(pitch_speed=pitch_spd, yaw_speed=yaw_spd)

        pos = self._gimbal.get_position()
        return {
            "status": self.status,
            "calibrated": self._calibrated,
            "active": self.pd.last_state.active,
            "homing": False,
            "yaw_speed": yaw_spd,
            "pitch_speed": pitch_spd,
            "position_pitch": round(pos[0], 2),
            "position_yaw": round(pos[1], 2),
            "pd": self.pd.last_state.to_dict(),
        }

    def close(self) -> None:
        self.stop()
        if self._gimbal is not None:
            try:
                del self._gimbal
            except Exception:
                pass
            self._gimbal = None
