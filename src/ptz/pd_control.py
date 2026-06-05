"""云台 PD 控制器（仅 P + D，无积分项）。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "ptz.yaml"


@dataclass
class PDConfig:
    frame_width: float = 640.0
    frame_height: float = 480.0
    center_x: float = 320.0
    center_y: float = 240.0
    deadzone_px: float = 20.0
    kp_yaw: float = 90.0
    kd_yaw: float = 35.0
    kp_pitch: float = 75.0
    kd_pitch: float = 30.0
    sign_yaw: float = -1.0
    sign_pitch: float = -1.0
    max_speed_rpm: int = 80
    min_speed_rpm: int = 4

    @classmethod
    def from_dict(cls, data: dict) -> "PDConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})


def load_pd_config(path: Union[str, Path, None] = None) -> PDConfig:
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return PDConfig()
    text = cfg_path.read_text(encoding="utf-8")
    if cfg_path.suffix.lower() in {".yaml", ".yml"}:
        import yaml

        data = (yaml.safe_load(text) or {}).get("ptz", {})
    else:
        data = json.loads(text).get("ptz", {})
    return PDConfig.from_dict(data)


@dataclass
class PDState:
    """最近一次 PD 计算状态（便于调试/显示）。"""

    error_x: float = 0.0
    error_y: float = 0.0
    deriv_x: float = 0.0
    deriv_y: float = 0.0
    yaw_speed: int = 0
    pitch_speed: int = 0
    active: bool = False

    def to_dict(self) -> dict:
        return {
            "error_x": round(self.error_x, 1),
            "error_y": round(self.error_y, 1),
            "deriv_x": round(self.deriv_x, 2),
            "deriv_y": round(self.deriv_y, 2),
            "yaw_speed": self.yaw_speed,
            "pitch_speed": self.pitch_speed,
            "active": self.active,
        }


class PDController:
    """像素偏差 → 云台速度（RPM）。"""

    def __init__(self, config: Optional[PDConfig] = None) -> None:
        self.config = config or PDConfig()
        self._prev_ex_n = 0.0
        self._prev_ey_n = 0.0
        self._prev_t: Optional[float] = None
        self.last_state = PDState()

    def reset(self) -> None:
        self._prev_ex_n = 0.0
        self._prev_ey_n = 0.0
        self._prev_t = None
        self.last_state = PDState()

    def _apply_speed(self, cmd: float) -> int:
        cfg = self.config
        if abs(cmd) < 1e-6:
            return 0
        capped = max(-cfg.max_speed_rpm, min(cfg.max_speed_rpm, cmd))
        if 0 < abs(capped) < cfg.min_speed_rpm:
            capped = cfg.min_speed_rpm if capped > 0 else -cfg.min_speed_rpm
        return int(round(capped))

    def compute(
        self,
        target_cx: float,
        target_cy: float,
        *,
        now: Optional[float] = None,
    ) -> Tuple[int, int]:
        """根据主目标中心计算 (yaw_speed, pitch_speed)。"""
        cfg = self.config
        t = time.monotonic() if now is None else now

        ex = target_cx - cfg.center_x
        ey = target_cy - cfg.center_y

        if abs(ex) < cfg.deadzone_px and abs(ey) < cfg.deadzone_px:
            self._prev_ex_n = 0.0
            self._prev_ey_n = 0.0
            self._prev_t = t
            self.last_state = PDState(error_x=ex, error_y=ey, active=False)
            return 0, 0

        half_w = max(cfg.frame_width * 0.5, 1.0)
        half_h = max(cfg.frame_height * 0.5, 1.0)
        ex_n = max(-1.0, min(1.0, ex / half_w))
        ey_n = max(-1.0, min(1.0, ey / half_h))

        if self._prev_t is None:
            dex_n = 0.0
            dey_n = 0.0
        else:
            dt = max(t - self._prev_t, 1e-3)
            dex_n = (ex_n - self._prev_ex_n) / dt
            dey_n = (ey_n - self._prev_ey_n) / dt

        yaw_cmd = cfg.sign_yaw * (cfg.kp_yaw * ex_n + cfg.kd_yaw * dex_n)
        pitch_cmd = cfg.sign_pitch * (cfg.kp_pitch * ey_n + cfg.kd_pitch * dey_n)

        yaw_spd = self._apply_speed(yaw_cmd)
        pitch_spd = self._apply_speed(pitch_cmd)

        self._prev_ex_n = ex_n
        self._prev_ey_n = ey_n
        self._prev_t = t
        self.last_state = PDState(
            error_x=ex,
            error_y=ey,
            deriv_x=dex_n,
            deriv_y=dey_n,
            yaw_speed=yaw_spd,
            pitch_speed=pitch_spd,
            active=True,
        )
        return yaw_spd, pitch_spd
