#!/usr/bin/env python3
"""红外(IR) -> 可见光(RGB) 单应标定工具（网页端交互）。

在 RDK X5 上以系统 python3 运行（系统 python 已含 fastapi/uvicorn/cv2(GStreamer)/numpy）。

工作流：
  1) 浏览器打开 http://<板子IP>:端口/ ，并排实时预览 RGB(640x480) 与 IR(256x192)。
  2) 把标定板摆到画面左上/右上/左下/右下/中心等多个位置，每个位置点“保存帧对”。
  3) 对每个位置按固定顺序（左->右、上->下）依次点选 9 个红点/热斑：
     先点 IR 热斑，再点对应 RGB 红点，凑成一对后点“确认/下一点”。
  4) 采足 >=4~5 组、总点数 >=30 后，点“求解并保存”：
     RANSAC 求 H、输出 RMSE/离群、overlay.png，并落盘 config/homography.npy + meta.json。

IR 取流依赖 uvc_ubuntu/uvc_demo（libuvc），需要 root：脚本用 `sudo -n ./uvc_demo web`。
请确保不要同时运行 web_temp_viewer.py（会抢占 IR 设备）。

注意：采集时移除画面里的人和热源、背景冷且均匀；不要用 ArUco/色卡做对应点。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse


# --------------------------------------------------------------------------------------
# 全局配置（在 main() 中由命令行填充；FastAPI lifespan 读取）
# --------------------------------------------------------------------------------------

IR_WIDTH = 256
IR_HEIGHT = 192
IR_LEN_BYTES = IR_WIDTH * IR_HEIGHT * 2  # 16-bit/px
RGB_WIDTH = 640
RGB_HEIGHT = 480
NUM_POINTS = 9  # 每组对应点数（3x3）

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class AppConfig:
    rgb_dev: str = "/dev/video_rgb"
    rgb_w: int = RGB_WIDTH
    rgb_h: int = RGB_HEIGHT
    rgb_fps: int = 60
    ir_dir: Path = REPO_ROOT / "uvc_ubuntu"
    ir_cmd: List[str] = field(default_factory=lambda: ["./uvc_demo", "web"])
    ir_debug_json: Path = REPO_ROOT / "uvc_ubuntu" / "thermal_cam_debug.json"
    calib_dir: Path = REPO_ROOT / "calib_data"
    out_dir: Path = REPO_ROOT / "config"
    sync_ms: float = 33.0
    ransac_thresh: float = 3.0
    z0: float = 0.0
    num_points: int = NUM_POINTS
    send_fps: float = 12.0
    jpeg_quality: int = 80


CONFIG = AppConfig()


# --------------------------------------------------------------------------------------
# IR raw 温度解析（精简自 uvc_ubuntu/web_temp_viewer.py）
# --------------------------------------------------------------------------------------

def select_endian(temp_bytes: bytes) -> np.ndarray:
    """从 raw 字节里挑更像“温度场”的字节序，返回 (H,W) uint16。"""
    le = np.frombuffer(temp_bytes, dtype="<u2").reshape((IR_HEIGHT, IR_WIDTH))
    be = np.frombuffer(temp_bytes, dtype=">u2").reshape((IR_HEIGHT, IR_WIDTH))

    def score(a: np.ndarray) -> float:
        s = a.ravel()
        if s.size == 0:
            return 0.0
        p50 = float(np.percentile(s, 50))
        p90 = float(np.percentile(s, 90))
        if 26000 < p50 < 34000 and 26000 < p90 < 38000:
            return 4.0
        if 1500 < p50 < 12000 and 1500 < p90 < 20000:
            return 3.0
        if 200 < p50 < 8000 and 200 < p90 < 12000:
            return 2.0
        return 1.0

    return be if score(be) > score(le) else le


def load_ir_roll(debug_json: Path) -> Tuple[int, int]:
    """读取 thermal_cam_debug.json 的 roll_x/roll_y（与 web_temp_viewer 一致）。"""
    roll_x = int(os.environ.get("THERMAL_ROLL_X", "-92"))
    roll_y = int(os.environ.get("THERMAL_ROLL_Y", "0"))
    try:
        if debug_json.exists():
            cfg = json.loads(debug_json.read_text(encoding="utf-8"))
            align = cfg.get("align") or {}
            if "roll_x" in align:
                roll_x = int(align["roll_x"])
            if "roll_y" in align:
                roll_y = int(align["roll_y"])
    except Exception:
        pass
    return roll_x, roll_y


# --------------------------------------------------------------------------------------
# 帧缓存
# --------------------------------------------------------------------------------------

class FrameStore:
    """线程安全的“最新帧”缓存。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bgr: Optional[np.ndarray] = None
        self._extra: Optional[np.ndarray] = None  # IR temp_raw16
        self._ts: float = 0.0
        self._status: str = "starting"

    def set(self, bgr: np.ndarray, ts: float, extra: Optional[np.ndarray] = None) -> None:
        with self._lock:
            self._bgr = bgr
            self._extra = extra
            self._ts = ts
            self._status = "ok"

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def get(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, str]:
        with self._lock:
            bgr = None if self._bgr is None else self._bgr.copy()
            extra = None if self._extra is None else self._extra.copy()
            return bgr, extra, self._ts, self._status


rgb_store = FrameStore()
ir_store = FrameStore()


# --------------------------------------------------------------------------------------
# 采集线程
# --------------------------------------------------------------------------------------

def _resolve_v4l2_index(dev: str) -> int:
    """把 /dev/video_rgb 这样的符号链接解析成 V4L2 设备号（用于回退打开）。"""
    try:
        real = os.path.realpath(dev)
        name = os.path.basename(real)  # videoN
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0
    except Exception:
        return 0


class RGBCaptureThread(threading.Thread):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def _open(self) -> Optional[cv2.VideoCapture]:
        c = self.cfg
        # 1) 优先 GStreamer：MJPG -> jpegdec -> BGR
        pipeline = (
            f"v4l2src device={c.rgb_dev} io-mode=2 ! "
            f"image/jpeg,width={c.rgb_w},height={c.rgb_h},framerate={c.rgb_fps}/1 ! "
            f"jpegdec ! videoconvert ! appsink drop=1 max-buffers=1"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            print(f"[RGB] opened via GStreamer: {c.rgb_dev}", file=sys.stderr)
            return cap
        try:
            cap.release()
        except Exception:
            pass

        # 2) 回退：V4L2 + MJPG fourcc
        idx = _resolve_v4l2_index(c.rgb_dev)
        cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, c.rgb_w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c.rgb_h)
            cap.set(cv2.CAP_PROP_FPS, c.rgb_fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            print(f"[RGB] opened via V4L2 index={idx}", file=sys.stderr)
            return cap
        return None

    def run(self) -> None:
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                rgb_store.set_status("无法打开 RGB 设备，重试中…")
                time.sleep(1.0)
                continue
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        rgb_store.set_status("RGB 读帧失败，重连…")
                        break
                    if frame.shape[1] != self.cfg.rgb_w or frame.shape[0] != self.cfg.rgb_h:
                        frame = cv2.resize(frame, (self.cfg.rgb_w, self.cfg.rgb_h))
                    rgb_store.set(frame, time.monotonic())
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
            time.sleep(0.5)


class IRCaptureThread(threading.Thread):
    def __init__(self, cfg: AppConfig) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self._stop = threading.Event()
        self.roll_x, self.roll_y = load_ir_roll(cfg.ir_debug_json)
        self._proc: Optional[subprocess.Popen] = None

    def stop(self) -> None:
        self._stop.set()
        if self._proc is not None:
            try:
                self._proc.terminate()
            except Exception:
                pass

    def _start_proc(self) -> Optional[subprocess.Popen]:
        c = self.cfg
        if not (c.ir_dir / c.ir_cmd[0].lstrip("./")).exists() and c.ir_cmd[0].startswith("./"):
            # 仅提示，不强制失败（可能是绝对路径或 PATH 命令）
            pass
        if os.geteuid() == 0:
            cmd = list(c.ir_cmd)
        else:
            cmd = ["sudo", "-n"] + list(c.ir_cmd)
        try:
            return subprocess.Popen(
                cmd, cwd=str(c.ir_dir),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except Exception as e:
            ir_store.set_status(f"启动 IR 采集失败: {e}")
            return None

    def run(self) -> None:
        frame_bytes = IR_LEN_BYTES * 2  # temp + yuv
        while not self._stop.is_set():
            proc = self._start_proc()
            self._proc = proc
            if proc is None or proc.stdout is None:
                time.sleep(1.0)
                continue
            try:
                while not self._stop.is_set():
                    buf = proc.stdout.read(frame_bytes)
                    if not buf or len(buf) != frame_bytes:
                        err = b""
                        try:
                            if proc.stderr is not None:
                                err = proc.stderr.read1(4096)
                        except Exception:
                            pass
                        msg = err.decode(errors="ignore").strip()[:300] if err else "IR 流中断"
                        ir_store.set_status(f"IR 读帧失败: {msg}")
                        break

                    temp_bytes = buf[:IR_LEN_BYTES]
                    yuv_bytes = buf[IR_LEN_BYTES:]

                    temp_raw16 = select_endian(temp_bytes).copy()
                    yuyv = np.frombuffer(yuv_bytes, dtype=np.uint8).reshape((IR_HEIGHT, IR_WIDTH, 2))

                    # 与 web_temp_viewer 一致：对 yuv 与 temp 同步 roll 对齐
                    if self.roll_y:
                        yuyv = np.roll(yuyv, self.roll_y, axis=0)
                        temp_raw16 = np.roll(temp_raw16, self.roll_y, axis=0)
                    if self.roll_x:
                        yuyv = np.roll(yuyv, self.roll_x, axis=1)
                        temp_raw16 = np.roll(temp_raw16, self.roll_x, axis=1)

                    bgr = cv2.cvtColor(np.ascontiguousarray(yuyv), cv2.COLOR_YUV2BGR_YUYV)
                    ir_store.set(bgr, time.monotonic(), extra=temp_raw16)
            finally:
                try:
                    proc.terminate()
                except Exception:
                    pass
            time.sleep(0.5)


# --------------------------------------------------------------------------------------
# 标定会话状态（仅事件循环线程访问）
# --------------------------------------------------------------------------------------

@dataclass
class CalibSession:
    groups: List[dict] = field(default_factory=list)        # 已完成的组
    cur_pairs: List[dict] = field(default_factory=list)      # 当前组已确认的点对
    pending_ir: Optional[List[float]] = None
    pending_rgb: Optional[List[float]] = None
    cur_pair_file: Optional[str] = None                      # 当前组关联的帧对文件名前缀
    saved_pairs: int = 0
    last_solve: Optional[dict] = None
    message: str = ""
    # 冻结：保存帧对后定格画面，便于逐点点选（图像不再随实时流跳动）
    frozen: bool = False
    frozen_rgb_b64: Optional[str] = None
    frozen_ir_b64: Optional[str] = None
    frozen_dt_ms: Optional[float] = None

    def total_points(self) -> int:
        return sum(len(g.get("points", [])) for g in self.groups)

    def unfreeze(self) -> None:
        self.frozen = False
        self.frozen_rgb_b64 = None
        self.frozen_ir_b64 = None
        self.frozen_dt_ms = None


session = CalibSession()


def correspondences_path() -> Path:
    return CONFIG.calib_dir / "correspondences.json"


def persist_correspondences() -> None:
    CONFIG.calib_dir.mkdir(parents=True, exist_ok=True)
    obj = {
        "schema": "ir_rgb_correspondences.v1",
        "updated": datetime.now().isoformat(timespec="seconds"),
        "ir_size": [IR_WIDTH, IR_HEIGHT],
        "rgb_size": [CONFIG.rgb_w, CONFIG.rgb_h],
        "num_points_per_group": CONFIG.num_points,
        "groups": session.groups,
    }
    tmp = correspondences_path().with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, correspondences_path())


def load_correspondences() -> None:
    p = correspondences_path()
    if not p.exists():
        return
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        groups = obj.get("groups")
        if isinstance(groups, list):
            session.groups = groups
            # 推断已保存帧对数量
            session.saved_pairs = len(list(CONFIG.calib_dir.glob("pair_*_rgb.png")))
    except Exception:
        pass


# --------------------------------------------------------------------------------------
# 帧对保存
# --------------------------------------------------------------------------------------

def _next_pair_index() -> int:
    existing = sorted(CONFIG.calib_dir.glob("pair_*_rgb.png"))
    idx = 0
    for p in existing:
        try:
            n = int(p.name.split("_")[1])
            idx = max(idx, n + 1)
        except Exception:
            continue
    return idx


def save_pair_now() -> dict:
    """保存当前同步帧对到 calib_data/。返回结果 dict。"""
    rgb, _, rgb_ts, rgb_st = rgb_store.get()
    ir, ir_temp, ir_ts, ir_st = ir_store.get()
    if rgb is None or ir is None:
        return {"ok": False, "msg": "RGB 或 IR 尚无可用帧"}
    dt_ms = abs(rgb_ts - ir_ts) * 1000.0
    if dt_ms > CONFIG.sync_ms:
        return {"ok": False, "msg": f"帧不同步(Δ={dt_ms:.0f}ms>{CONFIG.sync_ms:.0f}ms)，未保存"}

    CONFIG.calib_dir.mkdir(parents=True, exist_ok=True)
    idx = _next_pair_index()
    prefix = f"pair_{idx:02d}"
    cv2.imwrite(str(CONFIG.calib_dir / f"{prefix}_rgb.png"), rgb)
    cv2.imwrite(str(CONFIG.calib_dir / f"{prefix}_ir.png"), ir)
    if ir_temp is not None:
        np.save(str(CONFIG.calib_dir / f"{prefix}_ir_temp.npy"), ir_temp)

    session.saved_pairs += 1
    session.cur_pair_file = prefix

    # 冻结画面：缓存这一对（已 roll 对齐的）图像，后续点选都基于这张静止图。
    session.frozen = True
    session.frozen_rgb_b64 = _encode_jpeg_b64(rgb)
    session.frozen_ir_b64 = _encode_jpeg_b64(ir)
    session.frozen_dt_ms = round(dt_ms, 1)

    return {
        "ok": True,
        "msg": f"已保存帧对 {prefix}（Δ={dt_ms:.0f}ms），画面已冻结，请逐点点选",
        "prefix": prefix,
    }


# --------------------------------------------------------------------------------------
# 控制消息处理
# --------------------------------------------------------------------------------------

def handle_control(data: dict) -> None:
    t = data.get("type")
    if t == "save_pair":
        res = save_pair_now()
        session.message = res["msg"]
        return

    if t == "click":
        view = data.get("view")
        try:
            x = float(data.get("x"))
            y = float(data.get("y"))
        except (TypeError, ValueError):
            return
        if view == "ir":
            x = max(0.0, min(IR_WIDTH - 1.0, x))
            y = max(0.0, min(IR_HEIGHT - 1.0, y))
            session.pending_ir = [x, y]
        elif view == "rgb":
            x = max(0.0, min(CONFIG.rgb_w - 1.0, x))
            y = max(0.0, min(CONFIG.rgb_h - 1.0, y))
            session.pending_rgb = [x, y]
        return

    if t == "confirm":
        if len(session.cur_pairs) >= CONFIG.num_points:
            session.message = f"本组已有 {CONFIG.num_points} 点，请“完成本组”"
            return
        if session.pending_ir is None or session.pending_rgb is None:
            session.message = "请先在 IR 和 RGB 上各点一个对应点，再确认"
            return
        session.cur_pairs.append({
            "idx": len(session.cur_pairs),
            "ir": session.pending_ir,
            "rgb": session.pending_rgb,
        })
        session.pending_ir = None
        session.pending_rgb = None
        session.message = f"已确认第 {len(session.cur_pairs)}/{CONFIG.num_points} 点"
        return

    if t == "undo":
        if session.pending_ir is not None or session.pending_rgb is not None:
            session.pending_ir = None
            session.pending_rgb = None
            session.message = "已清除当前未确认点"
        elif session.cur_pairs:
            session.cur_pairs.pop()
            session.message = f"已撤销，回到第 {len(session.cur_pairs) + 1} 点"
        else:
            session.message = "没有可撤销的点"
        return

    if t == "unfreeze":
        session.unfreeze()
        session.message = "已解冻，恢复实时画面"
        return

    if t == "skip":
        session.cur_pairs = []
        session.pending_ir = None
        session.pending_rgb = None
        session.cur_pair_file = None
        session.unfreeze()
        session.message = "已跳过当前组，恢复实时画面"
        return

    if t == "next_group":
        if len(session.cur_pairs) < CONFIG.num_points:
            session.message = (
                f"当前仅 {len(session.cur_pairs)}/{CONFIG.num_points} 点，"
                f"建议点满 {CONFIG.num_points} 点再完成（仍可继续）"
            )
            if len(session.cur_pairs) < 1:
                return
        group = {
            "group_id": len(session.groups),
            "pair_file": session.cur_pair_file,
            "ts": datetime.now().isoformat(timespec="seconds"),
            "points": session.cur_pairs,
        }
        session.groups.append(group)
        session.cur_pairs = []
        session.pending_ir = None
        session.pending_rgb = None
        session.cur_pair_file = None
        session.unfreeze()
        persist_correspondences()
        session.message = (
            f"已完成第 {len(session.groups)} 组；累计 {session.total_points()} 点，恢复实时画面"
        )
        return

    if t == "solve":
        res = solve_and_save()
        session.last_solve = res
        session.message = res.get("msg", "")
        return


# --------------------------------------------------------------------------------------
# 求解 + 误差报告 + 落盘
# --------------------------------------------------------------------------------------

def _gather_points() -> Tuple[np.ndarray, np.ndarray, int]:
    pts_ir: List[List[float]] = []
    pts_rgb: List[List[float]] = []
    for g in session.groups:
        for p in g.get("points", []):
            pts_ir.append([float(p["ir"][0]), float(p["ir"][1])])
            pts_rgb.append([float(p["rgb"][0]), float(p["rgb"][1])])
    return (
        np.asarray(pts_ir, dtype=np.float64),
        np.asarray(pts_rgb, dtype=np.float64),
        len(pts_ir),
    )


def _draw_overlay(H: np.ndarray, pts_ir: np.ndarray, pts_rgb: np.ndarray,
                  errors: np.ndarray, outlier_mask: np.ndarray) -> Path:
    # 选一张底图：优先最近保存的 RGB 帧对，否则用当前实时帧，再否则黑底。
    base = None
    saved = sorted(CONFIG.calib_dir.glob("pair_*_rgb.png"))
    if saved:
        base = cv2.imread(str(saved[-1]))
    if base is None:
        live, _, _, _ = rgb_store.get()
        base = live
    if base is None:
        base = np.zeros((CONFIG.rgb_h, CONFIG.rgb_w, 3), dtype=np.uint8)
    base = base.copy()

    proj = cv2.perspectiveTransform(pts_ir.reshape(-1, 1, 2), H).reshape(-1, 2)
    for i in range(len(pts_rgb)):
        rgb_pt = (int(round(pts_rgb[i, 0])), int(round(pts_rgb[i, 1])))
        pr_pt = (int(round(proj[i, 0])), int(round(proj[i, 1])))
        is_out = bool(outlier_mask[i])
        # 点选 RGB 点：绿色；投影点：红色（离群用橙色）；连线
        cv2.circle(base, rgb_pt, 5, (0, 200, 0), 2)
        cv2.circle(base, pr_pt, 4, (0, 0, 255) if not is_out else (0, 165, 255), -1)
        cv2.line(base, rgb_pt, pr_pt, (200, 200, 200), 1)
        cv2.putText(base, f"{errors[i]:.1f}", (pr_pt[0] + 4, pr_pt[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)

    CONFIG.calib_dir.mkdir(parents=True, exist_ok=True)
    out_path = CONFIG.calib_dir / "overlay.png"
    cv2.imwrite(str(out_path), base)
    return out_path


def solve_and_save() -> dict:
    pts_ir, pts_rgb, n = _gather_points()
    if n < 4:
        return {"ok": False, "msg": f"对应点不足({n})，至少需要 4 点（建议 >=30）"}

    H, mask = cv2.findHomography(
        pts_ir, pts_rgb, cv2.RANSAC, ransacReprojThreshold=CONFIG.ransac_thresh
    )
    if H is None:
        return {"ok": False, "msg": "findHomography 失败（点退化？请检查对应点）"}

    proj = cv2.perspectiveTransform(pts_ir.reshape(-1, 1, 2), H).reshape(-1, 2)
    errors = np.linalg.norm(proj - pts_rgb, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.ones(n, dtype=bool)
    # 离群：RANSAC 外点 或 误差超过阈值
    outlier_mask = (~inlier_mask) | (errors > CONFIG.ransac_thresh)
    n_inliers = int(inlier_mask.sum())
    n_outliers = int(outlier_mask.sum())

    # 仅用内点 RMSE 也报告一份，便于判断
    if n_inliers > 0:
        rmse_inliers = float(np.sqrt(np.mean(errors[inlier_mask] ** 2)))
    else:
        rmse_inliers = rmse

    overlay_path = _draw_overlay(H, pts_ir, pts_rgb, errors, outlier_mask)

    CONFIG.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(CONFIG.out_dir / "homography.npy"), H.astype(np.float64))

    meta = {
        "schema": "homography_meta.v1",
        "date": datetime.now().isoformat(timespec="seconds"),
        "groups": len(session.groups),
        "total_points": int(n),
        "Z0": CONFIG.z0,
        "rmse_px": round(rmse, 4),
        "rmse_inliers_px": round(rmse_inliers, 4),
        "inliers": n_inliers,
        "outliers": n_outliers,
        "ransac_thresh": CONFIG.ransac_thresh,
        "sync_ms": CONFIG.sync_ms,
        "ir_roll_xy": list(load_ir_roll(CONFIG.ir_debug_json)),
        "ir_size": [IR_WIDTH, IR_HEIGHT],
        "rgb_size": [CONFIG.rgb_w, CONFIG.rgb_h],
        "max_error_px": round(float(errors.max()), 4),
        "passed": bool(rmse <= 5.0 and len(session.groups) >= 4 and n >= 30),
    }
    meta_path = CONFIG.out_dir / "homography_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    per_point = [
        {"i": i, "err_px": round(float(errors[i]), 3), "outlier": bool(outlier_mask[i])}
        for i in range(n)
    ]

    msg = (
        f"求解完成：RMSE={rmse:.2f}px（内点 {rmse_inliers:.2f}px），"
        f"组数={len(session.groups)}，点数={n}，内点={n_inliers}，离群={n_outliers}。"
        f"{'达标✓' if meta['passed'] else '未达标（RMSE<=5 且 >=4组 且 >=30点）'}"
    )
    return {
        "ok": True,
        "msg": msg,
        "rmse_px": round(rmse, 3),
        "rmse_inliers_px": round(rmse_inliers, 3),
        "inliers": n_inliers,
        "outliers": n_outliers,
        "groups": len(session.groups),
        "total_points": int(n),
        "passed": meta["passed"],
        "per_point": per_point,
        "overlay": "/calib_data/overlay.png?ts=" + str(int(time.time())),
    }


# --------------------------------------------------------------------------------------
# 帧消息编码（供 WebSocket 推送）
# --------------------------------------------------------------------------------------

def _encode_jpeg_b64(bgr: np.ndarray) -> Optional[str]:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.jpeg_quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def build_frame_message() -> Optional[str]:
    _, _, _, rgb_st = rgb_store.get()
    _, _, _, ir_st = ir_store.get()

    if session.frozen and session.frozen_rgb_b64 is not None:
        # 冻结模式：复用保存帧对时定格的图像，画面静止便于逐点点选
        rgb_b64 = session.frozen_rgb_b64
        ir_b64 = session.frozen_ir_b64
        dt_ms = session.frozen_dt_ms
        sync_ok = True
    else:
        rgb, _, rgb_ts, rgb_st = rgb_store.get()
        ir, _, ir_ts, ir_st = ir_store.get()
        dt_ms = None
        sync_ok = False
        if rgb is not None and ir is not None:
            dt_ms = round(abs(rgb_ts - ir_ts) * 1000.0, 1)
            sync_ok = dt_ms <= CONFIG.sync_ms
        rgb_b64 = _encode_jpeg_b64(rgb) if rgb is not None else None
        ir_b64 = _encode_jpeg_b64(ir) if ir is not None else None

    payload: Dict = {
        "type": "frame",
        "rgb": rgb_b64,
        "ir": ir_b64,
        "rgb_status": rgb_st,
        "ir_status": ir_st,
        "frozen": session.frozen,
        "sync_ok": sync_ok,
        "sync_dt_ms": dt_ms,
        "sync_ms": CONFIG.sync_ms,
        # 会话状态
        "groups": len(session.groups),
        "total_points": session.total_points(),
        "num_points": CONFIG.num_points,
        "cur_index": len(session.cur_pairs),
        "cur_pairs": session.cur_pairs,
        "pending_ir": session.pending_ir,
        "pending_rgb": session.pending_rgb,
        "saved_pairs": session.saved_pairs,
        "cur_pair_file": session.cur_pair_file,
        "message": session.message,
        "last_solve": session.last_solve,
    }
    return json.dumps(payload)


# --------------------------------------------------------------------------------------
# FastAPI 应用
# --------------------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    CONFIG.calib_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.out_dir.mkdir(parents=True, exist_ok=True)
    load_correspondences()
    rgb_thread = RGBCaptureThread(CONFIG)
    ir_thread = IRCaptureThread(CONFIG)
    rgb_thread.start()
    ir_thread.start()
    app.state.rgb_thread = rgb_thread
    app.state.ir_thread = ir_thread
    try:
        yield
    finally:
        rgb_thread.stop()
        ir_thread.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/config")
def get_config() -> JSONResponse:
    return JSONResponse({
        "rgb_dev": CONFIG.rgb_dev,
        "rgb_size": [CONFIG.rgb_w, CONFIG.rgb_h],
        "ir_size": [IR_WIDTH, IR_HEIGHT],
        "sync_ms": CONFIG.sync_ms,
        "ransac_thresh": CONFIG.ransac_thresh,
        "z0": CONFIG.z0,
        "num_points": CONFIG.num_points,
        "calib_dir": str(CONFIG.calib_dir),
        "out_dir": str(CONFIG.out_dir),
        "ir_roll_xy": list(load_ir_roll(CONFIG.ir_debug_json)),
    })


@app.get("/health")
def health() -> JSONResponse:
    _, _, _, rgb_st = rgb_store.get()
    _, _, _, ir_st = ir_store.get()
    return JSONResponse({
        "rgb_status": rgb_st,
        "ir_status": ir_st,
        "groups": len(session.groups),
        "total_points": session.total_points(),
        "saved_pairs": session.saved_pairs,
    })


@app.get("/api/solve")
def api_solve() -> JSONResponse:
    res = solve_and_save()
    session.last_solve = res
    session.message = res.get("msg", "")
    return JSONResponse(res)


@app.get("/calib_data/{name}")
def get_calib_file(name: str) -> FileResponse:
    # 仅允许读取 calib_data 目录下的文件（防目录穿越）
    safe = Path(name).name
    fp = CONFIG.calib_dir / safe
    if not fp.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(str(fp))


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    stop = asyncio.Event()
    interval = 1.0 / max(1.0, CONFIG.send_fps)

    async def sender() -> None:
        while not stop.is_set():
            try:
                msg = await asyncio.to_thread(build_frame_message)
                if msg:
                    await ws.send_text(msg)
            except Exception:
                break
            await asyncio.sleep(interval)

    task = asyncio.create_task(sender())
    try:
        while True:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue
            handle_control(data)
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass


# --------------------------------------------------------------------------------------
# 前端页面
# --------------------------------------------------------------------------------------

HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IR-RGB 单应标定</title>
<style>
  body { font-family: system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,sans-serif; margin:0; background:#0b1220; color:#e6edf3; }
  .top { padding:10px 16px; background:#111a2e; border-bottom:1px solid rgba(255,255,255,.08); }
  .wrap { display:flex; gap:16px; padding:16px; align-items:flex-start; flex-wrap:wrap; }
  .card { background:#111a2e; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px; }
  canvas { background:#000; border-radius:8px; cursor:crosshair; display:block; }
  .lbl { opacity:.8; font-size:13px; margin-bottom:6px; }
  .mono { font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  button { background:#1f2a44; color:#e6edf3; border:1px solid rgba(255,255,255,.14); border-radius:8px; padding:8px 12px; cursor:pointer; margin:2px; }
  button:hover { background:#243150; }
  button.primary { background:#2563eb; border-color:#2563eb; }
  button.danger { background:#7f1d1d; border-color:#991b1b; }
  .ok { color:#34d399; } .bad { color:#f87171; } .warn { color:#fbbf24; }
  .big { font-size:20px; font-weight:700; }
  .panel { min-width:300px; }
  #overlayImg { max-width:640px; border-radius:8px; margin-top:8px; display:none; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#1f2a44; margin:2px; font-size:12px; }
</style>
</head>
<body>
<div class="top">
  <span class="big">IR &rarr; RGB 单应标定</span>
  <span id="syncBadge" class="pill">同步: -</span>
  <span class="pill">组数 <b id="groups">0</b></span>
  <span class="pill">总点 <b id="totalPts">0</b></span>
  <span class="pill">已存帧对 <b id="savedPairs">0</b></span>
</div>

<div class="wrap">
  <div class="card">
    <div class="lbl">RGB (640&times;480) &mdash; 点选红圆点</div>
    <canvas id="rgbCanvas" width="640" height="480" style="width:640px;height:480px;"></canvas>
    <div class="lbl mono" id="rgbInfo" style="margin-top:6px">-</div>
  </div>
  <div class="card">
    <div class="lbl">IR (256&times;192) &mdash; 点选热斑（放大显示）</div>
    <canvas id="irCanvas" width="256" height="192" style="width:512px;height:384px;"></canvas>
    <div class="lbl mono" id="irInfo" style="margin-top:6px">-</div>
  </div>

  <div class="card panel">
    <div class="lbl">操作流程</div>
    <div class="mono" style="font-size:12px;line-height:1.5;opacity:.85">
      1. 摆好标定板 &rarr; 点【保存帧对(冻结)】，画面定格便于点选<br/>
      2. 按 左&rarr;右、上&rarr;下 顺序：先点 IR 热斑，再点 RGB 红点 &rarr; 点【确认/下一点】<br/>
      3. 点满 9 点 &rarr; 点【完成本组】(自动解冻)，换下一个位置重复<br/>
      4. &ge;4~5 组、&ge;30 点后 &rarr; 点【求解并保存】<br/>
      <span style="opacity:.7">想换定格瞬间：点【重新取流(解冻)】恢复实时，再【保存帧对】。</span>
    </div>
    <hr style="border-color:rgba(255,255,255,.1)"/>
    <div class="lbl">当前组进度</div>
    <div class="big"><span id="curIdx">0</span> / <span id="numPts">9</span> 点</div>
    <div id="pendStat" class="mono" style="font-size:12px;margin:6px 0">待确认: IR - , RGB -</div>
    <div id="pairList" style="margin:6px 0"></div>
    <hr style="border-color:rgba(255,255,255,.1)"/>
    <div>
      <button class="primary" onclick="send('save_pair')">保存帧对(冻结)</button>
      <button onclick="send('unfreeze')">重新取流(解冻)</button>
    </div>
    <div>
      <button class="primary" onclick="send('confirm')">确认/下一点</button>
      <button onclick="send('undo')">撤销</button>
    </div>
    <div>
      <button onclick="send('next_group')">完成本组</button>
      <button class="danger" onclick="send('skip')">跳过本组</button>
    </div>
    <div style="margin-top:8px">
      <button class="primary" onclick="send('solve')">求解并保存 H</button>
    </div>
    <div class="lbl" style="margin-top:10px">状态</div>
    <div id="message" class="mono" style="font-size:13px;min-height:20px">-</div>
    <div id="solveBox" class="mono" style="font-size:13px;margin-top:8px"></div>
    <img id="overlayImg"/>
  </div>
</div>

<script>
const rgbCanvas = document.getElementById('rgbCanvas');
const irCanvas  = document.getElementById('irCanvas');
const rgbCtx = rgbCanvas.getContext('2d');
const irCtx  = irCanvas.getContext('2d');
let state = {};
let rgbImg = new Image(), irImg = new Image();
// 关键：在图片真正解码完成(onload)后再绘制，避免“设 src 后立刻 drawImage”导致的
// 空操作（画面黑屏）+ 标记叠加。每次绘制都先铺底(图或黑)清屏，再叠加标记。
rgbImg.onload = drawRGB;
irImg.onload  = drawIR;

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);

ws.onmessage = (ev)=>{
  let msg;
  try { msg = JSON.parse(ev.data); } catch(e){ return; }
  if (msg.type !== 'frame') return;
  state = msg;
  updateUI(msg);
  // src 变化时设置新图(onload 触发重绘)；src 未变(如冻结同一张图)或无图时直接重绘以刷新标记。
  if (msg.rgb) { if (rgbImg.src !== msg.rgb) rgbImg.src = msg.rgb; else drawRGB(); } else drawRGB();
  if (msg.ir)  { if (irImg.src  !== msg.ir)  irImg.src  = msg.ir;  else drawIR();  } else drawIR();
};

function drawRGB(){
  if (rgbImg.complete && rgbImg.naturalWidth > 0)
    rgbCtx.drawImage(rgbImg,0,0,rgbCanvas.width,rgbCanvas.height);
  else { rgbCtx.fillStyle='#000'; rgbCtx.fillRect(0,0,rgbCanvas.width,rgbCanvas.height); }
  (state.cur_pairs||[]).forEach((p,i)=> drawMark(rgbCtx, p.rgb[0], p.rgb[1], '#34d399', String(i+1)));
  if (state.pending_rgb) drawMark(rgbCtx, state.pending_rgb[0], state.pending_rgb[1], '#fbbf24', '?');
}

function drawIR(){
  if (irImg.complete && irImg.naturalWidth > 0)
    irCtx.drawImage(irImg,0,0,irCanvas.width,irCanvas.height);
  else { irCtx.fillStyle='#000'; irCtx.fillRect(0,0,irCanvas.width,irCanvas.height); }
  (state.cur_pairs||[]).forEach((p,i)=> drawMark(irCtx, p.ir[0], p.ir[1], '#22d3ee', String(i+1)));
  if (state.pending_ir) drawMark(irCtx, state.pending_ir[0], state.pending_ir[1], '#fbbf24', '?');
}

function drawMark(ctx,x,y,color,txt){
  ctx.save();
  ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.arc(x,y,4,0,Math.PI*2); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(x-7,y); ctx.lineTo(x+7,y); ctx.moveTo(x,y-7); ctx.lineTo(x,y+7); ctx.stroke();
  ctx.font='12px monospace'; ctx.fillText(txt, x+6, y-6);
  ctx.restore();
}

function canvasClick(canvas, view, evt){
  const r = canvas.getBoundingClientRect();
  const x = (evt.clientX - r.left) * canvas.width / r.width;
  const y = (evt.clientY - r.top)  * canvas.height / r.height;
  ws.send(JSON.stringify({type:'click', view, x, y}));
}
rgbCanvas.addEventListener('click', e=>canvasClick(rgbCanvas,'rgb',e));
irCanvas.addEventListener('click',  e=>canvasClick(irCanvas,'ir',e));

function send(type){ ws.send(JSON.stringify({type})); }

function updateUI(m){
  document.getElementById('groups').textContent = m.groups;
  document.getElementById('totalPts').textContent = m.total_points;
  document.getElementById('savedPairs').textContent = m.saved_pairs;
  document.getElementById('curIdx').textContent = m.cur_index;
  document.getElementById('numPts').textContent = m.num_points;
  document.getElementById('message').textContent = m.message || '-';

  const badge = document.getElementById('syncBadge');
  if (m.frozen){
    badge.textContent = `已冻结·可点选 (Δ=${m.sync_dt_ms}ms)`; badge.className='pill';
    badge.style.background = '#1d4ed8';
  } else if (m.sync_dt_ms === null || m.sync_dt_ms === undefined){
    badge.textContent = '同步: 等待相机'; badge.className='pill warn'; badge.style.background='';
  } else if (m.sync_ok){
    badge.textContent = `同步 OK (Δ=${m.sync_dt_ms}ms)`; badge.className='pill ok'; badge.style.background='';
  } else {
    badge.textContent = `不同步 Δ=${m.sync_dt_ms}ms>${m.sync_ms}ms`; badge.className='pill bad'; badge.style.background='';
  }
  document.getElementById('rgbInfo').textContent = 'RGB: ' + (m.rgb_status||'-');
  document.getElementById('irInfo').textContent  = 'IR: ' + (m.ir_status||'-');

  const pi = m.pending_ir ? `(${m.pending_ir[0].toFixed(0)},${m.pending_ir[1].toFixed(0)})` : '-';
  const pr = m.pending_rgb ? `(${m.pending_rgb[0].toFixed(0)},${m.pending_rgb[1].toFixed(0)})` : '-';
  document.getElementById('pendStat').textContent = `待确认: IR ${pi}, RGB ${pr}`;

  const pl = document.getElementById('pairList');
  pl.innerHTML = (m.cur_pairs||[]).map((p,i)=>
    `<span class="pill">#${i+1} IR(${p.ir[0].toFixed(0)},${p.ir[1].toFixed(0)})→RGB(${p.rgb[0].toFixed(0)},${p.rgb[1].toFixed(0)})</span>`
  ).join('');

  if (m.last_solve && m.last_solve.ok){
    const s = m.last_solve;
    const cls = s.passed ? 'ok' : 'warn';
    document.getElementById('solveBox').innerHTML =
      `<span class="${cls}">RMSE=${s.rmse_px}px (内点 ${s.rmse_inliers_px}px) | 组=${s.groups} 点=${s.total_points} 内点=${s.inliers} 离群=${s.outliers} | ${s.passed?'达标✓':'未达标'}</span>`;
    const img = document.getElementById('overlayImg');
    if (s.overlay){ img.src = s.overlay; img.style.display='block'; }
  }
}
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IR->RGB 单应标定（网页端）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--rgb-dev", default=CONFIG.rgb_dev)
    p.add_argument("--rgb-w", type=int, default=CONFIG.rgb_w)
    p.add_argument("--rgb-h", type=int, default=CONFIG.rgb_h)
    p.add_argument("--rgb-fps", type=int, default=CONFIG.rgb_fps)
    p.add_argument("--ir-dir", default=str(CONFIG.ir_dir))
    p.add_argument("--ir-cmd", default="./uvc_demo web",
                   help="IR 取流命令（在 --ir-dir 下执行），默认 './uvc_demo web'")
    p.add_argument("--calib-dir", default=str(CONFIG.calib_dir))
    p.add_argument("--out-dir", default=str(CONFIG.out_dir))
    p.add_argument("--sync-ms", type=float, default=CONFIG.sync_ms)
    p.add_argument("--ransac-thresh", type=float, default=CONFIG.ransac_thresh)
    p.add_argument("--z0", type=float, default=CONFIG.z0)
    p.add_argument("--points", type=int, default=CONFIG.num_points)
    p.add_argument("--send-fps", type=float, default=CONFIG.send_fps)
    return p.parse_args(argv)


def apply_args(ns: argparse.Namespace) -> None:
    CONFIG.rgb_dev = ns.rgb_dev
    CONFIG.rgb_w = ns.rgb_w
    CONFIG.rgb_h = ns.rgb_h
    CONFIG.rgb_fps = ns.rgb_fps
    CONFIG.ir_dir = Path(ns.ir_dir)
    CONFIG.ir_cmd = ns.ir_cmd.split()
    CONFIG.calib_dir = Path(ns.calib_dir)
    CONFIG.out_dir = Path(ns.out_dir)
    CONFIG.sync_ms = ns.sync_ms
    CONFIG.ransac_thresh = ns.ransac_thresh
    CONFIG.z0 = ns.z0
    CONFIG.num_points = ns.points
    CONFIG.send_fps = ns.send_fps


def main(argv: Optional[List[str]] = None) -> None:
    ns = parse_args(argv)
    apply_args(ns)
    import uvicorn

    print(f"标定服务启动：http://{ns.host}:{ns.port}/  "
          f"(RGB={CONFIG.rgb_dev}, IR={' '.join(CONFIG.ir_cmd)} @ {CONFIG.ir_dir})", file=sys.stderr)
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")


if __name__ == "__main__":
    main()
