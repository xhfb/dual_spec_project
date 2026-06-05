#!/usr/bin/env python3
"""双光谱 Late Fusion — RGB YOLO + IR 热检测 融合预览。

流程：
  RGB YOLO 人体框 ──┐
                    ├─ IR 框投影到 RGB → IoU 关联 → 场景加权融合
  IR 热目标框 ──────┘

用法：
  python3 scripts/dual_fusion_web.py
  python3 scripts/dual_fusion_web.py --port 8004
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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
UVC_DIR = REPO_ROOT / "uvc_ubuntu"
sys.path.insert(0, str(REPO_ROOT))

from src.detection.ir_thermal import (  # noqa: E402
    IR_HEIGHT,
    IR_WIDTH,
    IRThermalDetector,
    align_temp_matrix,
    apply_row_crop,
    load_config as load_ir_config,
    load_roll_from_meta,
)
from src.detection.types import DetectionBox  # noqa: E402
from src.fusion.late_fusion import fuse_detections, load_fusion_config  # noqa: E402
from src.fusion.registration import load_homography  # noqa: E402
from src.tracking.tracker import TargetTracker, load_tracking_config  # noqa: E402
from utils.Yolov11_infer import YOLOv11Detector  # noqa: E402

try:
    from src.ptz.gimbal_tracker import GimbalTracker  # noqa: E402
    _PTZ_AVAILABLE = True
except Exception:
    GimbalTracker = None  # type: ignore
    _PTZ_AVAILABLE = False

PERSON_CLASS_ID = 0
RGB_W, RGB_H = 640, 480
LEN_BYTES = IR_WIDTH * IR_HEIGHT * 2

COLOR_RGB = (0, 220, 80)
COLOR_IR = (0, 180, 255)
COLOR_IR_PROJ = (0, 140, 255)
COLOR_FUSED = (0, 220, 255)
COLOR_TRACK = (0, 200, 255)
COLOR_PRIMARY = (255, 0, 255)


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8004
    rgb_dev: str = "/dev/video_rgb"
    rgb_fps: int = 60
    model_path: Path = REPO_ROOT / "model" / "yolov11_80cls.bin"
    yolo_conf: float = 0.35
    yolo_nms: float = 0.5
    ir_config: Path = REPO_ROOT / "config" / "ir_thermal.yaml"
    fusion_config: Path = REPO_ROOT / "config" / "fusion.yaml"
    tracking_config: Path = REPO_ROOT / "config" / "tracking.yaml"
    ptz_config: Path = REPO_ROOT / "config" / "ptz.yaml"
    uvc_dir: Path = UVC_DIR
    send_fps: float = 8.0
    jpeg_quality: int = 78
    enable_ptz: bool = False


CONFIG = AppConfig()


# ---------- IR 取流辅助（与 ir_thermal_detect_web 一致） ----------

def _select_endian(temp_bytes: bytes) -> np.ndarray:
    le = np.frombuffer(temp_bytes, dtype="<u2").reshape((IR_HEIGHT, IR_WIDTH))
    be = np.frombuffer(temp_bytes, dtype=">u2").reshape((IR_HEIGHT, IR_WIDTH))
    p50_le, p50_be = np.percentile(le, 50), np.percentile(be, 50)
    return (be if abs(p50_be - 4000) < abs(p50_le - 4000) else le).copy()


def _temp_colormap(temp: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(temp, [2, 98])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((temp.astype(np.float32) - lo) / (hi - lo), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)


def _yolo_to_boxes(boxes: np.ndarray, scores: np.ndarray) -> List[DetectionBox]:
    out = []
    for box, sc in zip(boxes, scores):
        x1, y1, x2, y2 = map(float, box)
        out.append(DetectionBox(
            x1, y1, x2, y2, score=float(sc),
            label="person", class_id=0, source="rgb",
        ))
    return out


def _draw_boxes(
    bgr: np.ndarray,
    boxes: List[DetectionBox],
    color: Tuple[int, int, int],
    *,
    prefix: str = "",
    dashed: bool = False,
) -> np.ndarray:
    out = bgr.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        if dashed:
            for x in range(x1, x2, 6):
                cv2.line(out, (x, y1), (min(x + 3, x2), y1), color, 1)
                cv2.line(out, (x, y2), (min(x + 3, x2), y2), color, 1)
            for y in range(y1, y2, 6):
                cv2.line(out, (x1, y), (x1, min(y + 3, y2)), color, 1)
                cv2.line(out, (x2, y), (x2, min(y + 3, y2)), color, 1)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        tag = f"{prefix}#{i} {b.score:.2f}"
        cv2.putText(out, tag, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def _draw_tracks(
    bgr: np.ndarray,
    tracks: list,
    primary_id: Optional[int],
) -> np.ndarray:
    out = bgr.copy()
    for tr in tracks:
        b = tr.box
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        is_pri = tr.track_id == primary_id
        color = COLOR_PRIMARY if is_pri else COLOR_TRACK
        thick = 3 if is_pri else 1
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        tag = f"{'★' if is_pri else ''}T{tr.track_id} {b.score:.2f}"
        cv2.putText(out, tag, (x1, max(14, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if is_pri:
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            cv2.drawMarker(out, (cx, cy), color, cv2.MARKER_CROSS, 16, 2)
    return out


def _encode_b64(bgr: Optional[np.ndarray]) -> Optional[str]:
    if bgr is None:
        return None
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.jpeg_quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


class _StreamStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bgr: Optional[np.ndarray] = None
        self._boxes: List[DetectionBox] = []
        self._fps: float = 0.0
        self._status: str = "starting"

    def set(self, bgr: np.ndarray, boxes: List[DetectionBox], fps: float) -> None:
        with self._lock:
            self._bgr = bgr
            self._boxes = boxes
            self._fps = fps
            self._status = "ok"

    def set_status(self, s: str) -> None:
        with self._lock:
            self._status = s

    def get(self) -> Tuple[Optional[np.ndarray], List[DetectionBox], float, str]:
        with self._lock:
            bgr = None if self._bgr is None else self._bgr.copy()
            return bgr, list(self._boxes), self._fps, self._status


rgb_store = _StreamStore()
ir_store = _StreamStore()
fusion_store: Dict = {"status": "starting"}
ptz_store: Dict = {"status": "disabled"}
ptz_lock = threading.Lock()


def _open_rgb() -> Optional[cv2.VideoCapture]:
    pipeline = (
        f"v4l2src device={CONFIG.rgb_dev} io-mode=2 ! "
        f"image/jpeg,width={RGB_W},height={RGB_H},framerate={CONFIG.rgb_fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=1 max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        return cap
    try:
        cap.release()
    except Exception:
        pass
    return None


class RGBThread(threading.Thread):
    def __init__(self, detector: YOLOv11Detector) -> None:
        super().__init__(daemon=True)
        self.detector = detector
        self._stop = threading.Event()
        self._hist: List[float] = []

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            cap = _open_rgb()
            if cap is None:
                rgb_store.set_status("RGB 打开失败")
                time.sleep(1)
                continue
            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    t0 = time.monotonic()
                    boxes, scores, classes = self.detector.detect(frame)
                    mask = classes == PERSON_CLASS_ID
                    boxes, scores = boxes[mask], scores[mask]
                    dt = max(time.monotonic() - t0, 1e-6)
                    self._hist.append(1.0 / dt)
                    if len(self._hist) > 20:
                        self._hist.pop(0)
                    dets = _yolo_to_boxes(boxes, scores)
                    vis = _draw_boxes(frame, dets, COLOR_RGB, prefix="rgb")
                    rgb_store.set(vis, dets, float(np.mean(self._hist)))
            finally:
                cap.release()
            time.sleep(0.5)


class IRThread(threading.Thread):
    def __init__(self, detector: IRThermalDetector) -> None:
        super().__init__(daemon=True)
        self.detector = detector
        self._stop = threading.Event()
        self._hist: List[float] = []
        self._roll_x, self._roll_y = load_roll_from_meta()
        detector.config.roll_x = self._roll_x
        detector.config.roll_y = self._roll_y

    def stop(self) -> None:
        self._stop.set()

    def _process(self, temp: np.ndarray, yuyv: np.ndarray) -> None:
        temp_a = align_temp_matrix(temp, self._roll_x, self._roll_y)
        yuyv_a = yuyv.copy()
        if self._roll_y:
            yuyv_a = np.roll(yuyv_a, self._roll_y, axis=0)
        if self._roll_x:
            yuyv_a = np.roll(yuyv_a, self._roll_x, axis=1)
        cfg = self.detector.config
        if cfg.crop_top or cfg.crop_bottom:
            bg = int(np.percentile(temp_a, 10))
            temp_a = apply_row_crop(temp_a, cfg.crop_top, cfg.crop_bottom, bg)
        t0 = time.monotonic()
        boxes = self.detector.detect(temp_a)
        dt = max(time.monotonic() - t0, 1e-6)
        self._hist.append(1.0 / dt)
        if len(self._hist) > 20:
            self._hist.pop(0)
        thermal = _temp_colormap(temp_a)
        if cfg.crop_top or cfg.crop_bottom:
            thermal = apply_row_crop(thermal, cfg.crop_top, cfg.crop_bottom, 0)
        vis = _draw_boxes(thermal, boxes, COLOR_IR, prefix="ir")
        ir_store.set(vis, boxes, float(np.mean(self._hist)))

    def run(self) -> None:
        demo = CONFIG.uvc_dir / "uvc_demo"
        while not self._stop.is_set():
            if not demo.exists():
                ir_store.set_status("uvc_demo 未编译")
                time.sleep(2)
                continue
            cmd = [str(demo), "web"] if os.geteuid() == 0 else ["sudo", "-n", str(demo), "web"]
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(CONFIG.uvc_dir),
                )
            except Exception as e:
                ir_store.set_status(f"IR 启动失败: {e}")
                time.sleep(2)
                continue
            assert proc.stdout
            fb = LEN_BYTES * 2
            try:
                while not self._stop.is_set():
                    buf = proc.stdout.read(fb)
                    if not buf or len(buf) != fb:
                        break
                    temp = _select_endian(buf[:LEN_BYTES])
                    yuyv = np.frombuffer(buf[LEN_BYTES:], dtype=np.uint8).reshape(
                        (IR_HEIGHT, IR_WIDTH, 2)
                    )
                    self._process(temp, yuyv)
            finally:
                proc.terminate()
            time.sleep(0.5)


class FusionThread(threading.Thread):
    def __init__(self, H: np.ndarray) -> None:
        super().__init__(daemon=True)
        self.H = H
        self.fusion_cfg = load_fusion_config(CONFIG.fusion_config)
        self.tracker = TargetTracker(load_tracking_config(CONFIG.tracking_config))
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            rgb_bgr, rgb_boxes, rgb_fps, rgb_st = rgb_store.get()
            ir_bgr, ir_boxes, ir_fps, ir_st = ir_store.get()
            if rgb_bgr is None and ir_bgr is None:
                fusion_store["status"] = "waiting frames"
                time.sleep(0.1)
                continue

            result = fuse_detections(rgb_boxes, ir_boxes, self.H, self.fusion_cfg)
            track_result = self.tracker.update(result.fused_boxes)

            rgb_base = rgb_bgr if rgb_bgr is not None else np.zeros((RGB_H, RGB_W, 3), np.uint8)
            panel_rgb = _draw_boxes(rgb_base, rgb_boxes, COLOR_RGB, prefix="rgb")

            panel_ir = ir_bgr if ir_bgr is not None else np.zeros((IR_HEIGHT, IR_WIDTH, 3), np.uint8)

            proj_vis = rgb_base.copy()
            proj_vis = _draw_boxes(proj_vis, rgb_boxes, COLOR_RGB, prefix="rgb")
            proj_vis = _draw_boxes(proj_vis, result.ir_projected, COLOR_IR_PROJ, prefix="ir→", dashed=True)

            fused_vis = _draw_boxes(rgb_base.copy(), result.fused_boxes, COLOR_FUSED, prefix="fuse")
            track_vis = _draw_tracks(rgb_base.copy(), track_result.tracks, track_result.primary_id)
            cv2.drawMarker(
                track_vis, (RGB_W // 2, RGB_H // 2), (160, 160, 160),
                cv2.MARKER_CROSS, 18, 1,
            )

            fusion_store.update({
                "status": "ok" if rgb_st == "ok" and ir_st == "ok" else f"rgb:{rgb_st} ir:{ir_st}",
                "rgb_fps": round(rgb_fps, 1),
                "ir_fps": round(ir_fps, 1),
                "rgb_count": len(rgb_boxes),
                "ir_count": len(ir_boxes),
                "fused_count": len(result.fused_boxes),
                "track_count": len(track_result.tracks),
                "primary_id": track_result.primary_id,
                "scene": result.scene,
                "w_rgb": result.w_rgb,
                "w_ir": result.w_ir,
                "fusion": result.to_dict(),
                "tracking": track_result.to_dict(),
                "panels": {
                    "rgb": _encode_b64(panel_rgb),
                    "ir": _encode_b64(panel_ir),
                    "projected": _encode_b64(proj_vis),
                    "fused": _encode_b64(fused_vis),
                    "track": _encode_b64(track_vis),
                },
            })
            time.sleep(1.0 / max(CONFIG.send_fps, 1))


class PTZThread(threading.Thread):
    """主目标 PD 云台跟踪线程。"""

    def __init__(self, tracker: "GimbalTracker") -> None:
        super().__init__(daemon=True)
        self.tracker = tracker
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        from src.ptz.gimbal_tracker import load_ptz_config

        hz = max(load_ptz_config(CONFIG.ptz_config).control_hz, 1.0)
        interval = 1.0 / hz
        while not self._stop.is_set():
            tracking = fusion_store.get("tracking") or {}
            primary = tracking.get("primary")
            cx = cy = None
            if primary and primary.get("box"):
                b = primary["box"]
                cx = (float(b["x1"]) + float(b["x2"])) / 2.0
                cy = (float(b["y1"]) + float(b["y2"])) / 2.0
            state = self.tracker.update(cx, cy)
            with ptz_lock:
                ptz_store.clear()
                ptz_store.update(state)
            time.sleep(interval)


def build_message() -> str:
    with ptz_lock:
        ptz = dict(ptz_store)
    return json.dumps({"type": "frame", **fusion_store, "ptz": ptz})


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CONFIG.model_path.exists():
        raise FileNotFoundError(f"模型不存在: {CONFIG.model_path}")
    H = load_homography()
    yolo = YOLOv11Detector(
        model_path=str(CONFIG.model_path),
        conf_thresh=CONFIG.yolo_conf,
        nms_thresh=CONFIG.yolo_nms,
        cls_num=80,
    )
    ir_det = IRThermalDetector(load_ir_config(CONFIG.ir_config))
    t_rgb = RGBThread(yolo)
    t_ir = IRThread(ir_det)
    t_fuse = FusionThread(H)
    t_ptz = None
    gimbal_tracker = None

    if CONFIG.enable_ptz:
        if not _PTZ_AVAILABLE or GimbalTracker is None:
            with ptz_lock:
                ptz_store.update({"status": "ptz module unavailable"})
        else:
            gimbal_tracker = GimbalTracker(config_path=CONFIG.ptz_config)
            ok = await asyncio.to_thread(gimbal_tracker.initialize)
            with ptz_lock:
                ptz_store.update({"status": gimbal_tracker.status, "calibrated": ok})
            if ok:
                t_ptz = PTZThread(gimbal_tracker)
            else:
                print(f"[PTZ] 初始化失败: {gimbal_tracker.status}", file=sys.stderr)

    threads = [t_rgb, t_ir, t_fuse]
    if t_ptz:
        threads.append(t_ptz)
    for t in threads:
        t.start()
    app.state.threads = tuple(threads)
    app.state.gimbal_tracker = gimbal_tracker
    try:
        yield
    finally:
        for t in threads:
            t.stop()
        if gimbal_tracker is not None:
            gimbal_tracker.close()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
def health() -> JSONResponse:
    return JSONResponse({k: v for k, v in fusion_store.items() if k != "panels"})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    stop = asyncio.Event()
    interval = 1.0 / max(1.0, CONFIG.send_fps)

    async def sender() -> None:
        while not stop.is_set():
            try:
                await ws.send_text(await asyncio.to_thread(build_message))
            except Exception:
                break
            await asyncio.sleep(interval)

    task = asyncio.create_task(sender())
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        stop.set()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


HTML_PAGE = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<title>双光谱 Late Fusion</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; font-family:system-ui,sans-serif; background:#0b1220; color:#e5e7eb;
    height:100vh; display:flex; flex-direction:column; overflow:hidden; }}
  header {{ flex:0 0 auto; padding:8px 14px; border-bottom:1px solid rgba(255,255,255,.08);
    display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
  h1 {{ margin:0; font-size:16px; }}
  .pill {{ font:11px/1.5 monospace; padding:2px 8px; border-radius:999px; background:#1f2937; color:#cbd5e1; }}
  .pill.ok {{ background:#14532d; color:#bbf7d0; }}
  .main {{
    flex:1; min-height:0; display:grid;
    grid-template-columns:minmax(0,1fr) minmax(0,1fr);
    gap:10px; padding:10px 12px 12px;
  }}
  @media(max-width:900px){{ .main {{ grid-template-columns:1fr; grid-template-rows:auto 1fr; }} }}
  .left-grid {{
    display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr;
    gap:8px; min-height:0;
  }}
  .right-panel {{ min-height:0; display:flex; flex-direction:column; }}
  .card {{
    background:#111827; border:1px solid rgba(255,255,255,.08);
    border-radius:8px; overflow:hidden; display:flex; flex-direction:column; min-height:0;
  }}
  .card h2 {{
    flex:0 0 auto; margin:0; padding:6px 10px; font-size:11px; color:#9ca3af;
    border-bottom:1px solid rgba(255,255,255,.06);
  }}
  .card p {{ flex:0 0 auto; margin:0; padding:0 10px 4px; font-size:10px; color:#6b7280; }}
  .thumb-wrap {{
    flex:1; min-height:0; background:#000; display:flex; align-items:center; justify-content:center;
  }}
  .thumb-wrap img {{
    display:block; max-width:100%; max-height:100%; width:auto; height:auto;
    image-rendering:pixelated;
  }}
  .track-main {{ flex:1; min-height:0; }}
  .track-view {{
    flex:1; min-height:0; background:#000; display:flex; align-items:center; justify-content:center;
    padding:4px;
  }}
  .track-view img {{
    display:block; max-width:100%; max-height:100%; width:auto; height:auto;
    image-rendering:auto;
  }}
  .track-bar {{
    flex:0 0 auto; font:11px/1.45 ui-monospace,monospace; color:#94a3b8;
    padding:6px 10px 8px; border-top:1px solid rgba(255,255,255,.06);
    display:grid; grid-template-columns:1fr 1fr; gap:4px 12px;
  }}
  .track-bar b {{ color:#93c5fd; font-weight:600; }}
  details.stats-fold {{ flex:0 0 auto; font:10px/1.4 monospace; color:#6b7280;
    padding:0 10px 8px; border-top:1px solid rgba(255,255,255,.04); }}
  details.stats-fold summary {{ cursor:pointer; color:#9ca3af; padding:4px 0; }}
  .match {{ padding:3px 6px; margin:3px 0; border-radius:4px; background:rgba(255,255,255,.04); }}
</style>
</head>
<body>
<header>
  <h1>双光谱 Late Fusion</h1>
  <span id="badge" class="pill warn">连接中</span>
  <span id="scene" class="pill">scene -</span>
  <span id="weights" class="pill">w -</span>
  <span id="primaryBadge" class="pill">主目标 -</span>
</header>
<div class="main">
  <div class="left-grid">
    <div class="card">
      <h2>RGB · YOLO</h2>
      <p>640×480 绿框</p>
      <div class="thumb-wrap"><img id="imgRgb" alt="rgb"/></div>
    </div>
    <div class="card">
      <h2>IR · 热检测</h2>
      <p>256×192 青框</p>
      <div class="thumb-wrap"><img id="imgIr" alt="ir"/></div>
    </div>
    <div class="card">
      <h2>配准叠加</h2>
      <p>绿=RGB 蓝虚线=IR投影</p>
      <div class="thumb-wrap"><img id="imgProj" alt="projected"/></div>
    </div>
    <div class="card">
      <h2>融合输出</h2>
      <p>黄框=融合候选</p>
      <div class="thumb-wrap"><img id="imgFused" alt="fused"/></div>
    </div>
  </div>
  <div class="right-panel">
    <div class="card track-main">
      <h2>主目标跟踪 · Track（录制主画面）</h2>
      <p>★洋红=主目标 + 十字中心 · 黄框=其他轨迹</p>
      <div class="track-view"><img id="imgTrack" alt="track"/></div>
      <div class="track-bar">
        <div><b>状态</b> <span id="status">-</span></div>
        <div><b>主目标</b> <span id="trackInfo">-</span></div>
        <div><b>中心</b> <span id="primaryCenter">-</span></div>
        <div><b>融合框</b> <span id="fuseCount">-</span></div>
        <div><b>RGB</b> <span id="rgbInfo">-</span></div>
        <div><b>IR</b> <span id="irInfo">-</span></div>
        <div><b>云台</b> <span id="ptzInfo">-</span></div>
        <div><b>PD 偏差</b> <span id="ptzErr">-</span></div>
      </div>
      <details class="stats-fold">
        <summary>融合关联 / 轨迹详情</summary>
        <div id="matchList"></div>
        <div id="trackList"></div>
      </details>
    </div>
  </div>
</div>
<script>
const ws = new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');
ws.onopen = () => {{ document.getElementById('badge').textContent='已连接';
  document.getElementById('badge').className='pill ok'; }};
ws.onclose = () => {{ document.getElementById('badge').textContent='已断开';
  document.getElementById('badge').className='pill warn'; }};
ws.onmessage = (ev) => {{
  let m; try{{ m=JSON.parse(ev.data); }}catch(e){{ return; }}
  if(m.type!=='frame') return;
  const p = m.panels||{{}};
  if(p.rgb) document.getElementById('imgRgb').src = p.rgb;
  if(p.ir) document.getElementById('imgIr').src = p.ir;
  if(p.projected) document.getElementById('imgProj').src = p.projected;
  if(p.fused) document.getElementById('imgFused').src = p.fused;
  if(p.track) document.getElementById('imgTrack').src = p.track;
  document.getElementById('status').textContent = m.status||'-';
  document.getElementById('rgbInfo').textContent = (m.rgb_fps||'-')+' / '+(m.rgb_count??'-');
  document.getElementById('irInfo').textContent = (m.ir_fps||'-')+' / '+(m.ir_count??'-');
  document.getElementById('fuseCount').textContent = m.fused_count??0;
  document.getElementById('trackInfo').textContent =
    (m.track_count??0)+' / '+(m.primary_id!=null?'T'+m.primary_id:'-');
  document.getElementById('primaryBadge').textContent =
    m.primary_id!=null ? '主目标 T'+m.primary_id : '主目标 -';
  const pri = m.tracking&&m.tracking.primary;
  if(pri&&pri.box) {{
    const b=pri.box; const cx=((b.x1+b.x2)/2).toFixed(0); const cy=((b.y1+b.y2)/2).toFixed(0);
    document.getElementById('primaryCenter').textContent = cx+', '+cy+' conf='+(b.confidence||0).toFixed(2);
  }} else document.getElementById('primaryCenter').textContent='-';
  const ptz = m.ptz||{{}};
  let ptzTxt = ptz.status||'-';
  if(ptz.homing) ptzTxt += ' 回中位';
  else if(ptz.active) ptzTxt += ` Y${{ptz.yaw_speed||0}} P${{ptz.pitch_speed||0}}`;
  else if(ptz.lost_seconds!==undefined) ptzTxt += ` 丢失${{ptz.lost_seconds}}s`;
  document.getElementById('ptzInfo').textContent = ptzTxt;
  const pd = ptz.pd||{{}};
  document.getElementById('ptzErr').textContent =
    pd.error_x!==undefined ? `ex=${{pd.error_x}} ey=${{pd.error_y}}` : '-';
  document.getElementById('scene').textContent = 'scene '+ (m.scene||'-');
  document.getElementById('weights').textContent =
    `w_rgb=${{(m.w_rgb||0).toFixed(2)}} w_ir=${{(m.w_ir||0).toFixed(2)}}`;
  const matches = (m.fusion&&m.fusion.matches)||[];
  const el = document.getElementById('matchList');
  if(!matches.length) {{ el.innerHTML='<div class="match">无关联</div>'; return; }}
  el.innerHTML = matches.map((x,i) => {{
    const f = x.fused||{{}};
    const src = f.source||'?';
    const ri = x.rgb_index!=null ? 'R'+x.rgb_index : '-';
    const ii = x.ir_index!=null ? 'I'+x.ir_index : '-';
    return `<div class="match">#${{i+1}} [${{src}}] rgb=${{ri}} ir=${{ii}}`
      + ` iou=${{x.iou}} conf=${{(f.confidence||0).toFixed(2)}}</div>`;
  }}).join('');
  const tracks = (m.tracking&&m.tracking.tracks)||[];
  const tl = document.getElementById('trackList');
  if(!tracks.length) {{ tl.innerHTML='<div class="match">无轨迹</div>'; return; }}
  tl.innerHTML = tracks.map(t => {{
    const b=t.box||{{}}; const star=t.is_primary?'★':'';
    return `<div class="match">${{star}}T${{t.track_id}} hits=${{t.hits}}`
      + ` lost=${{t.time_since_update}} conf=${{(b.confidence||0).toFixed(2)}}</div>`;
  }}).join('');
}};
</script>
</body>
</html>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="双光谱 Late Fusion Web")
    p.add_argument("--host", default=CONFIG.host)
    p.add_argument("--port", type=int, default=CONFIG.port)
    p.add_argument("--rgb-dev", default=CONFIG.rgb_dev)
    p.add_argument("--conf", type=float, default=CONFIG.yolo_conf)
    p.add_argument(
        "--ptz", action="store_true",
        help="启用云台 PD 跟踪（启动时执行 pitch/yaw 双轴校准）",
    )
    p.add_argument("--ptz-config", default=str(CONFIG.ptz_config))
    return p.parse_args(argv)


def main(argv=None):
    ns = parse_args(argv)
    CONFIG.host = ns.host
    CONFIG.port = ns.port
    CONFIG.rgb_dev = ns.rgb_dev
    CONFIG.yolo_conf = ns.conf
    CONFIG.enable_ptz = ns.ptz
    CONFIG.ptz_config = Path(ns.ptz_config)
    import uvicorn
    ptz_mode = "ptz ON" if CONFIG.enable_ptz else "ptz OFF"
    print(f"双光谱融合：http://{ns.host}:{ns.port}/  {ptz_mode}", file=sys.stderr)
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")


if __name__ == "__main__":
    main()
