#!/usr/bin/env python3
"""RGB 相机 YOLOv11 人体检测 — 网页端实时预览。

在 RDK X5 上以系统 python3 运行（需 hobot_dnn、带 GStreamer 的 cv2、fastapi、uvicorn）。

- 模型：model/yolov11_80cls.bin（COCO 80 类，class 0 = person）
- 取流：GStreamer v4l2src /dev/video_rgb，MJPG 640×480@60fps
- 预览：浏览器打开 http://<板子IP>:端口/

用法：
  python3 scripts/person_detect_web.py
  python3 scripts/person_detect_web.py --host 0.0.0.0 --port 8002 --conf 0.35
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
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
sys.path.insert(0, str(REPO_ROOT))

from utils.Yolov11_infer import YOLOv11Detector  # noqa: E402

PERSON_CLASS_ID = 0
RGB_WIDTH = 640
RGB_HEIGHT = 480


@dataclass
class AppConfig:
    rgb_dev: str = "/dev/video_rgb"
    rgb_w: int = RGB_WIDTH
    rgb_h: int = RGB_HEIGHT
    rgb_fps: int = 60
    model_path: Path = REPO_ROOT / "model" / "yolov11_80cls.bin"
    conf_thresh: float = 0.35
    nms_thresh: float = 0.5
    send_fps: float = 15.0
    jpeg_quality: int = 80


CONFIG = AppConfig()


class FrameStore:
    """线程安全的最新检测帧与统计缓存。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bgr: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._status: str = "starting"
        self._infer_fps: float = 0.0
        self._person_count: int = 0
        self._detections: List[dict] = []

    def set(
        self,
        bgr: np.ndarray,
        ts: float,
        *,
        infer_fps: float,
        person_count: int,
        detections: List[dict],
    ) -> None:
        with self._lock:
            self._bgr = bgr
            self._ts = ts
            self._status = "ok"
            self._infer_fps = infer_fps
            self._person_count = person_count
            self._detections = detections

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def get(self) -> Tuple[Optional[np.ndarray], float, str, float, int, List[dict]]:
        with self._lock:
            bgr = None if self._bgr is None else self._bgr.copy()
            return (
                bgr,
                self._ts,
                self._status,
                self._infer_fps,
                self._person_count,
                list(self._detections),
            )


frame_store = FrameStore()


def _resolve_v4l2_index(dev: str) -> int:
    try:
        real = os.path.realpath(dev)
        name = os.path.basename(real)
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else 0
    except Exception:
        return 0


def open_rgb_capture(cfg: AppConfig) -> Optional[cv2.VideoCapture]:
    pipeline = (
        f"v4l2src device={cfg.rgb_dev} io-mode=2 ! "
        f"image/jpeg,width={cfg.rgb_w},height={cfg.rgb_h},framerate={cfg.rgb_fps}/1 ! "
        f"jpegdec ! videoconvert ! appsink drop=1 max-buffers=1"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print(f"[RGB] opened via GStreamer: {cfg.rgb_dev}", file=sys.stderr)
        return cap
    try:
        cap.release()
    except Exception:
        pass

    idx = _resolve_v4l2_index(cfg.rgb_dev)
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.rgb_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.rgb_h)
        cap.set(cv2.CAP_PROP_FPS, cfg.rgb_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print(f"[RGB] opened via V4L2 index={idx}", file=sys.stderr)
        return cap
    return None


def filter_person(
    boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if boxes.size == 0:
        return boxes, scores, classes
    mask = classes == PERSON_CLASS_ID
    return boxes[mask], scores[mask], classes[mask]


def draw_persons(
    img: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    infer_fps: float,
    person_count: int,
) -> np.ndarray:
    out = img.copy()
    color = (0, 220, 80)

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, box)
        label = f"person {score:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            2,
        )

    cv2.putText(
        out,
        f"Infer FPS: {infer_fps:.1f}",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        out,
        f"Persons: {person_count}",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )
    return out


class PersonDetectThread(threading.Thread):
    def __init__(self, cfg: AppConfig, detector: YOLOv11Detector) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.detector = detector
        self._stop = threading.Event()
        self._fps_hist: List[float] = []

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.is_set():
            cap = open_rgb_capture(self.cfg)
            if cap is None:
                frame_store.set_status("无法打开 RGB 设备，重试中…")
                time.sleep(1.0)
                continue

            try:
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        frame_store.set_status("RGB 读帧失败，重连…")
                        break

                    if frame.shape[1] != self.cfg.rgb_w or frame.shape[0] != self.cfg.rgb_h:
                        frame = cv2.resize(frame, (self.cfg.rgb_w, self.cfg.rgb_h))

                    t0 = time.monotonic()
                    boxes, scores, classes = self.detector.detect(frame)
                    boxes, scores, classes = filter_person(boxes, scores, classes)
                    elapsed = max(time.monotonic() - t0, 1e-6)
                    infer_fps = 1.0 / elapsed

                    self._fps_hist.append(infer_fps)
                    if len(self._fps_hist) > 30:
                        self._fps_hist.pop(0)
                    avg_fps = float(np.mean(self._fps_hist))

                    detections = [
                        {
                            "box": [float(x) for x in box],
                            "score": float(score),
                            "class_id": PERSON_CLASS_ID,
                            "class_name": "person",
                        }
                        for box, score in zip(boxes, scores)
                    ]

                    result = draw_persons(
                        frame,
                        boxes,
                        scores,
                        infer_fps=avg_fps,
                        person_count=len(boxes),
                    )
                    frame_store.set(
                        result,
                        time.monotonic(),
                        infer_fps=avg_fps,
                        person_count=len(boxes),
                        detections=detections,
                    )
            finally:
                try:
                    cap.release()
                except Exception:
                    pass
            time.sleep(0.5)


def _encode_jpeg_b64(bgr: np.ndarray) -> Optional[str]:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.jpeg_quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def build_frame_message() -> Optional[str]:
    bgr, _, status, infer_fps, person_count, detections = frame_store.get()
    payload: Dict = {
        "type": "frame",
        "rgb": _encode_jpeg_b64(bgr) if bgr is not None else None,
        "status": status,
        "infer_fps": round(infer_fps, 2),
        "person_count": person_count,
        "detections": detections,
        "model": str(CONFIG.model_path.name),
        "rgb_dev": CONFIG.rgb_dev,
        "rgb_size": [CONFIG.rgb_w, CONFIG.rgb_h],
    }
    return json.dumps(payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not CONFIG.model_path.exists():
        raise FileNotFoundError(f"模型不存在: {CONFIG.model_path}")

    detector = YOLOv11Detector(
        model_path=str(CONFIG.model_path),
        conf_thresh=CONFIG.conf_thresh,
        nms_thresh=CONFIG.nms_thresh,
        cls_num=80,
    )
    detect_thread = PersonDetectThread(CONFIG, detector)
    detect_thread.start()
    app.state.detect_thread = detect_thread
    app.state.detector = detector
    try:
        yield
    finally:
        detect_thread.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
def health() -> JSONResponse:
    _, _, status, infer_fps, person_count, detections = frame_store.get()
    return JSONResponse(
        {
            "status": status,
            "infer_fps": infer_fps,
            "person_count": person_count,
            "detections": detections,
            "rgb_dev": CONFIG.rgb_dev,
            "model": str(CONFIG.model_path),
        }
    )


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


HTML_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>YOLOv11 人体检测</title>
<style>
  :root { color-scheme: dark; }
  body {
    margin: 0; font-family: system-ui, sans-serif;
    background: #0b1220; color: #e5e7eb;
    display: flex; flex-direction: column; min-height: 100vh;
  }
  header {
    padding: 12px 16px; border-bottom: 1px solid rgba(255,255,255,.08);
    display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
  }
  h1 { margin: 0; font-size: 18px; font-weight: 600; }
  .pill {
    font: 12px/1.6 monospace; padding: 2px 8px; border-radius: 999px;
    background: #1f2937; color: #cbd5e1;
  }
  .pill.ok { background: #14532d; color: #bbf7d0; }
  .pill.warn { background: #713f12; color: #fde68a; }
  main {
    flex: 1; display: grid; grid-template-columns: 1fr 280px; gap: 12px;
    padding: 12px 16px 16px;
  }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .panel {
    background: #111827; border: 1px solid rgba(255,255,255,.08);
    border-radius: 10px; overflow: hidden;
  }
  .panel h2 {
    margin: 0; padding: 10px 12px; font-size: 13px; font-weight: 600;
    border-bottom: 1px solid rgba(255,255,255,.06); color: #9ca3af;
  }
  #viewWrap { position: relative; background: #000; }
  #rgbCanvas { display: block; width: 100%; height: auto; }
  .side { padding: 12px; font: 13px/1.5 monospace; }
  .stat { margin-bottom: 8px; }
  .stat b { color: #93c5fd; }
  #detList { margin-top: 8px; max-height: 50vh; overflow: auto; }
  .det-item {
    padding: 6px 8px; margin-bottom: 6px; border-radius: 6px;
    background: rgba(255,255,255,.04); font-size: 12px;
  }
</style>
</head>
<body>
<header>
  <h1>YOLOv11 人体检测 (RGB)</h1>
  <span id="statusBadge" class="pill warn">连接中…</span>
  <span id="modelInfo" class="pill">-</span>
</header>
<main>
  <div class="panel">
    <h2>实时画面 · 640×480</h2>
    <div id="viewWrap">
      <canvas id="rgbCanvas" width="640" height="480"></canvas>
    </div>
  </div>
  <div class="panel">
    <h2>检测信息</h2>
    <div class="side">
      <div class="stat"><b>推理 FPS</b><br/><span id="inferFps">-</span></div>
      <div class="stat"><b>人体数量</b><br/><span id="personCount">-</span></div>
      <div class="stat"><b>相机状态</b><br/><span id="camStatus">-</span></div>
      <div id="detList"></div>
    </div>
  </div>
</main>
<script>
const rgbCanvas = document.getElementById('rgbCanvas');
const rgbCtx = rgbCanvas.getContext('2d');
let rgbImg = new Image();
rgbImg.onload = () => {
  rgbCtx.drawImage(rgbImg, 0, 0, rgbCanvas.width, rgbCanvas.height);
};

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${proto}://${location.host}/ws`);

ws.onopen = () => {
  document.getElementById('statusBadge').textContent = '已连接';
  document.getElementById('statusBadge').className = 'pill ok';
};
ws.onclose = () => {
  document.getElementById('statusBadge').textContent = '已断开';
  document.getElementById('statusBadge').className = 'pill warn';
};

ws.onmessage = (ev) => {
  let msg;
  try { msg = JSON.parse(ev.data); } catch (e) { return; }
  if (msg.type !== 'frame') return;

  document.getElementById('inferFps').textContent = msg.infer_fps ?? '-';
  document.getElementById('personCount').textContent = msg.person_count ?? 0;
  document.getElementById('camStatus').textContent = msg.status || '-';
  document.getElementById('modelInfo').textContent =
    `${msg.rgb_dev || ''} · ${(msg.rgb_size||[]).join('x')} · ${msg.model || ''}`;

  if (msg.rgb) {
    if (rgbImg.src !== msg.rgb) rgbImg.src = msg.rgb;
    else rgbImg.onload();
  } else {
    rgbCtx.fillStyle = '#000';
    rgbCtx.fillRect(0, 0, rgbCanvas.width, rgbCanvas.height);
  }

  const list = document.getElementById('detList');
  const dets = msg.detections || [];
  if (!dets.length) {
    list.innerHTML = '<div class="det-item">未检测到人体</div>';
    return;
  }
  list.innerHTML = dets.map((d, i) => {
    const b = d.box || [];
    return `<div class="det-item">#${i+1} person ${(d.score||0).toFixed(2)}`
      + ` · [${b.map(v=>v.toFixed(0)).join(', ')}]</div>`;
  }).join('');
};
</script>
</body>
</html>
"""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLOv11 人体检测（网页端）")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--rgb-dev", default=CONFIG.rgb_dev)
    p.add_argument("--rgb-w", type=int, default=CONFIG.rgb_w)
    p.add_argument("--rgb-h", type=int, default=CONFIG.rgb_h)
    p.add_argument("--rgb-fps", type=int, default=CONFIG.rgb_fps)
    p.add_argument("--model", default=str(CONFIG.model_path))
    p.add_argument("--conf", type=float, default=CONFIG.conf_thresh, help="置信度阈值")
    p.add_argument("--nms", type=float, default=CONFIG.nms_thresh, help="NMS 阈值")
    p.add_argument("--send-fps", type=float, default=CONFIG.send_fps, help="网页推送帧率")
    return p.parse_args(argv)


def apply_args(ns: argparse.Namespace) -> None:
    CONFIG.rgb_dev = ns.rgb_dev
    CONFIG.rgb_w = ns.rgb_w
    CONFIG.rgb_h = ns.rgb_h
    CONFIG.rgb_fps = ns.rgb_fps
    CONFIG.model_path = Path(ns.model)
    CONFIG.conf_thresh = ns.conf
    CONFIG.nms_thresh = ns.nms
    CONFIG.send_fps = ns.send_fps


def main(argv: Optional[List[str]] = None) -> None:
    ns = parse_args(argv)
    apply_args(ns)
    import uvicorn

    print(
        f"人体检测服务启动：http://{ns.host}:{ns.port}/  "
        f"(RGB={CONFIG.rgb_dev}, model={CONFIG.model_path})",
        file=sys.stderr,
    )
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")


if __name__ == "__main__":
    main()
