#!/usr/bin/env python3
"""IR 热目标检测 — 网页端流水线调试预览。

展示完整中间过程：原图 → 温度伪彩 → 二值化 → 开/闭运算 → 连通域 → 最终检测。

用法：
  python3 scripts/ir_thermal_detect_web.py
  python3 scripts/ir_thermal_detect_web.py --synthetic
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
    IRThermalConfig,
    IRThermalDetector,
    align_temp_matrix,
    apply_row_crop,
    load_config,
    load_roll_from_meta,
)
from src.detection.types import DetectionBox  # noqa: E402

LEN_BYTES = IR_WIDTH * IR_HEIGHT * 2


class TempUnit:
    RAW = "raw"
    CK = "cK"
    DC = "dC"
    CC = "cC"


def _guess_unit(raw16: np.ndarray) -> str:
    r = raw16.astype(np.uint16).ravel()
    if r.size == 0:
        return TempUnit.RAW
    p50 = int(np.percentile(r, 50))
    p90 = int(np.percentile(r, 90))
    p99 = int(np.percentile(r, 99))
    if 26000 < p50 < 34000 and 26000 < p90 < 36000:
        return TempUnit.CK
    if 2500 <= p50 <= 9000 and p90 <= 12000:
        return TempUnit.CC
    if 800 <= p50 < 20000 and p90 >= 15000:
        return TempUnit.DC
    if 20000 < p99 < 120000:
        return TempUnit.CK
    return TempUnit.CC


def _select_endian(temp_bytes: bytes) -> Tuple[np.ndarray, str]:
    le = np.frombuffer(temp_bytes, dtype="<u2").reshape((IR_HEIGHT, IR_WIDTH))
    be = np.frombuffer(temp_bytes, dtype=">u2").reshape((IR_HEIGHT, IR_WIDTH))

    def score(a: np.ndarray) -> float:
        s = a.ravel()
        p50 = float(np.percentile(s, 50))
        p90 = float(np.percentile(s, 90))
        if 26000 < p50 < 34000 and 26000 < p90 < 38000:
            return 4.0
        if 1500 < p50 < 12000 and 1500 < p90 < 20000:
            return 3.0
        if 200 < p50 < 8000 and 200 < p90 < 12000:
            return 2.0
        return 1.0

    if score(be) > score(le):
        return be.copy(), "BE"
    return le.copy(), "LE"


def temp_to_colormap_bgr(temp: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(temp, [2, 98])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((temp.astype(np.float32) - lo) / (hi - lo), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


def mask_to_bgr(mask: Optional[np.ndarray]) -> np.ndarray:
    """二值 mask → 绿前景 / 黑背景。"""
    out = np.zeros((IR_HEIGHT, IR_WIDTH, 3), dtype=np.uint8)
    if mask is not None:
        out[mask > 0] = (0, 220, 80)
    return out


def draw_boxes_on(
    bgr: np.ndarray,
    boxes: List[DetectionBox],
    *,
    color: Tuple[int, int, int] = (0, 255, 80),
) -> np.ndarray:
    out = bgr.copy()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box.x1), int(box.y1), int(box.x2), int(box.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{i} {box.score:.2f}"
        cv2.putText(
            out, label, (x1, max(14, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
        )
    return out


def build_pipeline_stages(
    camera_bgr: np.ndarray,
    thermal_bgr: np.ndarray,
    detector: IRThermalDetector,
    boxes: List[DetectionBox],
    stats: dict,
    det_cfg: IRThermalConfig,
) -> List[dict]:
    """构建流水线各阶段画面与说明（BGR，由调用方编码）。"""
    thr = stats.get("threshold", 0)
    method = stats.get("threshold_method", det_cfg.threshold_method)
    white_px = int(detector.last_mask_binary.sum()) if detector.last_mask_binary is not None else 0
    open_px = int(detector.last_mask_open.sum()) if detector.last_mask_open is not None else 0
    close_px = int(detector.last_mask.sum()) if detector.last_mask is not None else 0

    result_bgr = draw_boxes_on(thermal_bgr, boxes)

    stages: List[dict] = [
        {
            "id": "camera",
            "title": "① IR 相机原图",
            "desc": "YUYV 转 BGR，roll 对齐；上下各裁掉 crop 行（涂黑）去除机芯边缘噪点。",
            "params": {
                "roll_x": stats.get("roll_x"),
                "roll_y": stats.get("roll_y"),
                "crop_top": det_cfg.crop_top,
                "crop_bottom": det_cfg.crop_bottom,
                "size": f"{IR_WIDTH}×{IR_HEIGHT}",
            },
            "bgr": camera_bgr,
        },
        {
            "id": "thermal",
            "title": "② 温度矩阵伪彩",
            "desc": "分割输入：uint16 温度矩阵按 p2–p98 归一化后 INFERNO 伪彩。",
            "params": {
                "p50": stats.get("p50"),
                "p95": stats.get("p95"),
                "min": stats.get("t_min"),
                "max": stats.get("t_max"),
                "unit_guess": stats.get("unit"),
            },
            "bgr": thermal_bgr,
        },
        {
            "id": "binary",
            "title": "③ 自适应阈值二值化",
            "desc": f"mask = (temp > thr)。策略：{method}，thr 经分位 clamp。",
            "params": {
                "threshold_method": method,
                "thr": round(thr, 1),
                "thr_lo": stats.get("thr_lo"),
                "thr_hi": stats.get("thr_hi"),
                "white_pixels": white_px,
            },
            "bgr": mask_to_bgr(detector.last_mask_binary),
        },
        {
            "id": "morph_open",
            "title": "④ 开运算（去噪）",
            "desc": "椭圆结构元开运算，去除孤立小噪点。",
            "params": {
                "kernel": f"ellipse {det_cfg.morph_open}×{det_cfg.morph_open}",
                "white_pixels": open_px,
            },
            "bgr": mask_to_bgr(detector.last_mask_open),
        },
        {
            "id": "morph_close",
            "title": "⑤ 闭运算（填洞）",
            "desc": "椭圆结构元闭运算，连接同一目标内断裂区域。",
            "params": {
                "kernel": f"ellipse {det_cfg.morph_close}×{det_cfg.morph_close}",
                "white_pixels": close_px,
            },
            "bgr": mask_to_bgr(detector.last_mask),
        },
        {
            "id": "components",
            "title": "⑥ 连通域分析",
            "desc": "8-连通域标注；各色块为独立连通域，黑色为背景。",
            "params": {
                "num_components": stats.get("num_components", 0),
                "filter_rejected": stats.get("filter_rejected", 0),
                "min_area": det_cfg.min_area,
                "max_area": det_cfg.max_area,
            },
            "bgr": detector.last_labels_vis
            if detector.last_labels_vis is not None
            else np.zeros((IR_HEIGHT, IR_WIDTH, 3), dtype=np.uint8),
        },
        {
            "id": "result",
            "title": "⑦ 滤波 + 检测框",
            "desc": "面积/宽高比/温差/贴边滤波 → NMS → Top-K 输出 IR 框。",
            "params": {
                "detections": len(boxes),
                "min_aspect": det_cfg.min_aspect,
                "max_aspect": det_cfg.max_aspect,
                "min_temp_delta_ratio": det_cfg.min_temp_delta_ratio,
                "nms_iou": det_cfg.nms_iou,
                "max_detections": det_cfg.max_detections,
            },
            "bgr": result_bgr,
        },
    ]
    return stages


@dataclass
class AppConfig:
    host: str = "0.0.0.0"
    port: int = 8003
    config_path: Path = REPO_ROOT / "config" / "ir_thermal.yaml"
    uvc_dir: Path = UVC_DIR
    send_fps: float = 8.0
    stage_jpeg_quality: int = 68
    synthetic: bool = False


CONFIG = AppConfig()


class FrameStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._status: str = "starting"
        self._infer_fps: float = 0.0
        self._det_count: int = 0
        self._detections: List[dict] = []
        self._stats: dict = {}
        self._stages_encoded: List[dict] = []

    def set(
        self,
        *,
        infer_fps: float,
        det_count: int,
        detections: List[dict],
        stats: dict,
        stages_encoded: List[dict],
    ) -> None:
        with self._lock:
            self._status = "ok"
            self._infer_fps = infer_fps
            self._det_count = det_count
            self._detections = detections
            self._stats = stats
            self._stages_encoded = stages_encoded

    def set_status(self, status: str) -> None:
        with self._lock:
            self._status = status

    def get(self) -> Tuple[str, float, int, List[dict], dict, List[dict]]:
        with self._lock:
            return (
                self._status,
                self._infer_fps,
                self._det_count,
                list(self._detections),
                dict(self._stats),
                list(self._stages_encoded),
            )


frame_store = FrameStore()


def _make_synthetic_frame(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = np.full((IR_HEIGHT, IR_WIDTH), 3200, dtype=np.uint16)
    base += rng.integers(0, 60, size=base.shape, dtype=np.uint16)

    def blob(cx: int, cy: int, bw: int, bh: int, peak: int) -> None:
        y0, y1 = max(0, cy - bh // 2), min(IR_HEIGHT, cy + bh // 2)
        x0, x1 = max(0, cx - bw // 2), min(IR_WIDTH, cx + bw // 2)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dy = (yy - cy) / max(bh / 2, 1)
        dx = (xx - cx) / max(bw / 2, 1)
        dist = np.sqrt(dx * dx + dy * dy)
        blob_val = (peak * np.clip(1.0 - dist, 0, 1)).astype(np.uint16)
        base[y0:y1, x0:x1] = np.maximum(base[y0:y1, x0:x1], blob_val)

    blob(80, 60, 36, 72, 5000 + seed % 200)
    blob(180, 130, 28, 56, 4600 + seed % 150)
    bgr = temp_to_colormap_bgr(base)
    return base, bgr


def _start_uvc_subprocess(uvc_dir: Path) -> subprocess.Popen:
    demo = uvc_dir / "uvc_demo"
    if not demo.exists():
        raise FileNotFoundError(f"找不到 {demo}，请先在 uvc_ubuntu 目录执行 make")
    if os.geteuid() == 0:
        cmd = [str(demo), "web"]
    else:
        cmd = ["sudo", "-n", str(demo), "web"]
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(uvc_dir),
    )


def _encode_jpeg_b64(bgr: np.ndarray, quality: int) -> Optional[str]:
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def _encode_stages(stages: List[dict]) -> List[dict]:
    out = []
    q = CONFIG.stage_jpeg_quality
    for st in stages:
        img = _encode_jpeg_b64(st["bgr"], q)
        out.append({
            "id": st["id"],
            "title": st["title"],
            "desc": st["desc"],
            "params": st["params"],
            "image": img,
        })
    return out


class IRCaptureDetectThread(threading.Thread):
    def __init__(self, cfg: AppConfig, detector: IRThermalDetector) -> None:
        super().__init__(daemon=True)
        self.cfg = cfg
        self.detector = detector
        self._stop = threading.Event()
        self._fps_hist: List[float] = []
        self._roll_x, self._roll_y = load_roll_from_meta()
        detector.config.roll_x = self._roll_x
        detector.config.roll_y = self._roll_y

    def stop(self) -> None:
        self._stop.set()

    def _process_frame(self, temp_raw16: np.ndarray, yuyv: np.ndarray) -> None:
        temp_aligned = align_temp_matrix(temp_raw16, self._roll_x, self._roll_y)
        yuyv_a = yuyv.copy()
        if self._roll_y:
            yuyv_a = np.roll(yuyv_a, self._roll_y, axis=0)
        if self._roll_x:
            yuyv_a = np.roll(yuyv_a, self._roll_x, axis=1)

        t0 = time.monotonic()
        boxes = self.detector.detect(temp_aligned)
        elapsed = max(time.monotonic() - t0, 1e-6)
        infer_fps = 1.0 / elapsed
        self._fps_hist.append(infer_fps)
        if len(self._fps_hist) > 30:
            self._fps_hist.pop(0)
        avg_fps = float(np.mean(self._fps_hist))

        cfg = self.detector.config
        temp_vis = temp_aligned
        if cfg.crop_top > 0 or cfg.crop_bottom > 0:
            bg_val = int(np.percentile(temp_aligned, 10))
            temp_vis = apply_row_crop(temp_aligned, cfg.crop_top, cfg.crop_bottom, bg_val)

        camera_bgr = cv2.cvtColor(yuyv_a, cv2.COLOR_YUV2BGR_YUYV)
        thermal_bgr = temp_to_colormap_bgr(temp_vis)
        if cfg.crop_top > 0 or cfg.crop_bottom > 0:
            camera_bgr = apply_row_crop(camera_bgr, cfg.crop_top, cfg.crop_bottom, 0)
            thermal_bgr = apply_row_crop(thermal_bgr, cfg.crop_top, cfg.crop_bottom, 0)

        stats = dict(self.detector.last_stats)
        stats["infer_fps"] = avg_fps
        stats["roll_x"] = self._roll_x
        stats["roll_y"] = self._roll_y
        stats["unit"] = _guess_unit(temp_aligned)

        stages = build_pipeline_stages(
            camera_bgr, thermal_bgr, self.detector, boxes, stats, self.detector.config,
        )
        stages_encoded = _encode_stages(stages)

        frame_store.set(
            infer_fps=avg_fps,
            det_count=len(boxes),
            detections=[b.to_dict() for b in boxes],
            stats=stats,
            stages_encoded=stages_encoded,
        )

    def _run_synthetic(self) -> None:
        seed = 0
        while not self._stop.is_set():
            temp, bgr_hint = _make_synthetic_frame(seed)
            seed += 1
            yuyv = cv2.cvtColor(bgr_hint, cv2.COLOR_BGR2YUV_YUYV)
            self._process_frame(temp, yuyv)
            time.sleep(1.0 / 8.0)

    def _run_live(self) -> None:
        while not self._stop.is_set():
            try:
                proc = _start_uvc_subprocess(self.cfg.uvc_dir)
            except Exception as e:
                frame_store.set_status(f"启动 uvc_demo 失败: {e}")
                time.sleep(2.0)
                continue

            assert proc.stdout is not None
            frame_bytes = LEN_BYTES * 2
            try:
                while not self._stop.is_set():
                    buf = proc.stdout.read(frame_bytes)
                    if not buf or len(buf) != frame_bytes:
                        frame_store.set_status("IR 读帧失败，重连…")
                        break
                    temp_raw16, _ = _select_endian(buf[:LEN_BYTES])
                    yuyv = np.frombuffer(buf[LEN_BYTES:], dtype=np.uint8).reshape(
                        (IR_HEIGHT, IR_WIDTH, 2)
                    )
                    self._process_frame(temp_raw16, yuyv)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()
            time.sleep(0.5)

    def run(self) -> None:
        if self.cfg.synthetic:
            self._run_synthetic()
        else:
            self._run_live()


def build_frame_message() -> Optional[str]:
    status, infer_fps, det_count, detections, stats, stages = frame_store.get()
    return json.dumps({
        "type": "frame",
        "status": status,
        "infer_fps": round(infer_fps, 2),
        "det_count": det_count,
        "detections": detections,
        "stats": stats,
        "stages": stages,
        "ir_size": [IR_WIDTH, IR_HEIGHT],
        "synthetic": CONFIG.synthetic,
        "config": str(CONFIG.config_path.name),
    })


@asynccontextmanager
async def lifespan(app: FastAPI):
    det_cfg = load_config(CONFIG.config_path)
    detector = IRThermalDetector(det_cfg)
    thread = IRCaptureDetectThread(CONFIG, detector)
    thread.start()
    app.state.detect_thread = thread
    try:
        yield
    finally:
        thread.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/health")
def health() -> JSONResponse:
    status, infer_fps, det_count, detections, stats, stages = frame_store.get()
    return JSONResponse({
        "status": status,
        "infer_fps": infer_fps,
        "det_count": det_count,
        "detections": detections,
        "stats": stats,
        "stage_ids": [s["id"] for s in stages],
        "synthetic": CONFIG.synthetic,
    })


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


HTML_PAGE = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>IR 热目标检测 · 流水线调试</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; font-family: system-ui, sans-serif;
    background: #0b1220; color: #e5e7eb;
  }}
  header {{
    padding: 10px 16px; border-bottom: 1px solid rgba(255,255,255,.08);
    display: flex; gap: 10px; align-items: center; flex-wrap: wrap;
  }}
  h1 {{ margin: 0; font-size: 17px; font-weight: 600; }}
  .pill {{
    font: 11px/1.5 monospace; padding: 2px 8px; border-radius: 999px;
    background: #1f2937; color: #cbd5e1;
  }}
  .pill.ok {{ background: #14532d; color: #bbf7d0; }}
  .pill.warn {{ background: #713f12; color: #fde68a; }}
  .layout {{
    display: grid; grid-template-columns: 1fr 280px; gap: 12px;
    padding: 12px 16px 16px; align-items: start;
  }}
  @media (max-width: 1100px) {{ .layout {{ grid-template-columns: 1fr; }} }}
  .pipeline {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
  }}
  @media (max-width: 900px) {{ .pipeline {{ grid-template-columns: repeat(2, 1fr); }} }}
  @media (max-width: 560px) {{ .pipeline {{ grid-template-columns: 1fr; }} }}
  .stage {{
    background: #111827; border: 1px solid rgba(255,255,255,.08);
    border-radius: 8px; overflow: hidden;
  }}
  .stage-head {{
    padding: 8px 10px; border-bottom: 1px solid rgba(255,255,255,.06);
  }}
  .stage-head h3 {{ margin: 0 0 4px; font-size: 12px; color: #93c5fd; }}
  .stage-head p {{ margin: 0; font-size: 11px; color: #9ca3af; line-height: 1.35; }}
  .stage-img {{
    background: #000; display: block; width: 100%;
    aspect-ratio: {IR_WIDTH}/{IR_HEIGHT};
    image-rendering: pixelated; object-fit: contain;
  }}
  .stage-params {{
    padding: 6px 10px 8px; font: 10px/1.45 ui-monospace, monospace;
    color: #94a3b8; border-top: 1px solid rgba(255,255,255,.05);
    white-space: pre-wrap; word-break: break-all;
  }}
  .sidebar {{
    background: #111827; border: 1px solid rgba(255,255,255,.08);
    border-radius: 8px; padding: 12px; font: 12px/1.5 monospace;
    position: sticky; top: 12px;
  }}
  .sidebar h2 {{ margin: 0 0 10px; font-size: 13px; color: #9ca3af; }}
  .stat {{ margin-bottom: 8px; }}
  .stat b {{ color: #93c5fd; }}
  #detList {{ margin-top: 8px; max-height: 30vh; overflow: auto; }}
  .det-item {{
    padding: 5px 7px; margin-bottom: 5px; border-radius: 5px;
    background: rgba(255,255,255,.04); font-size: 11px;
  }}
</style>
</head>
<body>
<header>
  <h1>IR 热目标检测 · 流水线调试</h1>
  <span id="statusBadge" class="pill warn">连接中…</span>
  <span id="modeInfo" class="pill">{IR_WIDTH}×{IR_HEIGHT}</span>
  <span id="fpsBadge" class="pill">FPS -</span>
</header>
<div class="layout">
  <div class="pipeline" id="pipeline">
    <!-- 7 个阶段占位，由 JS 填充 -->
  </div>
  <aside class="sidebar">
    <h2>检测信息</h2>
    <div class="stat"><b>检测 FPS</b><br/><span id="inferFps">-</span></div>
    <div class="stat"><b>热目标数</b><br/><span id="detCount">-</span></div>
    <div class="stat"><b>阈值 thr</b><br/><span id="thrVal">-</span></div>
    <div class="stat"><b>roll_x, roll_y</b><br/><span id="rollVal">-</span></div>
    <div class="stat"><b>帧统计</b><br/><span id="tempStats">-</span></div>
    <div class="stat"><b>状态</b><br/><span id="camStatus">-</span></div>
    <div id="detList"></div>
  </aside>
</div>
<script>
const STAGE_IDS = [
  'camera','thermal','binary','morph_open','morph_close','components','result'
];

function ensureStageCards() {{
  const root = document.getElementById('pipeline');
  if (root.children.length === STAGE_IDS.length) return;
  root.innerHTML = STAGE_IDS.map(id => `
    <div class="stage" id="stage-${{id}}">
      <div class="stage-head">
        <h3 id="title-${{id}}">-</h3>
        <p id="desc-${{id}}">-</p>
      </div>
      <img class="stage-img" id="img-${{id}}" alt="${{id}}"/>
      <div class="stage-params" id="params-${{id}}">-</div>
    </div>
  `).join('');
}}
ensureStageCards();

function fmtParams(obj) {{
  if (!obj) return '-';
  return Object.entries(obj).map(([k,v]) => `${{k}}: ${{v}}`).join('\\n');
}}

const proto = location.protocol === 'https:' ? 'wss' : 'ws';
const ws = new WebSocket(`${{proto}}://${{location.host}}/ws`);

ws.onopen = () => {{
  document.getElementById('statusBadge').textContent = '已连接';
  document.getElementById('statusBadge').className = 'pill ok';
}};
ws.onclose = () => {{
  document.getElementById('statusBadge').textContent = '已断开';
  document.getElementById('statusBadge').className = 'pill warn';
}};

ws.onmessage = (ev) => {{
  let msg;
  try {{ msg = JSON.parse(ev.data); }} catch (e) {{ return; }}
  if (msg.type !== 'frame') return;

  document.getElementById('inferFps').textContent = msg.infer_fps ?? '-';
  document.getElementById('detCount').textContent = msg.det_count ?? 0;
  document.getElementById('camStatus').textContent = msg.status || '-';
  document.getElementById('fpsBadge').textContent = `FPS ${{msg.infer_fps ?? '-'}}`;
  document.getElementById('modeInfo').textContent =
    `${{(msg.ir_size||[]).join('x')}}` + (msg.synthetic ? ' · synthetic' : '');

  const st = msg.stats || {{}};
  document.getElementById('thrVal').textContent =
    st.threshold !== undefined ? st.threshold.toFixed(0) : '-';
  document.getElementById('rollVal').textContent =
    `${{st.roll_x ?? '-'}}, ${{st.roll_y ?? '-'}}`;
  document.getElementById('tempStats').textContent =
    `p50=${{st.p50 ?? '-'}} p95=${{st.p95 ?? '-'}} range=${{st.temp_range ?? '-'}}`;

  (msg.stages || []).forEach(stage => {{
    const id = stage.id;
    const title = document.getElementById('title-' + id);
    const desc = document.getElementById('desc-' + id);
    const img = document.getElementById('img-' + id);
    const params = document.getElementById('params-' + id);
    if (!img) return;
    if (title) title.textContent = stage.title || id;
    if (desc) desc.textContent = stage.desc || '';
    if (params) params.textContent = fmtParams(stage.params);
    if (stage.image) img.src = stage.image;
  }});

  const list = document.getElementById('detList');
  const dets = msg.detections || [];
  if (!dets.length) {{
    list.innerHTML = '<div class="det-item">未检测到热目标</div>';
    return;
  }}
  list.innerHTML = dets.map((d, i) =>
    `<div class="det-item">#${{i+1}} person ${{d.confidence.toFixed(2)}}`
    + ` · [${{[d.x1,d.y1,d.x2,d.y2].map(v=>v.toFixed(0)).join(', ')}}]</div>`
  ).join('');
}};
</script>
</body>
</html>
"""


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IR 热目标检测流水线 Web 调试")
    p.add_argument("--host", default=CONFIG.host)
    p.add_argument("--port", type=int, default=CONFIG.port)
    p.add_argument("--config", default=str(CONFIG.config_path))
    p.add_argument("--uvc-dir", default=str(CONFIG.uvc_dir))
    p.add_argument("--send-fps", type=float, default=CONFIG.send_fps)
    p.add_argument("--synthetic", action="store_true")
    return p.parse_args(argv)


def apply_args(ns: argparse.Namespace) -> None:
    CONFIG.host = ns.host
    CONFIG.port = ns.port
    CONFIG.config_path = Path(ns.config)
    CONFIG.uvc_dir = Path(ns.uvc_dir)
    CONFIG.send_fps = ns.send_fps
    CONFIG.synthetic = ns.synthetic


def main(argv: Optional[List[str]] = None) -> None:
    ns = parse_args(argv)
    apply_args(ns)
    import uvicorn

    mode = "synthetic" if CONFIG.synthetic else f"live ({CONFIG.uvc_dir}/uvc_demo)"
    print(f"IR 流水线调试：http://{ns.host}:{ns.port}/  mode={mode}", file=sys.stderr)
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")


if __name__ == "__main__":
    main()
