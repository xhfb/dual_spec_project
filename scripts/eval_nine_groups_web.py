#!/usr/bin/env python3
"""T9 九组对比实验 — Web 可视化版（S1/S2/S3 × RGB/IR/Fusion）。

用法（板端）::

    python3 scripts/eval_nine_groups_web.py --all --trials 10 --frames 90 --out experiments/records/
    python3 scripts/eval_nine_groups_web.py --scenario S2_low_light --mode fusion --port 8005

浏览器打开 http://<板子IP>:8005/ 查看实时预览并控制实验流程。
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import math
import sys
import threading
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.dual_fusion_web import (  # noqa: E402
    COLOR_FUSED,
    COLOR_IR,
    COLOR_IR_PROJ,
    COLOR_PRIMARY,
    COLOR_RGB,
    _draw_boxes,
    _temp_colormap,
)
from src.detection.types import DetectionBox  # noqa: E402
from src.experiments.camera_live import DualEvalCapture  # noqa: E402
from src.experiments.pipeline import (  # noqa: E402
    RGB_CENTER,
    EvalConfig,
    EvalPipeline,
    TrialResult,
    aggregate_trials,
    run_trial,
)
from src.fusion.registration import project_box  # noqa: E402

SCENARIOS = ["S1_normal", "S2_low_light", "S3_backlight"]
MODES = ["rgb", "ir", "fusion"]
SCENARIO_HINTS = {
    "S1_normal": "正常光照 >200 lux，操作员站 1.5m 画面中央",
    "S2_low_light": "低照度 <10 lux，操作员站 1.5m 画面中央",
    "S3_backlight": "强背光/逆光，RGB 局部过曝，操作员站 1.5m",
}
MODE_LABELS = {"rgb": "RGB", "ir": "IR", "fusion": "Fusion"}


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8005
    trials: int = 10
    frames: int = 90
    out_dir: Path = _REPO / "experiments" / "records"
    send_fps: float = 8.0
    jpeg_quality: int = 78
    scenarios: List[str] = field(default_factory=lambda: list(SCENARIOS))
    modes: List[str] = field(default_factory=lambda: list(MODES))


CONFIG = WebConfig()


def _json_safe(obj: Any) -> Any:
    """将 NaN/Inf 转为 null，保证 JSON / WebSocket 可序列化。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    return obj


class _LockedCapture:
    """串行化 RGB/IR 读帧，避免预览线程与 trial 线程抢流。"""

    def __init__(self, inner: DualEvalCapture, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def read(self):
        with self._lock:
            return self._inner.read()

    def release(self) -> None:
        self._inner.release()


def _trial_public(t: TrialResult) -> Dict[str, Any]:
    d = asdict(t)
    d.pop("frame_metrics", None)
    return _json_safe(d)


def _encode_b64(bgr: Optional[np.ndarray]) -> Optional[str]:
    if bgr is None:
        return None
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.jpeg_quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def _render_eval_frame(
    pipe: EvalPipeline,
    rgb: np.ndarray,
    temp: np.ndarray,
    mode: str,
    primary: Optional[DetectionBox],
    fm: Dict[str, Any],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """绘制评测预览：检测框 + 画面中心十字 + HUD。"""
    rgb_boxes = pipe.detect_rgb(rgb)
    ir_boxes = pipe.detect_ir(temp)
    mode_boxes, mean_iou = pipe.boxes_for_mode(rgb_boxes, ir_boxes)

    vis = rgb.copy()
    cv2.drawMarker(
        vis,
        (int(RGB_CENTER[0]), int(RGB_CENTER[1])),
        (160, 160, 160),
        cv2.MARKER_CROSS,
        18,
        1,
    )

    if mode == "rgb":
        vis = _draw_boxes(vis, rgb_boxes, COLOR_RGB, prefix="rgb")
    elif mode == "ir":
        projected = [project_box(b, pipe.H) for b in ir_boxes]
        vis = _draw_boxes(vis, projected, COLOR_IR, prefix="ir")
    else:
        vis = _draw_boxes(vis, rgb_boxes, COLOR_RGB, prefix="rgb")
        result = pipe.fuse(rgb_boxes, ir_boxes)
        vis = _draw_boxes(vis, result.ir_projected, COLOR_IR_PROJ, prefix="ir→", dashed=True)
        vis = _draw_boxes(vis, mode_boxes, COLOR_FUSED, prefix="fuse")

    if primary is not None:
        x1, y1, x2, y2 = map(int, (primary.x1, primary.y1, primary.x2, primary.y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PRIMARY, 3)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.drawMarker(vis, (cx, cy), COLOR_PRIMARY, cv2.MARKER_CROSS, 16, 2)
        err = fm.get("center_error_px", float("nan"))
        if not math.isnan(err):
            cv2.line(
                vis,
                (cx, cy),
                (int(RGB_CENTER[0]), int(RGB_CENTER[1])),
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )

    ir_thumb = _temp_colormap(temp)
    ir_small = cv2.resize(ir_thumb, (160, 120))
    vis[8 : 8 + 120, 8 : 8 + 160] = ir_small
    cv2.rectangle(vis, (8, 8), (168, 128), (80, 80, 80), 1)

    det = "DET" if fm.get("detected") else "LOST"
    err_px = fm.get("center_error_px", float("nan"))
    err_txt = f"{err_px:.0f}px" if not math.isnan(err_px) else "-"
    iou = fm.get("fusion_match_iou", float("nan"))
    iou_txt = f" iou={iou:.2f}" if mode == "fusion" and not math.isnan(iou) else ""
    hud = f"[{frame_idx + 1}/{total_frames}] {det} err={err_txt}{iou_txt}"
    cv2.putText(vis, hud, (8, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


def _render_eval_frame_fast(
    rgb: np.ndarray,
    temp: np.ndarray,
    primary: Optional[DetectionBox],
    fm: Dict[str, Any],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
    """trial 预览：不再重复推理，仅叠加主目标与 HUD。"""
    vis = rgb.copy()
    cv2.drawMarker(
        vis,
        (int(RGB_CENTER[0]), int(RGB_CENTER[1])),
        (160, 160, 160),
        cv2.MARKER_CROSS,
        18,
        1,
    )
    if primary is not None:
        x1, y1, x2, y2 = map(int, (primary.x1, primary.y1, primary.x2, primary.y2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PRIMARY, 3)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        cv2.drawMarker(vis, (cx, cy), COLOR_PRIMARY, cv2.MARKER_CROSS, 16, 2)
        err = fm.get("center_error_px", float("nan"))
        if not math.isnan(err):
            cv2.line(
                vis,
                (cx, cy),
                (int(RGB_CENTER[0]), int(RGB_CENTER[1])),
                (255, 180, 0),
                1,
                cv2.LINE_AA,
            )
    try:
        ir_thumb = _temp_colormap(temp)
        ir_small = cv2.resize(ir_thumb, (160, 120))
        vis[8 : 8 + 120, 8 : 8 + 160] = ir_small
        cv2.rectangle(vis, (8, 8), (168, 128), (80, 80, 80), 1)
    except Exception:
        pass
    det = "DET" if fm.get("detected") else "LOST"
    err_px = fm.get("center_error_px", float("nan"))
    err_txt = f"{err_px:.0f}px" if not math.isnan(err_px) else "-"
    hud = f"[{frame_idx + 1}/{total_frames}] {det} err={err_txt}"
    cv2.putText(vis, hud, (8, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


class ExperimentSession:
    """九组实验状态机（线程安全）。"""

    def __init__(self, cfg: WebConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.plan: List[Tuple[str, str]] = [(s, m) for s in cfg.scenarios for m in cfg.modes]
        self.group_idx = 0
        self.trial_idx = 0
        self.phase = "init"  # init | need_env | ready | running | done
        self.status_msg = "正在初始化相机…"
        self.env_records: Dict[str, Dict[str, Any]] = {}
        self.all_trials: List[TrialResult] = []
        self.live: Dict[str, Any] = {
            "preview_b64": None,
            "frame_idx": 0,
            "total_frames": cfg.frames,
            "running_det_rate": 0.0,
            "running_fps": 0.0,
        }
        self._stop_trial = threading.Event()
        self._capture: Optional[_LockedCapture] = None
        self._capture_lock = threading.Lock()
        self._preview_stop = threading.Event()
        self._preview_thread: Optional[threading.Thread] = None
        self._trial_thread: Optional[threading.Thread] = None
        self._pipe_cache: Dict[str, EvalPipeline] = {}

    def _current_group(self) -> Tuple[str, str]:
        return self.plan[self.group_idx]

    def _current_scenario(self) -> str:
        return self.plan[self.group_idx][0]

    def _current_mode(self) -> str:
        return self.plan[self.group_idx][1]

    def _get_pipe(self, mode: str) -> EvalPipeline:
        if mode not in self._pipe_cache:
            self._pipe_cache[mode] = EvalPipeline(EvalConfig(mode=mode))
        return self._pipe_cache[mode]

    def start(self) -> None:
        raw = DualEvalCapture()
        self._capture = _LockedCapture(raw, self._capture_lock)
        self._preview_stop.clear()
        self._preview_thread = threading.Thread(target=self._preview_loop, daemon=True)
        self._preview_thread.start()
        with self._lock:
            self.phase = "need_env"
            self.status_msg = f"请填写场景环境参数：{self._current_scenario()}"

    def shutdown(self) -> None:
        self._stop_trial.set()
        self._preview_stop.set()
        if self._trial_thread and self._trial_thread.is_alive():
            self._trial_thread.join(timeout=3.0)
        if self._preview_thread and self._preview_thread.is_alive():
            self._preview_thread.join(timeout=3.0)
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _preview_loop(self) -> None:
        while not self._preview_stop.is_set():
            with self._lock:
                phase = self.phase
                if phase not in ("ready", "need_env") or self.group_idx >= len(self.plan):
                    mode = None
                else:
                    mode = self._current_mode()
            if mode is None:
                time.sleep(0.05)
                continue
            try:
                assert self._capture is not None
                bundle = self._capture.read()
                pipe = self._get_pipe(mode)
                primary, err, tid, mean_iou = pipe.process_frame(bundle.rgb, bundle.temp)
                fm = {
                    "detected": primary is not None,
                    "center_error_px": err,
                    "fusion_match_iou": mean_iou,
                    "track_id": tid,
                }
                vis = _render_eval_frame_fast(
                    bundle.rgb, bundle.temp, primary, fm, 0, self.cfg.frames
                )
                with self._lock:
                    self.live["preview_b64"] = _encode_b64(vis)
                    self.live["frame_idx"] = 0
                    self.live["total_frames"] = self.cfg.frames
            except Exception as exc:
                with self._lock:
                    self.status_msg = f"预览异常: {exc}"
            time.sleep(1.0 / max(CONFIG.send_fps, 1))

    def submit_env(self, lux_raw: str, notes: str) -> None:
        with self._lock:
            if self.phase not in ("need_env", "ready"):
                raise ValueError("当前阶段不可提交环境参数")
            scenario = self._current_scenario()
            lux = float(lux_raw) if lux_raw.strip() else float("nan")
            self.env_records[scenario] = {"scenario": scenario, "lux": lux, "notes": notes.strip()}
            self.phase = "ready"
            self.status_msg = (
                f"场景 {scenario} · 模式 {MODE_LABELS[self._current_mode()]} · "
                f"Trial {self.trial_idx + 1}/{self.cfg.trials} — 请操作员站入 1.5m 画面中央"
            )

    def start_trial(self) -> None:
        with self._lock:
            if self.phase != "ready":
                raise ValueError("请先填写环境参数并等待就绪")
            if self._trial_thread and self._trial_thread.is_alive():
                raise ValueError("已有 trial 正在运行")
            scenario = self._current_scenario()
            if scenario not in self.env_records:
                raise ValueError("请先提交当前场景的环境参数")
            self.phase = "running"
            self._stop_trial.clear()
            self.status_msg = "采集中…"
        self._trial_thread = threading.Thread(target=self._run_trial_worker, daemon=True)
        self._trial_thread.start()

    def stop_trial(self) -> None:
        self._stop_trial.set()

    def _run_trial_worker(self) -> None:
        scenario, mode = self._current_group()
        env = self.env_records[scenario]
        lux = env["lux"]
        notes = env["notes"]
        cfg = EvalConfig(mode=mode)
        running_detected = 0
        running_count = 0
        t0 = time.perf_counter()

        def on_frame(fi, total, rgb, temp, primary, fm, pipe) -> None:
            nonlocal running_detected, running_count, t0
            running_count = fi + 1
            if fm.get("detected"):
                running_detected += 1
            elapsed = max(time.perf_counter() - t0, 1e-6)
            vis = _render_eval_frame_fast(rgb, temp, primary, fm, fi, total)
            b64 = _encode_b64(vis)
            with self._lock:
                self.live.update(
                    {
                        "preview_b64": b64,
                        "frame_idx": fi + 1,
                        "total_frames": total,
                        "running_det_rate": running_detected / max(1, running_count),
                        "running_fps": running_count / elapsed,
                    }
                )

        try:
            assert self._capture is not None
            pipe = self._get_pipe(mode)
            result = run_trial(
                self._capture,
                cfg,
                scenario,
                self.trial_idx,
                frames=self.cfg.frames,
                env_lux=lux,
                notes=notes,
                on_frame=on_frame,
                should_stop=self._stop_trial.is_set,
                pipe=pipe,
            )
        except Exception as exc:
            with self._lock:
                self.status_msg = f"Trial 失败: {exc}"
                self.phase = "ready"
            return

        with self._lock:
            self.all_trials.append(result)
            self.live.update(
                {
                    "frame_idx": 0,
                    "total_frames": self.cfg.frames,
                    "running_det_rate": 0.0,
                    "running_fps": 0.0,
                }
            )
            self.trial_idx += 1
            if self.trial_idx >= self.cfg.trials:
                self.trial_idx = 0
                self.group_idx += 1
                if self.group_idx >= len(self.plan):
                    self.phase = "done"
                    self.status_msg = "全部实验完成"
                    self._save_results()
                    return
                next_scenario = self._current_scenario()
                if next_scenario not in self.env_records:
                    self.phase = "need_env"
                    self.status_msg = f"请填写场景环境参数：{next_scenario}"
                else:
                    self.phase = "ready"
                    self.status_msg = (
                        f"场景 {next_scenario} · 模式 {MODE_LABELS[self._current_mode()]} · "
                        f"Trial {self.trial_idx + 1}/{self.cfg.trials} — 请操作员就位"
                    )
            else:
                self.phase = "ready"
                self.status_msg = (
                    f"场景 {scenario} · 模式 {MODE_LABELS[mode]} · "
                    f"Trial {self.trial_idx + 1}/{self.cfg.trials} — 请操作员就位"
                )

    def _save_results(self) -> None:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        summary = aggregate_trials(self.all_trials)
        payload = {
            "meta": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trials_per_group": self.cfg.trials,
                "frames_per_trial": self.cfg.frames,
                "protocol": "presence_based",
                "distance_m": 1.5,
                "source": "eval_nine_groups_web",
            },
            "environment": list(self.env_records.values()),
            "trials": [asdict(t) for t in self.all_trials],
            "summary": summary,
        }
        json_path = self.cfg.out_dir / "benchmark_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)

        csv_path = self.cfg.out_dir / "experiment_log.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "scenario",
                    "mode",
                    "trial_id",
                    "detection_rate",
                    "center_error_px",
                    "fps",
                    "lost_count",
                    "recovery_ms",
                    "fusion_match_iou_mean",
                    "env_lux",
                ]
            )
            for t in self.all_trials:
                w.writerow(
                    [
                        t.scenario,
                        t.mode,
                        t.trial_id,
                        f"{t.detection_rate:.4f}",
                        f"{t.center_error_px:.2f}" if t.center_error_px == t.center_error_px else "",
                        f"{t.fps:.2f}",
                        t.lost_count,
                        f"{t.recovery_ms:.1f}" if t.recovery_ms == t.recovery_ms else "",
                        f"{t.fusion_match_iou_mean:.3f}"
                        if t.fusion_match_iou_mean == t.fusion_match_iou_mean
                        else "",
                        t.env_lux,
                    ]
                )
        self._saved_paths = {"json": str(json_path), "csv": str(csv_path)}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            scenario, mode = self._current_group() if self.group_idx < len(self.plan) else ("-", "-")
            progress = self._build_progress()
            last = self.all_trials[-1] if self.all_trials else None
            raw = {
                "phase": self.phase,
                "status_msg": self.status_msg,
                "scenario": scenario,
                "mode": mode,
                "scenario_hint": SCENARIO_HINTS.get(scenario, ""),
                "trial_idx": self.trial_idx,
                "trials_per_group": self.cfg.trials,
                "frames_per_trial": self.cfg.frames,
                "group_idx": self.group_idx,
                "group_total": len(self.plan),
                "plan": [{"scenario": s, "mode": m} for s, m in self.plan],
                "env_records": _json_safe(dict(self.env_records)),
                "progress": progress,
                "live": dict(self.live),
                "last_trial": _trial_public(last) if last else None,
                "recent_trials": [_trial_public(t) for t in self.all_trials[-20:]],
                "summary": _json_safe(aggregate_trials(self.all_trials)) if self.all_trials else {},
                "saved_paths": getattr(self, "_saved_paths", None),
            }
            return _json_safe(raw)

    def _build_progress(self) -> List[Dict[str, Any]]:
        counts: Dict[Tuple[str, str], int] = {}
        for t in self.all_trials:
            key = (t.scenario, t.mode)
            counts[key] = counts.get(key, 0) + 1
        out = []
        for s, m in self.plan:
            done = counts.get((s, m), 0)
            cur = self.group_idx == self.plan.index((s, m))
            out.append(
                {
                    "scenario": s,
                    "mode": m,
                    "done": done,
                    "total": self.cfg.trials,
                    "current": cur and self.phase in ("ready", "running", "need_env"),
                    "complete": done >= self.cfg.trials,
                }
            )
        return out


session: Optional[ExperimentSession] = None


def build_ws_message() -> str:
    assert session is not None
    snap = session.snapshot()
    return json.dumps({"type": "state", **snap})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    session = ExperimentSession(CONFIG)
    await asyncio.to_thread(session.start)
    try:
        yield
    finally:
        if session is not None:
            await asyncio.to_thread(session.shutdown)
        session = None


app = FastAPI(title="九组对比实验", lifespan=lifespan)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(HTML_PAGE)


@app.get("/api/state")
def api_state() -> JSONResponse:
    if session is None:
        raise HTTPException(503, "session not ready")
    return JSONResponse(session.snapshot())


@app.post("/api/env")
async def api_env(body: Dict[str, str]) -> JSONResponse:
    if session is None:
        raise HTTPException(503, "session not ready")
    try:
        await asyncio.to_thread(session.submit_env, body.get("lux", ""), body.get("notes", ""))
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({"ok": True, **session.snapshot()})


@app.post("/api/trial/start")
async def api_trial_start() -> JSONResponse:
    if session is None:
        raise HTTPException(503, "session not ready")
    try:
        await asyncio.to_thread(session.start_trial)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return JSONResponse({"ok": True})


@app.post("/api/trial/stop")
async def api_trial_stop() -> JSONResponse:
    if session is None:
        raise HTTPException(503, "session not ready")
    await asyncio.to_thread(session.stop_trial)
    return JSONResponse({"ok": True})


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    stop = asyncio.Event()
    interval = 1.0 / max(1.0, CONFIG.send_fps)

    async def sender() -> None:
        while not stop.is_set():
            try:
                msg = await asyncio.to_thread(build_ws_message)
                await ws.send_text(msg)
            except Exception as exc:
                print(f"[ws] push failed: {exc}", file=sys.stderr)
                await asyncio.sleep(interval)
                continue
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
<title>T9 九组对比实验</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; font-family:system-ui,sans-serif; background:#0b1220; color:#e5e7eb;
    min-height:100vh; display:flex; flex-direction:column; }
  header { padding:10px 16px; border-bottom:1px solid rgba(255,255,255,.08);
    display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  h1 { margin:0; font-size:17px; }
  .pill { font:11px/1.5 monospace; padding:3px 10px; border-radius:999px; background:#1f2937; color:#cbd5e1; }
  .pill.ok { background:#14532d; color:#bbf7d0; }
  .pill.run { background:#1e3a5f; color:#93c5fd; }
  .pill.done { background:#422006; color:#fde68a; }
  .main { flex:1; display:grid; grid-template-columns:minmax(0,1.2fr) minmax(320px,0.8fr);
    gap:12px; padding:12px 16px 16px; min-height:0; }
  @media(max-width:960px){ .main { grid-template-columns:1fr; } }
  .card { background:#111827; border:1px solid rgba(255,255,255,.08); border-radius:10px;
    overflow:hidden; display:flex; flex-direction:column; min-height:0; }
  .card h2 { margin:0; padding:8px 12px; font-size:12px; color:#9ca3af;
    border-bottom:1px solid rgba(255,255,255,.06); }
  .preview-wrap { flex:1; min-height:280px; background:#000; display:flex; align-items:center;
    justify-content:center; position:relative; }
  .preview-wrap img { max-width:100%; max-height:100%; display:block; image-rendering:auto; }
  .progress-bar { height:4px; background:#1f2937; }
  .progress-bar > div { height:100%; background:#3b82f6; transition:width .15s; }
  .hud { padding:8px 12px; font:12px/1.5 ui-monospace,monospace; color:#94a3b8;
    border-top:1px solid rgba(255,255,255,.06); display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; }
  .hud b { color:#93c5fd; }
  .side { display:flex; flex-direction:column; gap:10px; min-height:0; overflow:auto; }
  .env-form { padding:10px 12px; display:flex; flex-direction:column; gap:8px; }
  .env-form label { font-size:11px; color:#9ca3af; }
  .env-form input, .env-form textarea { background:#0f172a; border:1px solid #334155; color:#e2e8f0;
    border-radius:6px; padding:6px 8px; font-size:13px; }
  .env-form textarea { min-height:52px; resize:vertical; }
  .hint { font-size:11px; color:#6b7280; margin:0; padding:0 12px 8px; }
  .btn-row { padding:0 12px 12px; display:flex; gap:8px; flex-wrap:wrap; }
  button { border:none; border-radius:8px; padding:8px 14px; font-size:13px; cursor:pointer; font-weight:600; }
  .btn-primary { background:#2563eb; color:#fff; }
  .btn-primary:disabled { background:#374151; color:#6b7280; cursor:not-allowed; }
  .btn-danger { background:#7f1d1d; color:#fecaca; }
  .btn-danger:disabled { opacity:.4; cursor:not-allowed; }
  .grid9 { display:grid; grid-template-columns:repeat(3,1fr); gap:4px; padding:8px 10px 10px; }
  .cell { font:10px/1.3 monospace; text-align:center; padding:6px 2px; border-radius:6px;
    background:#1f2937; color:#6b7280; border:1px solid transparent; }
  .cell.cur { border-color:#3b82f6; color:#93c5fd; }
  .cell.ok { background:#14532d; color:#bbf7d0; }
  .log { flex:1; min-height:120px; max-height:220px; overflow:auto; font:10px/1.4 monospace;
    padding:6px 10px; color:#9ca3af; }
  .log div { padding:2px 0; border-bottom:1px solid rgba(255,255,255,.04); }
  .saved { padding:8px 12px; font-size:11px; color:#86efac; }
</style>
</head>
<body>
<header>
  <h1>T9 九组对比实验</h1>
  <span id="badge" class="pill">连接中</span>
  <span id="phasePill" class="pill">-</span>
  <span id="posPill" class="pill">-</span>
</header>
<div class="main">
  <div class="card">
    <h2>实时预览 · 检测框 + 画面中心</h2>
    <div class="progress-bar"><div id="frameProg" style="width:0%"></div></div>
    <div class="preview-wrap"><img id="preview" alt="preview"/></div>
    <div class="hud">
      <div><b>状态</b> <span id="statusMsg">-</span></div>
      <div><b>帧进度</b> <span id="frameInfo">-</span></div>
      <div><b>实时检出率</b> <span id="runDet">-</span></div>
      <div><b>实时 FPS</b> <span id="runFps">-</span></div>
      <div><b>最近 trial</b> <span id="lastTrial">-</span></div>
      <div><b>组进度</b> <span id="groupInfo">-</span></div>
    </div>
  </div>
  <div class="side">
    <div class="card" id="envCard">
      <h2>场景环境 · <span id="envScenario">-</span></h2>
      <p class="hint" id="envHint">-</p>
      <div class="env-form">
        <label>照度 lux（未知留空）</label>
        <input id="luxInput" type="text" inputmode="decimal" placeholder="例如 250 或 5"/>
        <label>备注</label>
        <textarea id="notesInput" placeholder="可选"></textarea>
      </div>
      <div class="btn-row">
        <button class="btn-primary" id="btnEnv" onclick="submitEnv()">确认环境</button>
      </div>
    </div>
    <div class="card">
      <h2>操作</h2>
      <p class="hint">操作员站 1.5m 标定距离、画面中央后点击开始采集</p>
      <div class="btn-row">
        <button class="btn-primary" id="btnStart" onclick="startTrial()">开始采集</button>
        <button class="btn-danger" id="btnStop" onclick="stopTrial()" disabled>中止</button>
      </div>
    </div>
    <div class="card">
      <h2>九组进度</h2>
      <div class="grid9" id="grid9"></div>
    </div>
    <div class="card" style="flex:1">
      <h2>Trial 日志</h2>
      <div class="log" id="log"></div>
      <div class="saved" id="saved" style="display:none"></div>
    </div>
  </div>
</div>
<script>
const ws = new WebSocket((location.protocol==='https:'?'wss':'ws')+'://'+location.host+'/ws');
ws.onopen = () => { document.getElementById('badge').textContent='已连接'; document.getElementById('badge').className='pill ok'; };
ws.onclose = () => { document.getElementById('badge').textContent='已断开'; document.getElementById('badge').className='pill'; };

function modeLabel(m){ return ({rgb:'RGB',ir:'IR',fusion:'Fus'})[m]||m; }
function scenShort(s){ return s.replace('S1_','').replace('S2_','').replace('S3_',''); }

function applyState(s){
  const phase = s.phase||'-';
  const pp = document.getElementById('phasePill');
  pp.textContent = phase;
  pp.className = 'pill ' + (phase==='running'?'run':phase==='done'?'done':'');
  document.getElementById('posPill').textContent =
    `${s.scenario||'-'} / ${(s.mode||'-').toUpperCase()} · T${(s.trial_idx||0)+1}/${s.trials_per_group||'-'}`;
  document.getElementById('statusMsg').textContent = s.status_msg||'-';
  document.getElementById('groupInfo').textContent = `${(s.group_idx||0)+1} / ${s.group_total||'-'}`;

  const live = s.live||{};
  if(live.preview_b64) document.getElementById('preview').src = live.preview_b64;
  const fi = live.frame_idx||0, tf = live.total_frames||1;
  document.getElementById('frameInfo').textContent = `${fi} / ${tf}`;
  document.getElementById('frameProg').style.width = (100*fi/tf)+'%';
  document.getElementById('runDet').textContent =
    phase==='running' ? ((live.running_det_rate||0)*100).toFixed(1)+'%' : '-';
  document.getElementById('runFps').textContent =
    phase==='running' ? (live.running_fps||0).toFixed(1) : '-';

  const lt = s.last_trial;
  document.getElementById('lastTrial').textContent = lt
    ? `det=${(lt.detection_rate*100).toFixed(1)}% err=${lt.center_error_px?.toFixed?.(1)||'-'}px fps=${lt.fps?.toFixed?.(1)||'-'}`
    : '-';

  document.getElementById('envScenario').textContent = s.scenario||'-';
  document.getElementById('envHint').textContent = s.scenario_hint||'';
  const envCard = document.getElementById('envCard');
  envCard.style.display = (phase==='need_env') ? 'flex' : 'none';

  const btnEnv = document.getElementById('btnEnv');
  const btnStart = document.getElementById('btnStart');
  const btnStop = document.getElementById('btnStop');
  btnStart.disabled = phase !== 'ready';
  btnStop.disabled = phase !== 'running';
  btnEnv.disabled = phase !== 'need_env';

  const grid = document.getElementById('grid9');
  grid.innerHTML = (s.progress||[]).map(p => {
    let cls = 'cell';
    if(p.complete) cls += ' ok';
    else if(p.current) cls += ' cur';
    return `<div class="${cls}">${scenShort(p.scenario)}<br>${modeLabel(p.mode)}<br>${p.done}/${p.total}</div>`;
  }).join('');

  const log = document.getElementById('log');
  log.innerHTML = (s.recent_trials||[]).slice().reverse().map(t =>
    `<div>${t.scenario}/${t.mode} #${t.trial_id} det=${(t.detection_rate*100).toFixed(1)}% `
    + `err=${t.center_error_px?.toFixed?.(1)||'-'}px fps=${t.fps?.toFixed?.(1)||'-'} lost=${t.lost_count}</div>`
  ).join('');

  const saved = document.getElementById('saved');
  if(s.saved_paths){
    saved.style.display='block';
    saved.textContent = '已保存: '+s.saved_paths.json+' · '+s.saved_paths.csv;
  }
}

ws.onmessage = (ev) => {
  let m; try{ m=JSON.parse(ev.data); }catch(e){ return; }
  if(m.type!=='state') return;
  applyState(m);
};

async function post(url, body){
  const r = await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'},
    body: body?JSON.stringify(body):'{}'});
  if(!r.ok){ const t=await r.text(); alert(t||r.statusText); return null; }
  return r.json();
}
function submitEnv(){
  post('/api/env', {lux: document.getElementById('luxInput').value, notes: document.getElementById('notesInput').value});
}
function startTrial(){ post('/api/trial/start'); }
function stopTrial(){ post('/api/trial/stop'); }
</script>
</body>
</html>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="九组对比实验 Web 可视化")
    p.add_argument("--scenario", choices=SCENARIOS)
    p.add_argument("--mode", choices=MODES)
    p.add_argument("--all", action="store_true", help="跑完全部 9 组")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--out", type=Path, default=_REPO / "experiments" / "records")
    p.add_argument("--host", default=CONFIG.host)
    p.add_argument("--port", type=int, default=CONFIG.port)
    return p.parse_args(argv)


def main(argv=None):
    ns = parse_args(argv)
    if not ns.all and (not ns.scenario or not ns.mode):
        raise SystemExit("请指定 --scenario + --mode，或使用 --all")

    CONFIG.host = ns.host
    CONFIG.port = ns.port
    CONFIG.trials = ns.trials
    CONFIG.frames = ns.frames
    CONFIG.out_dir = ns.out
    CONFIG.scenarios = list(SCENARIOS) if ns.all else [ns.scenario]
    CONFIG.modes = list(MODES) if ns.all else [ns.mode]

    import uvicorn

    plan_n = len(CONFIG.scenarios) * len(CONFIG.modes)
    print(
        f"九组实验 Web：http://{ns.host}:{ns.port}/\n"
        f"  计划 {plan_n} 组 × {CONFIG.trials} trials × {CONFIG.frames} 帧\n"
        f"  输出目录: {CONFIG.out_dir}",
        file=sys.stderr,
    )
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")


if __name__ == "__main__":
    main()
