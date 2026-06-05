#!/usr/bin/env python3
"""消融实验 A1/A2/A3 — Web 可视化版（Fusion 恶劣场景）。

对照组:
  baseline           — 全功能
  A1_no_registration — IR 框不投影
  A2_fixed_weights   — 固定权重 0.5/0.5
  A3_no_tracker      — 关闭跟踪器

用法（板端）::

    python3 scripts/eval_ablation_web.py --all-scenarios --trials 10 --out experiments/records/
    python3 scripts/eval_ablation_web.py --scenario S2_low_light --port 8006

浏览器打开 http://<板子IP>:8006/
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

from scripts.dual_fusion_web import COLOR_PRIMARY, _temp_colormap  # noqa: E402
from src.detection.types import DetectionBox  # noqa: E402
from src.experiments.camera_live import DualEvalCapture  # noqa: E402
from src.experiments.pipeline import (  # noqa: E402
    RGB_CENTER,
    EvalConfig,
    EvalPipeline,
    TrialResult,
    run_trial,
)

ABLATIONS: Dict[str, EvalConfig] = {
    "baseline": EvalConfig(mode="fusion"),
    "A1_no_registration": EvalConfig(mode="fusion", no_registration=True),
    "A2_fixed_weights": EvalConfig(mode="fusion", fixed_weights=(0.5, 0.5)),
    "A3_no_tracker": EvalConfig(mode="fusion", no_tracker=True),
}
ABLATION_LABELS = {
    "baseline": "Baseline",
    "A1_no_registration": "A1 无配准",
    "A2_fixed_weights": "A2 固定权重",
    "A3_no_tracker": "A3 无跟踪",
}
ABLATION_HINTS = {
    "baseline": "全功能 Fusion（对照组）",
    "A1_no_registration": "IR 框不投影到 RGB，故意错误对齐",
    "A2_fixed_weights": "融合权重固定 w_rgb=w_ir=0.5",
    "A3_no_tracker": "关闭 TargetTracker，仅融合检测",
}
DEFAULT_SCENARIOS = ["S2_low_light", "S3_backlight"]
ALL_SCENARIOS = ["S1_normal", "S2_low_light", "S3_backlight"]
SCENARIO_HINTS = {
    "S1_normal": "正常光照 >200 lux，操作员站 1.5m 画面中央",
    "S2_low_light": "低照度 <10 lux，操作员站 1.5m 画面中央",
    "S3_backlight": "强背光/逆光，RGB 局部过曝，操作员站 1.5m",
}


@dataclass
class WebConfig:
    host: str = "0.0.0.0"
    port: int = 8006
    trials: int = 10
    frames: int = 90
    out_dir: Path = _REPO / "experiments" / "records"
    send_fps: float = 8.0
    jpeg_quality: int = 78
    scenarios: List[str] = field(default_factory=lambda: list(DEFAULT_SCENARIOS))


CONFIG = WebConfig()


def _json_safe(obj: Any) -> Any:
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
    def __init__(self, inner: DualEvalCapture, lock: threading.Lock) -> None:
        self._inner = inner
        self._lock = lock

    def read(self):
        with self._lock:
            return self._inner.read()

    def release(self) -> None:
        self._inner.release()


def _record_public(rec: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(rec)
    d.pop("frame_metrics", None)
    return _json_safe(d)


def _encode_b64(bgr: Optional[np.ndarray]) -> Optional[str]:
    if bgr is None:
        return None
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), CONFIG.jpeg_quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("ascii")


def _render_eval_frame_fast(
    rgb: np.ndarray,
    temp: np.ndarray,
    primary: Optional[DetectionBox],
    fm: Dict[str, Any],
    frame_idx: int,
    total_frames: int,
) -> np.ndarray:
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
    iou = fm.get("fusion_match_iou", float("nan"))
    iou_txt = f" iou={iou:.2f}" if not math.isnan(iou) else ""
    tid = fm.get("track_id")
    tid_txt = f" T{tid}" if tid is not None else ""
    hud = f"[{frame_idx + 1}/{total_frames}] {det} err={err_txt}{iou_txt}{tid_txt}"
    cv2.putText(vis, hud, (8, vis.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1, cv2.LINE_AA)
    return vis


class AblationSession:
    """消融实验状态机。"""

    def __init__(self, cfg: WebConfig) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.plan: List[Tuple[str, str]] = [
            (scenario, ab_name) for scenario in cfg.scenarios for ab_name in ABLATIONS
        ]
        self.group_idx = 0
        self.trial_idx = 0
        self.phase = "init"
        self.status_msg = "正在初始化相机…"
        self.env_records: Dict[str, Dict[str, Any]] = {}
        self.all_records: List[Dict[str, Any]] = []
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

    def _current_ablation(self) -> str:
        return self.plan[self.group_idx][1]

    def _get_pipe(self, ablation: str) -> EvalPipeline:
        if ablation not in self._pipe_cache:
            self._pipe_cache[ablation] = EvalPipeline(ABLATIONS[ablation])
        return self._pipe_cache[ablation]

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
                    ablation = None
                else:
                    ablation = self._current_ablation()
            if ablation is None:
                time.sleep(0.05)
                continue
            try:
                assert self._capture is not None
                bundle = self._capture.read()
                pipe = self._get_pipe(ablation)
                primary, err, tid, mean_iou = pipe.process_frame(bundle.rgb, bundle.temp)
                fm = {
                    "detected": primary is not None,
                    "center_error_px": err,
                    "fusion_match_iou": mean_iou,
                    "track_id": tid,
                }
                vis = _render_eval_frame_fast(bundle.rgb, bundle.temp, primary, fm, 0, self.cfg.frames)
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
            ab = self._current_ablation()
            self.status_msg = (
                f"场景 {scenario} · {ABLATION_LABELS[ab]} · "
                f"Trial {self.trial_idx + 1}/{self.cfg.trials} — 请操作员站入 1.5m 画面中央"
            )

    def start_trial(self) -> None:
        with self._lock:
            if self.phase != "ready":
                raise ValueError("请先填写环境参数并等待就绪")
            if self._trial_thread and self._trial_thread.is_alive():
                raise ValueError("已有 trial 正在运行")
            if self._current_scenario() not in self.env_records:
                raise ValueError("请先提交当前场景的环境参数")
            self.phase = "running"
            self._stop_trial.clear()
            self.status_msg = "采集中…"
        self._trial_thread = threading.Thread(target=self._run_trial_worker, daemon=True)
        self._trial_thread.start()

    def stop_trial(self) -> None:
        self._stop_trial.set()

    def _run_trial_worker(self) -> None:
        scenario, ab_name = self._current_group()
        env = self.env_records[scenario]
        lux = env["lux"]
        cfg = ABLATIONS[ab_name]
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
            with self._lock:
                self.live.update(
                    {
                        "preview_b64": _encode_b64(vis),
                        "frame_idx": fi + 1,
                        "total_frames": total,
                        "running_det_rate": running_detected / max(1, running_count),
                        "running_fps": running_count / elapsed,
                    }
                )

        try:
            assert self._capture is not None
            pipe = self._get_pipe(ab_name)
            result = run_trial(
                self._capture,
                cfg,
                scenario,
                self.trial_idx,
                frames=self.cfg.frames,
                env_lux=lux,
                notes=ab_name,
                on_frame=on_frame,
                should_stop=self._stop_trial.is_set,
                pipe=pipe,
            )
        except Exception as exc:
            with self._lock:
                self.status_msg = f"Trial 失败: {exc}"
                self.phase = "ready"
            return

        record = {**asdict(result), "ablation": ab_name}
        with self._lock:
            self.all_records.append(record)
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
                    self.status_msg = "全部消融实验完成"
                    self._save_results()
                    return
                next_scenario = self._current_scenario()
                if next_scenario not in self.env_records:
                    self.phase = "need_env"
                    self.status_msg = f"请填写场景环境参数：{next_scenario}"
                else:
                    ab = self._current_ablation()
                    self.phase = "ready"
                    self.status_msg = (
                        f"场景 {next_scenario} · {ABLATION_LABELS[ab]} · "
                        f"Trial {self.trial_idx + 1}/{self.cfg.trials} — 请操作员就位"
                    )
            else:
                self.phase = "ready"
                self.status_msg = (
                    f"场景 {scenario} · {ABLATION_LABELS[ab_name]} · "
                    f"Trial {self.trial_idx + 1}/{self.cfg.trials} — 请操作员就位"
                )

    def _save_results(self) -> None:
        self.cfg.out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "trials_per_group": self.cfg.trials,
                "frames_per_trial": self.cfg.frames,
                "scenarios": self.cfg.scenarios,
                "ablations": list(ABLATIONS.keys()),
                "source": "eval_ablation_web",
            },
            "environment": list(self.env_records.values()),
            "records": self.all_records,
        }
        json_path = self.cfg.out_dir / "ablation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2)

        csv_path = self.cfg.out_dir / "ablation_log.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "scenario",
                    "ablation",
                    "trial_id",
                    "detection_rate",
                    "center_error_px",
                    "fps",
                    "lost_count",
                    "recovery_ms",
                    "fusion_match_iou_mean",
                    "id_switch_count",
                    "env_lux",
                ]
            )
            for r in self.all_records:
                w.writerow(
                    [
                        r["scenario"],
                        r["ablation"],
                        r["trial_id"],
                        f"{r['detection_rate']:.4f}",
                        f"{r['center_error_px']:.2f}"
                        if r["center_error_px"] == r["center_error_px"]
                        else "",
                        f"{r['fps']:.2f}",
                        r["lost_count"],
                        f"{r['recovery_ms']:.1f}" if r["recovery_ms"] == r["recovery_ms"] else "",
                        f"{r['fusion_match_iou_mean']:.3f}"
                        if r["fusion_match_iou_mean"] == r["fusion_match_iou_mean"]
                        else "",
                        r.get("id_switch_count", 0),
                        r["env_lux"],
                    ]
                )
        self._saved_paths = {"json": str(json_path), "csv": str(csv_path)}

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            if self.group_idx < len(self.plan):
                scenario, ablation = self._current_group()
            else:
                scenario, ablation = "-", "-"
            last = self.all_records[-1] if self.all_records else None
            raw = {
                "phase": self.phase,
                "status_msg": self.status_msg,
                "scenario": scenario,
                "ablation": ablation,
                "ablation_label": ABLATION_LABELS.get(ablation, ablation),
                "ablation_hint": ABLATION_HINTS.get(ablation, ""),
                "scenario_hint": SCENARIO_HINTS.get(scenario, ""),
                "trial_idx": self.trial_idx,
                "trials_per_group": self.cfg.trials,
                "frames_per_trial": self.cfg.frames,
                "group_idx": self.group_idx,
                "group_total": len(self.plan),
                "plan": [{"scenario": s, "ablation": a} for s, a in self.plan],
                "env_records": _json_safe(dict(self.env_records)),
                "progress": self._build_progress(),
                "live": dict(self.live),
                "last_trial": _record_public(last) if last else None,
                "recent_trials": [_record_public(r) for r in self.all_records[-20:]],
                "saved_paths": getattr(self, "_saved_paths", None),
            }
            return _json_safe(raw)

    def _build_progress(self) -> List[Dict[str, Any]]:
        counts: Dict[Tuple[str, str], int] = {}
        for r in self.all_records:
            key = (r["scenario"], r["ablation"])
            counts[key] = counts.get(key, 0) + 1
        out = []
        for idx, (s, a) in enumerate(self.plan):
            done = counts.get((s, a), 0)
            out.append(
                {
                    "scenario": s,
                    "ablation": a,
                    "ablation_label": ABLATION_LABELS.get(a, a),
                    "done": done,
                    "total": self.cfg.trials,
                    "current": idx == self.group_idx and self.phase in ("ready", "running", "need_env"),
                    "complete": done >= self.cfg.trials,
                }
            )
        return out


session: Optional[AblationSession] = None


def build_ws_message() -> str:
    assert session is not None
    return json.dumps({"type": "state", **session.snapshot()})


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    session = AblationSession(CONFIG)
    await asyncio.to_thread(session.start)
    try:
        yield
    finally:
        if session is not None:
            await asyncio.to_thread(session.shutdown)
        session = None


app = FastAPI(title="消融实验", lifespan=lifespan)


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
                await ws.send_text(await asyncio.to_thread(build_ws_message))
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
<title>消融实验 A1/A2/A3</title>
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
  .preview-wrap { flex:1; min-height:280px; background:#000; display:flex; align-items:center; justify-content:center; }
  .preview-wrap img { max-width:100%; max-height:100%; display:block; }
  .progress-bar { height:4px; background:#1f2937; }
  .progress-bar > div { height:100%; background:#8b5cf6; transition:width .15s; }
  .hud { padding:8px 12px; font:12px/1.5 ui-monospace,monospace; color:#94a3b8;
    border-top:1px solid rgba(255,255,255,.06); display:grid; grid-template-columns:1fr 1fr; gap:4px 12px; }
  .hud b { color:#c4b5fd; }
  .side { display:flex; flex-direction:column; gap:10px; min-height:0; overflow:auto; }
  .env-form { padding:10px 12px; display:flex; flex-direction:column; gap:8px; }
  .env-form label { font-size:11px; color:#9ca3af; }
  .env-form input, .env-form textarea { background:#0f172a; border:1px solid #334155; color:#e2e8f0;
    border-radius:6px; padding:6px 8px; font-size:13px; }
  .hint { font-size:11px; color:#6b7280; margin:0; padding:0 12px 8px; }
  .btn-row { padding:0 12px 12px; display:flex; gap:8px; flex-wrap:wrap; }
  button { border:none; border-radius:8px; padding:8px 14px; font-size:13px; cursor:pointer; font-weight:600; }
  .btn-primary { background:#7c3aed; color:#fff; }
  .btn-primary:disabled { background:#374151; color:#6b7280; cursor:not-allowed; }
  .btn-danger { background:#7f1d1d; color:#fecaca; }
  .btn-danger:disabled { opacity:.4; cursor:not-allowed; }
  .grid-progress { display:grid; grid-template-columns:repeat(4,1fr); gap:4px; padding:8px 10px 10px; }
  @media(max-width:700px){ .grid-progress { grid-template-columns:repeat(2,1fr); } }
  .cell { font:9px/1.3 monospace; text-align:center; padding:6px 2px; border-radius:6px;
    background:#1f2937; color:#6b7280; border:1px solid transparent; }
  .cell.cur { border-color:#8b5cf6; color:#c4b5fd; }
  .cell.ok { background:#14532d; color:#bbf7d0; }
  .log { flex:1; min-height:120px; max-height:220px; overflow:auto; font:10px/1.4 monospace;
    padding:6px 10px; color:#9ca3af; }
  .log div { padding:2px 0; border-bottom:1px solid rgba(255,255,255,.04); }
  .saved { padding:8px 12px; font-size:11px; color:#86efac; }
  .ablation-desc { padding:0 12px 8px; font-size:11px; color:#a78bfa; }
</style>
</head>
<body>
<header>
  <h1>消融实验 Fusion · A1/A2/A3</h1>
  <span id="badge" class="pill">连接中</span>
  <span id="phasePill" class="pill">-</span>
  <span id="posPill" class="pill">-</span>
</header>
<div class="main">
  <div class="card">
    <h2>实时预览 · Fusion 主目标</h2>
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
    <div class="card">
      <h2>当前消融 · <span id="ablationLabel">-</span></h2>
      <p class="ablation-desc" id="ablationHint">-</p>
    </div>
    <div class="card" id="envCard">
      <h2>场景环境 · <span id="envScenario">-</span></h2>
      <p class="hint" id="envHint">-</p>
      <div class="env-form">
        <label>照度 lux（未知留空）</label>
        <input id="luxInput" type="text" inputmode="decimal" placeholder="例如 5"/>
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
      <h2>消融进度（场景 × 对照/A1/A2/A3）</h2>
      <div class="grid-progress" id="gridProgress"></div>
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

function scenShort(s){ return (s||'-').replace('S1_','').replace('S2_','').replace('S3_',''); }

function applyState(s){
  const phase = s.phase||'-';
  const pp = document.getElementById('phasePill');
  pp.textContent = phase;
  pp.className = 'pill ' + (phase==='running'?'run':phase==='done'?'done':'');
  document.getElementById('posPill').textContent =
    `${s.scenario||'-'} / ${s.ablation_label||'-'} · T${(s.trial_idx||0)+1}/${s.trials_per_group||'-'}`;
  document.getElementById('statusMsg').textContent = s.status_msg||'-';
  document.getElementById('groupInfo').textContent = `${(s.group_idx||0)+1} / ${s.group_total||'-'}`;
  document.getElementById('ablationLabel').textContent = s.ablation_label||'-';
  document.getElementById('ablationHint').textContent = s.ablation_hint||'-';

  const live = s.live||{};
  if(live.preview_b64) document.getElementById('preview').src = live.preview_b64;
  const fi = live.frame_idx||0, tf = live.total_frames||1;
  document.getElementById('frameInfo').textContent = `${fi} / ${tf}`;
  document.getElementById('frameProg').style.width = (100*fi/tf)+'%';
  document.getElementById('runDet').textContent = phase==='running' ? ((live.running_det_rate||0)*100).toFixed(1)+'%' : '-';
  document.getElementById('runFps').textContent = phase==='running' ? (live.running_fps||0).toFixed(1) : '-';

  const lt = s.last_trial;
  document.getElementById('lastTrial').textContent = lt
    ? `det=${(lt.detection_rate*100).toFixed(1)}% iou=${lt.fusion_match_iou_mean?.toFixed?.(3)||'-'} id_sw=${lt.id_switch_count??0}`
    : '-';

  document.getElementById('envScenario').textContent = s.scenario||'-';
  document.getElementById('envHint').textContent = s.scenario_hint||'';
  document.getElementById('envCard').style.display = (phase==='need_env') ? 'flex' : 'none';

  document.getElementById('btnStart').disabled = phase !== 'ready';
  document.getElementById('btnStop').disabled = phase !== 'running';
  document.getElementById('btnEnv').disabled = phase !== 'need_env';

  document.getElementById('gridProgress').innerHTML = (s.progress||[]).map(p => {
    let cls = 'cell';
    if(p.complete) cls += ' ok';
    else if(p.current) cls += ' cur';
    const ab = (p.ablation_label||p.ablation||'').replace('A1 无配准','A1').replace('A2 固定权重','A2').replace('A3 无跟踪','A3').replace('Baseline','BL');
    return `<div class="${cls}">${scenShort(p.scenario)}<br>${ab}<br>${p.done}/${p.total}</div>`;
  }).join('');

  document.getElementById('log').innerHTML = (s.recent_trials||[]).slice().reverse().map(t =>
    `<div>${t.scenario}/${t.ablation} #${t.trial_id} det=${(t.detection_rate*100).toFixed(1)}% `
    + `iou=${t.fusion_match_iou_mean?.toFixed?.(3)||'-'} id=${t.id_switch_count??0} rec=${t.recovery_ms?.toFixed?.(0)||'-'}ms</div>`
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
  if(!r.ok){ alert(await r.text()||r.statusText); return null; }
  return r.json();
}
function submitEnv(){ post('/api/env', {lux: document.getElementById('luxInput').value, notes: document.getElementById('notesInput').value}); }
function startTrial(){ post('/api/trial/start'); }
function stopTrial(){ post('/api/trial/stop'); }
</script>
</body>
</html>
"""


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="消融实验 Web 可视化")
    p.add_argument("--scenario", choices=ALL_SCENARIOS)
    p.add_argument("--all-scenarios", action="store_true", help="跑 S2 + S3（默认恶劣场景）")
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--frames", type=int, default=90)
    p.add_argument("--out", type=Path, default=_REPO / "experiments" / "records")
    p.add_argument("--host", default=CONFIG.host)
    p.add_argument("--port", type=int, default=CONFIG.port)
    return p.parse_args(argv)


def main(argv=None):
    ns = parse_args(argv)
    if ns.all_scenarios:
        scenarios = list(DEFAULT_SCENARIOS)
    elif ns.scenario:
        scenarios = [ns.scenario]
    else:
        scenarios = ["S2_low_light"]

    CONFIG.host = ns.host
    CONFIG.port = ns.port
    CONFIG.trials = ns.trials
    CONFIG.frames = ns.frames
    CONFIG.out_dir = ns.out
    CONFIG.scenarios = scenarios

    import uvicorn

    plan_n = len(scenarios) * len(ABLATIONS)
    print(
        f"消融实验 Web：http://{ns.host}:{ns.port}/\n"
        f"  场景 {scenarios}\n"
        f"  计划 {plan_n} 组 × {CONFIG.trials} trials × {CONFIG.frames} 帧\n"
        f"  输出: {CONFIG.out_dir}/ablation_results.json",
        file=sys.stderr,
    )
    uvicorn.run(app, host=ns.host, port=ns.port, log_level="info")


if __name__ == "__main__":
    main()
