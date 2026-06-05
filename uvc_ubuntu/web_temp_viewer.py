import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse


WIDTH = 256
HEIGHT = 192
LEN_BYTES = WIDTH * HEIGHT * 2  # 16-bit per pixel

# 画面/温度矩阵对齐：与 img.py 里用 np.roll 修正偏移是同一类问题。
# 为了方便现场调参，这里支持 Web 端热更新（WASD/方向键），默认值仍可用环境变量覆盖。
ROLL_X = int(os.environ.get("THERMAL_ROLL_X", "-92"))
ROLL_Y = int(os.environ.get("THERMAL_ROLL_Y", "0"))

# 温度线性标定：最终温度 = raw_to_celsius(unit) * GAIN + OFFSET
# 用于把“相对正确但偏高/偏低”的读数校到参考温度（例如体温枪/温湿度计）。
TEMP_GAIN = float(os.environ.get("THERMAL_TEMP_GAIN", "1.0"))
TEMP_OFFSET = float(os.environ.get("THERMAL_TEMP_OFFSET", "0.0"))

DEBUG_CONFIG_PATH = Path(os.environ.get("THERMAL_DEBUG_CONFIG", str(Path(__file__).resolve().parent / "thermal_cam_debug.json")))
_save_task: asyncio.Task | None = None

# 两点标定输入（用于落盘，便于复现/给其他程序读取）
KNOWN_CENTER_C: float | None = None
KNOWN_CLICK_C: float | None = None
last_click: tuple[int, int] | None = None


class TempUnit:
    RAW = "raw"
    CK = "cK"   # Kelvin*100
    DC = "dC"   # Celsius*10
    CC = "cC"   # Celsius*100


def guess_unit(raw16: np.ndarray) -> str:
    """
    用分位数而不是均值：热像图里常有少量极热/极冷像素或无效像素，
    会把均值/最大值“拉偏”，导致误判为 cK（典型症状：点位 raw ~5xxx 但显示 -200°C）。
    """
    r = raw16.astype(np.uint16).ravel()
    if r.size == 0:
        return TempUnit.RAW

    p50 = int(np.percentile(r, 50))
    p90 = int(np.percentile(r, 90))
    p99 = int(np.percentile(r, 99))

    # 经验优先级：先看“主体温度”所在的量级（p50/p90），再看极端值（p99）
    # - cK：室温附近 raw ~ 29315；人体表面常见 ~ 30~40℃ -> 303~313K -> raw 30300~31300
    if 26000 < p50 < 34000 and 26000 < p90 < 36000:
        return TempUnit.CK

    # - cC：人体/近距离场景非常常见：raw 3xxx~6xxx（30~60℃）
    #   注意：不要被少量离群热像素（max/p99 很大）误导单位。
    if 2500 <= p50 <= 9000 and p90 <= 12000:
        return TempUnit.CC

    # - dC：更像工业高温场景（几十~几百度），raw 往往整体更大且跨度更宽
    if 800 <= p50 < 20000 and p90 >= 15000:
        return TempUnit.DC

    # 兜底：如果极端分位数仍然落在“万级”，更像 cK；否则偏向 cC（比误判 cK 更安全）
    if 20000 < p99 < 120000:
        return TempUnit.CK

    return TempUnit.CC


def raw_to_celsius_base(v: int, unit: str) -> float:
    if unit == TempUnit.CK:
        return v / 100.0 - 273.15
    if unit == TempUnit.DC:
        return v / 10.0
    if unit == TempUnit.CC:
        return v / 100.0
    return float(v)


def apply_calibration(temp_c: float) -> float:
    return temp_c * TEMP_GAIN + TEMP_OFFSET


def raw_to_celsius(v: int, unit: str) -> float:
    return apply_calibration(raw_to_celsius_base(v, unit))


def build_debug_config() -> dict:
    return {
        "schema": "thermal_cam_debug.v1",
        "updated_unix": time.time(),
        "frame": {"width": WIDTH, "height": HEIGHT},
        "align": {"roll_x": ROLL_X, "roll_y": ROLL_Y},
        "calibration": {
            "gain": TEMP_GAIN,
            "offset": TEMP_OFFSET,
            "two_point": {
                "known_center_c": KNOWN_CENTER_C,
                "known_click_c": KNOWN_CLICK_C,
                "last_click": {"x": last_click[0], "y": last_click[1]} if last_click else None,
            },
        },
        "paths": {"config_file": str(DEBUG_CONFIG_PATH)},
    }


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="thermal_cam_debug.", suffix=".json", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


async def save_debug_config_now() -> None:
    cfg = build_debug_config()
    await asyncio.to_thread(_atomic_write_json, DEBUG_CONFIG_PATH, cfg)


def schedule_save_debug_config(delay_s: float = 0.15) -> None:
    global _save_task

    async def _runner():
        await asyncio.sleep(delay_s)
        await save_debug_config_now()

    try:
        if _save_task and not _save_task.done():
            _save_task.cancel()
    except Exception:
        pass
    _save_task = asyncio.create_task(_runner())


async def load_debug_config() -> None:
    global ROLL_X, ROLL_Y, TEMP_GAIN, TEMP_OFFSET, KNOWN_CENTER_C, KNOWN_CLICK_C, last_click

    def _read() -> dict | None:
        if not DEBUG_CONFIG_PATH.exists():
            return None
        try:
            return json.loads(DEBUG_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            return None

    cfg = await asyncio.to_thread(_read)
    if not cfg:
        return

    try:
        align = cfg.get("align") or {}
        cal = cfg.get("calibration") or {}
        tp = cal.get("two_point") or {}

        if "roll_x" in align:
            ROLL_X = int(align["roll_x"])
        if "roll_y" in align:
            ROLL_Y = int(align["roll_y"])

        if "gain" in cal:
            TEMP_GAIN = float(cal["gain"])
        if "offset" in cal:
            TEMP_OFFSET = float(cal["offset"])

        if "known_center_c" in tp and tp["known_center_c"] is not None:
            KNOWN_CENTER_C = float(tp["known_center_c"])
        if "known_click_c" in tp and tp["known_click_c"] is not None:
            KNOWN_CLICK_C = float(tp["known_click_c"])

        lc = tp.get("last_click")
        if isinstance(lc, dict) and "x" in lc and "y" in lc:
            last_click = (int(lc["x"]), int(lc["y"]))
    except Exception:
        # 配置文件损坏时忽略，继续使用环境变量默认值
        return

def select_endian_and_unit(temp_bytes: bytes) -> tuple[np.ndarray, str, str]:
    """
    返回 (raw16(H,W), unit, endian)。
    先判断字节序（小端/大端）哪个更像“温度值分布”，再做单位猜测。
    """
    le = np.frombuffer(temp_bytes, dtype="<u2").reshape((HEIGHT, WIDTH))
    be = np.frombuffer(temp_bytes, dtype=">u2").reshape((HEIGHT, WIDTH))

    def score(a: np.ndarray) -> float:
        s = a.ravel()
        if s.size == 0:
            return 0.0
        p50 = float(np.percentile(s, 50))
        p90 = float(np.percentile(s, 90))
        # 更像“温度场”的分布：p50/p90 同量级且处于合理区间
        if 26000 < p50 < 34000 and 26000 < p90 < 38000:
            return 4.0
        if 1500 < p50 < 12000 and 1500 < p90 < 20000:
            return 3.0
        if 200 < p50 < 8000 and 200 < p90 < 12000:
            return 2.0
        return 1.0

    sle, sbe = score(le), score(be)
    if sbe > sle:
        raw = be
        endian = "BE"
    else:
        raw = le
        endian = "LE"

    unit = guess_unit(raw)
    return raw, unit, endian


def align_yuyv_and_temp(yuyv: np.ndarray, temp_raw16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ROLL_Y:
        yuyv = np.roll(yuyv, ROLL_Y, axis=0)
        temp_raw16 = np.roll(temp_raw16, ROLL_Y, axis=0)
    if ROLL_X:
        yuyv = np.roll(yuyv, ROLL_X, axis=1)
        temp_raw16 = np.roll(temp_raw16, ROLL_X, axis=1)
    return yuyv, temp_raw16


@dataclass
class LatestFrame:
    yuyv: np.ndarray | None = None          # (H, W, 2) uint8
    temp_raw16: np.ndarray | None = None    # (H, W) uint16
    unit: str = TempUnit.RAW
    endian: str = "LE"
    stats: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    await load_debug_config()
    await save_debug_config_now()
    task = asyncio.create_task(capture_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
latest = LatestFrame()
clients: set[WebSocket] = set()
status: dict = {"ok": False, "message": "starting..."}


HTML = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width,initial-scale=1" />
    <title>Thermal Web Test</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, sans-serif; margin: 0; background:#0b1220; color:#e6edf3; }}
      .wrap {{ display:flex; gap:16px; padding:16px; align-items:flex-start; }}
      .card {{ background:#111a2e; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px; }}
      #canvas {{ width:{WIDTH*3}px; height:{HEIGHT*3}px; background:#000; border-radius:10px; cursor: crosshair; }}
      .mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace; }}
      .hint {{ opacity:.85; font-size: 13px; line-height: 1.4; }}
      .big {{ font-size: 28px; font-weight: 700; }}
      .row {{ display:flex; gap:8px; align-items:center; }}
      input[type="number"] {{ width: 110px; background:#0b1220; color:#e6edf3; border:1px solid rgba(255,255,255,.14); border-radius:8px; padding:6px 8px; }}
      button {{ background:#1f2a44; color:#e6edf3; border:1px solid rgba(255,255,255,.14); border-radius:8px; padding:6px 10px; cursor:pointer; }}
      button:hover {{ background:#243150; }}
      .sep {{ height:1px; background:rgba(255,255,255,.10); margin:12px 0; }}
    </style>
  </head>
  <body>
    <div class="wrap">
      <div class="card">
        <canvas id="canvas" width="{WIDTH}" height="{HEIGHT}"></canvas>
        <div class="hint" style="margin-top:10px">
          点击画面任意位置测温；右侧会显示该像素温度。<br/>
          如果温度数值明显不对，说明设备输出单位需要按协议修正（服务端有自动猜测）。
          <br/>可用 <span class="mono">W/A/S/D</span> 或方向键微调对齐（ROLL_X/ROLL_Y），用于消除左右拼接缝。
        </div>
      </div>
      <div class="card" style="min-width: 280px">
        <div class="hint mono">center (always)</div>
        <div id="center_xy" class="mono" style="margin-bottom:6px">{WIDTH//2},{HEIGHT//2}</div>
        <div class="hint mono">center temp (°C)</div>
        <div id="center_temp" class="big mono" style="margin-bottom:10px">-</div>
        <div class="hint mono">roll_x, roll_y</div>
        <div id="roll" class="mono" style="margin-bottom:10px">-</div>
        <div class="hint mono">gain, offset</div>
        <div id="cal" class="mono" style="margin-bottom:10px">-</div>
        <div class="row" style="margin-bottom:10px">
          <div class="mono">gain</div>
          <input id="gain" type="number" step="0.001" />
        </div>
        <div class="row" style="margin-bottom:10px">
          <div class="mono">offset</div>
          <input id="offset" type="number" step="0.1" />
        </div>
        <div class="row" style="margin-bottom:14px">
          <button id="apply">Apply</button>
          <button id="reset">Reset</button>
        </div>
        <div class="sep"></div>
        <div class="hint mono">2-point calibration</div>
        <div class="row" style="margin:8px 0">
          <div class="mono">center ℃</div>
          <input id="k_center" type="number" step="0.1" placeholder="known" />
        </div>
        <div class="row" style="margin:8px 0">
          <div class="mono">click ℃</div>
          <input id="k_click" type="number" step="0.1" placeholder="known" />
        </div>
        <div class="row" style="margin-bottom:14px">
          <button id="solve">Solve</button>
        </div>
        <div class="hint mono">click x,y</div>
        <div id="xy" class="mono" style="margin-bottom:10px">-</div>
        <div class="hint mono">click temp (°C)</div>
        <div id="temp" class="big mono">-</div>
        <div class="hint mono" style="margin-top:10px">unit guess</div>
        <div id="unit" class="mono">-</div>
        <div class="hint mono" style="margin-top:10px">endian</div>
        <div id="endian" class="mono">-</div>
        <div class="hint mono" style="margin-top:10px">raw / stats</div>
        <div id="raw" class="mono">-</div>
      </div>
    </div>
    <script>
      const canvas = document.getElementById("canvas");
      const ctx = canvas.getContext("2d");
      const xy = document.getElementById("xy");
      const temp = document.getElementById("temp");
      const center_temp = document.getElementById("center_temp");
      const roll = document.getElementById("roll");
      const cal = document.getElementById("cal");
      const gain = document.getElementById("gain");
      const offset = document.getElementById("offset");
      const applyBtn = document.getElementById("apply");
      const resetBtn = document.getElementById("reset");
      const kCenter = document.getElementById("k_center");
      const kClick = document.getElementById("k_click");
      const solveBtn = document.getElementById("solve");
      const unit = document.getElementById("unit");
      const endian = document.getElementById("endian");
      const raw = document.getElementById("raw");
      let lastClick = null; // {{x,y}}

      function drawCross(x, y, color) {{
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - 7, y); ctx.lineTo(x + 7, y);
        ctx.moveTo(x, y - 7); ctx.lineTo(x, y + 7);
        ctx.stroke();
        // 小圆点更容易对准
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(x, y, 1.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }}

      function canvasToPixel(evt) {{
        const r = canvas.getBoundingClientRect();
        const x = Math.floor((evt.clientX - r.left) * canvas.width / r.width);
        const y = Math.floor((evt.clientY - r.top)  * canvas.height / r.height);
        return {{x: Math.max(0, Math.min(canvas.width-1, x)), y: Math.max(0, Math.min(canvas.height-1, y))}};
      }}

      const ws = new WebSocket(`ws://${{location.host}}/ws`);
      ws.binaryType = "arraybuffer";

      async function loadConfig() {{
        try {{
          const r = await fetch("/config");
          if (!r.ok) return;
          const cfg = await r.json();
          const a = cfg.align || {{}};
          const c = cfg.calibration || {{}};
          const tp = c.two_point || {{}};
          if (a.roll_x !== undefined) roll.textContent = `${{a.roll_x}}, ${{a.roll_y}}`;
          if (c.gain !== undefined) gain.value = String(c.gain);
          if (c.offset !== undefined) offset.value = String(c.offset);
          if (tp.known_center_c !== undefined && tp.known_center_c !== null) kCenter.value = String(tp.known_center_c);
          if (tp.known_click_c !== undefined && tp.known_click_c !== null) kClick.value = String(tp.known_click_c);
        }} catch (e) {{}}
      }}
      loadConfig();

      ws.onmessage = (ev) => {{
        if (typeof ev.data === "string") {{
          const msg = JSON.parse(ev.data);
          if (msg.type === "center") {{
            center_temp.textContent = msg.temp_c.toFixed(2);
            unit.textContent = msg.unit;
            endian.textContent = msg.endian;
            roll.textContent = `${{msg.roll_x}}, ${{msg.roll_y}}`;
            cal.textContent = `${{msg.gain}}, ${{msg.offset}}`;
            if (gain.value === \"\") gain.value = msg.gain;
            if (offset.value === \"\") offset.value = msg.offset;
            raw.textContent = `raw=${{msg.raw}} | min=${{msg.min}} max=${{msg.max}} avg=${{msg.avg}} | p50=${{msg.p50}} p90=${{msg.p90}} p99=${{msg.p99}}`;
            return;
          }}
          if (msg.type === "temp") {{
            xy.textContent = `${{msg.x}},${{msg.y}}`;
            temp.textContent = msg.temp_c.toFixed(2);
            unit.textContent = msg.unit;
            endian.textContent = msg.endian;
            raw.textContent = `raw=${{msg.raw}} | min=${{msg.min}} max=${{msg.max}} avg=${{msg.avg}} | p50=${{msg.p50}} p90=${{msg.p90}} p99=${{msg.p99}}`;
            lastClick = {{x: msg.x, y: msg.y}};
          }}
          return;
        }}
        const blob = new Blob([ev.data], {{type:"image/jpeg"}});
        const url = URL.createObjectURL(blob);
        const img = new Image();
        img.onload = () => {{
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          // 叠加十字叉：中心点（青色）+ 点击点（黄绿色）
          drawCross({WIDTH//2}, {HEIGHT//2}, "#22d3ee");
          if (lastClick) drawCross(lastClick.x, lastClick.y, "#a3e635");
          URL.revokeObjectURL(url);
        }};
        img.src = url;
      }};

      canvas.addEventListener("click", (evt) => {{
        const p = canvasToPixel(evt);
        ws.send(JSON.stringify({{type:"click", x:p.x, y:p.y}}));
      }});

      function sendRoll(dx, dy) {{
        ws.send(JSON.stringify({{type:\"roll\", dx, dy}}));
      }}

      function sendCal(g, o) {{
        ws.send(JSON.stringify({{type:\"cal\", gain:g, offset:o}}));
      }}

      function sendSolve(kc, kk) {{
        ws.send(JSON.stringify({{type:\"solve\", known_center:kc, known_click:kk}}));
      }}

      applyBtn.addEventListener(\"click\", () => {{
        const g = parseFloat(gain.value);
        const o = parseFloat(offset.value);
        if (!Number.isFinite(g) || !Number.isFinite(o)) return;
        sendCal(g, o);
      }});

      resetBtn.addEventListener(\"click\", () => {{
        gain.value = \"1.0\";
        offset.value = \"0.0\";
        sendCal(1.0, 0.0);
      }});

      solveBtn.addEventListener(\"click\", () => {{
        const kc = parseFloat(kCenter.value);
        const kk = parseFloat(kClick.value);
        if (!Number.isFinite(kc) || !Number.isFinite(kk)) return;
        sendSolve(kc, kk);
      }});

      window.addEventListener(\"keydown\", (e) => {{
        if (e.key === \"w\" || e.key === \"ArrowUp\") sendRoll(0, -1);
        else if (e.key === \"s\" || e.key === \"ArrowDown\") sendRoll(0, 1);
        else if (e.key === \"a\" || e.key === \"ArrowLeft\") sendRoll(-1, 0);
        else if (e.key === \"d\" || e.key === \"ArrowRight\") sendRoll(1, 0);
      }});
    </script>
  </body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(HTML)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    global ROLL_X, ROLL_Y, TEMP_GAIN, TEMP_OFFSET, last_click, KNOWN_CENTER_C, KNOWN_CLICK_C
    try:
        while True:
            msg = await ws.receive_text()
            data = json.loads(msg)
            if data.get("type") == "cal":
                g = data.get("gain")
                o = data.get("offset")
                try:
                    TEMP_GAIN = float(g)
                    TEMP_OFFSET = float(o)
                except Exception:
                    # ignore invalid input
                    continue

                schedule_save_debug_config()

                # 立即回一条中心点信息，让页面更新 gain/offset 显示
                t = latest.temp_raw16
                if t is not None:
                    cx, cy = WIDTH // 2, HEIGHT // 2
                    v_c = int(t[cy, cx])
                    tc_c = raw_to_celsius(v_c, latest.unit)
                    st = latest.stats or {}
                    await ws.send_text(json.dumps({
                        "type": "center",
                        "x": cx,
                        "y": cy,
                        "temp_c": tc_c,
                        "unit": latest.unit,
                        "endian": latest.endian,
                        "raw": v_c,
                        "min": st.get("min"),
                        "max": st.get("max"),
                        "avg": st.get("avg"),
                        "p50": st.get("p50"),
                        "p90": st.get("p90"),
                        "p99": st.get("p99"),
                        "roll_x": ROLL_X,
                        "roll_y": ROLL_Y,
                        "gain": TEMP_GAIN,
                        "offset": TEMP_OFFSET,
                    }))
                continue
            if data.get("type") == "solve":
                # 两点标定：用中心点和点击点的“已知温度”，自动求 gain/offset
                t = latest.temp_raw16
                if t is None or last_click is None:
                    continue
                try:
                    known_center = float(data.get("known_center"))
                    known_click = float(data.get("known_click"))
                except Exception:
                    continue

                KNOWN_CENTER_C = known_center
                KNOWN_CLICK_C = known_click

                cx, cy = WIDTH // 2, HEIGHT // 2
                x, y = last_click

                v_center = int(t[cy, cx])
                v_click = int(t[y, x])

                # 用“未标定”的基础温度作为 x 轴，再解线性系数
                tc0 = raw_to_celsius_base(v_center, latest.unit)
                tk0 = raw_to_celsius_base(v_click, latest.unit)
                denom = (tc0 - tk0)
                if abs(denom) < 1e-6:
                    continue

                TEMP_GAIN = (known_center - known_click) / denom
                TEMP_OFFSET = known_center - TEMP_GAIN * tc0

                schedule_save_debug_config()

                # 回显中心点
                tc_c = apply_calibration(tc0)
                st = latest.stats or {}
                await ws.send_text(json.dumps({
                    "type": "center",
                    "x": cx,
                    "y": cy,
                    "temp_c": tc_c,
                    "unit": latest.unit,
                    "endian": latest.endian,
                    "raw": v_center,
                    "min": st.get("min"),
                    "max": st.get("max"),
                    "avg": st.get("avg"),
                    "p50": st.get("p50"),
                    "p90": st.get("p90"),
                    "p99": st.get("p99"),
                    "roll_x": ROLL_X,
                    "roll_y": ROLL_Y,
                    "gain": TEMP_GAIN,
                    "offset": TEMP_OFFSET,
                }))
                continue
            if data.get("type") == "roll":
                dx = int(data.get("dx", 0))
                dy = int(data.get("dy", 0))
                ROLL_X += dx
                ROLL_Y += dy

                schedule_save_debug_config()

                # 立即回一条中心点信息，让页面更新 roll 显示
                t = latest.temp_raw16
                if t is not None:
                    cx, cy = WIDTH // 2, HEIGHT // 2
                    v_c = int(t[cy, cx])
                    tc_c = raw_to_celsius(v_c, latest.unit)
                    st = latest.stats or {}
                    await ws.send_text(json.dumps({
                        "type": "center",
                        "x": cx,
                        "y": cy,
                        "temp_c": tc_c,
                        "unit": latest.unit,
                        "endian": latest.endian,
                        "raw": v_c,
                        "min": st.get("min"),
                        "max": st.get("max"),
                        "avg": st.get("avg"),
                        "p50": st.get("p50"),
                        "p90": st.get("p90"),
                        "p99": st.get("p99"),
                        "roll_x": ROLL_X,
                        "roll_y": ROLL_Y,
                        "gain": TEMP_GAIN,
                        "offset": TEMP_OFFSET,
                    }))
                continue
            if data.get("type") == "click":
                x = int(data["x"])
                y = int(data["y"])
                last_click = (x, y)

                schedule_save_debug_config()

                t = latest.temp_raw16
                if t is None:
                    await ws.send_text(json.dumps({
                        "type": "temp",
                        "x": x,
                        "y": y,
                        "temp_c": float("nan"),
                        "unit": latest.unit,
                        "endian": latest.endian,
                        "raw": None,
                        "min": None,
                        "max": None,
                        "avg": None,
                        "p50": None,
                        "p90": None,
                        "p99": None,
                    }))
                    continue
                v = int(t[y, x])
                tc = raw_to_celsius(v, latest.unit)
                st = latest.stats or {}
                await ws.send_text(json.dumps({
                    "type": "temp",
                    "x": x,
                    "y": y,
                    "temp_c": tc,
                    "unit": latest.unit,
                    "endian": latest.endian,
                    "raw": v,
                    "min": st.get("min"),
                    "max": st.get("max"),
                    "avg": st.get("avg"),
                    "p50": st.get("p50"),
                    "p90": st.get("p90"),
                    "p99": st.get("p99"),
                }))
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


async def broadcast_jpeg(jpeg_bytes: bytes):
    dead = []
    for ws in clients:
        try:
            await ws.send_bytes(jpeg_bytes)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


async def broadcast_text(payload: str):
    dead = []
    for ws in clients:
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        clients.discard(ws)


def start_uvc_subprocess() -> subprocess.Popen:
    if not os.path.exists("./uvc_demo"):
        raise RuntimeError("找不到 ./uvc_demo，请先在 uvc_ubuntu 目录执行 make")

    # web 模式：uvc_demo web -> stdout 输出 temp_raw + yuv
    # 注意：直接 sudo 可能会等待密码导致卡死，这里用 -n 让它失败得更明确
    if os.geteuid() == 0:
        cmd = ["./uvc_demo", "web"]
    else:
        cmd = ["sudo", "-n", "./uvc_demo", "web"]

    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


async def capture_loop():
    global status
    try:
        proc = start_uvc_subprocess()
    except Exception as e:
        status = {
            "ok": False,
            "message": f"启动 ./uvc_demo 失败: {e}. 建议：用 sudo 运行 uvicorn，或配置 sudo 免密/权限。",
        }
        print(status["message"], file=sys.stderr)
        return

    assert proc.stdout is not None
    assert proc.stderr is not None

    frame_bytes = LEN_BYTES * 2  # temp + yuv
    try:
        while True:
            # 关键修复：read 是阻塞IO，必须放线程里，否则会卡住 event loop（表现为 uvicorn 一直 "Waiting for application startup"）
            buf = await asyncio.to_thread(proc.stdout.read, frame_bytes)
            if not buf or len(buf) != frame_bytes:
                # 尝试读一下 stderr，看看是不是 sudo 或设备权限错误
                err = await asyncio.to_thread(proc.stderr.read1, 4096)
                if err:
                    status = {"ok": False, "message": err.decode(errors="ignore").strip()[:500]}
                await asyncio.sleep(0.05)
                continue

            temp_bytes = buf[:LEN_BYTES]
            yuv_bytes = buf[LEN_BYTES:]

            temp_raw16, unit, endian = select_endian_and_unit(temp_bytes)

            yuyv = np.frombuffer(yuv_bytes, dtype=np.uint8).reshape((HEIGHT, WIDTH, 2))
            yuyv, temp_raw16 = align_yuyv_and_temp(yuyv, temp_raw16)

            # 对齐后再更新单位/统计（避免拼接错位影响分位数判断）
            latest.unit = guess_unit(temp_raw16)
            latest.endian = endian
            latest.temp_raw16 = temp_raw16
            r = temp_raw16.ravel()
            latest.stats = {
                "min": int(r.min()),
                "max": int(r.max()),
                "avg": int(r.mean()),
                "p50": int(np.percentile(r, 50)),
                "p90": int(np.percentile(r, 90)),
                "p99": int(np.percentile(r, 99)),
            }

            latest.yuyv = yuyv

            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                status = {"ok": True, "message": "streaming"}
                cx, cy = WIDTH // 2, HEIGHT // 2
                v_c = int(temp_raw16[cy, cx])
                tc_c = raw_to_celsius(v_c, latest.unit)
                st = latest.stats or {}
                await broadcast_text(json.dumps({
                    "type": "center",
                    "x": cx,
                    "y": cy,
                    "temp_c": tc_c,
                    "unit": latest.unit,
                    "endian": latest.endian,
                    "raw": v_c,
                    "min": st.get("min"),
                    "max": st.get("max"),
                    "avg": st.get("avg"),
                    "p50": st.get("p50"),
                    "p90": st.get("p90"),
                    "p99": st.get("p99"),
                    "roll_x": ROLL_X,
                    "roll_y": ROLL_Y,
                    "gain": TEMP_GAIN,
                    "offset": TEMP_OFFSET,
                }))

                # 实时刷新“最后一次点击点”的温度（如果用户点过）
                if last_click is not None:
                    x, y = last_click
                    v = int(temp_raw16[y, x])
                    tc = raw_to_celsius(v, latest.unit)
                    await broadcast_text(json.dumps({
                        "type": "temp",
                        "x": x,
                        "y": y,
                        "temp_c": tc,
                        "unit": latest.unit,
                        "endian": latest.endian,
                        "raw": v,
                        "min": st.get("min"),
                        "max": st.get("max"),
                        "avg": st.get("avg"),
                        "p50": st.get("p50"),
                        "p90": st.get("p90"),
                        "p99": st.get("p99"),
                    }))
                await broadcast_jpeg(enc.tobytes())
            await asyncio.sleep(0)  # give event loop a chance
    finally:
        proc.terminate()


@app.get("/config")
def get_debug_config():
    return build_debug_config()


@app.get("/health")
def health():
    return {
        **status,
        "unit": latest.unit,
        "endian": latest.endian,
        "stats": latest.stats,
        "config_path": str(DEBUG_CONFIG_PATH),
    }

