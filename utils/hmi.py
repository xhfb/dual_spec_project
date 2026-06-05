#!/usr/bin/env python3
"""
TJC3224 HMI serial test — bidirectional I/O for main page.

Screen -> RDK: parse 0x65 touch key events (发送键值).
RDK -> screen: push t0-t5 status texts and g0 scroll log.

Hardware (RDK X5):
  Port: /dev/ttyS6, 9600, 8N1
  Wiring: RDK TX->Pin16, RX->Pin36; cross-connect with screen.

Widget id mapping (main page, as-built):
  t0-t5, g0     RDK -> screen (txt attribute)
  b0 AUTO   -> id 1
  b1 MANUAL -> id 2
  b2 UP     -> id 3
  b3 RIGHT  -> id 4
  b4 LEFT   -> id 5
  b5 DOWN   -> id 6
  b6 ZERO   -> id 7
  b7 STOP   -> id 8   (recommended; use 置顶 in USART HMI)

Frame format (TJC docs):
  0x65 + page_id + component_id + event + 0xFF 0xFF 0xFF
  event: 0x01 = press, 0x00 = release

Usage:
  pip install pyserial
  python tjc_hmi_test.py --port COM26
  python tjc_hmi_test.py --port COM26 --demo-feedback
  python tjc_hmi_test.py --port COM26 --demo-status
  python tjc_hmi_test.py --port COM26 --push-snapshot
  python tjc_hmi_test.py --port COM26 --send 'main.t1.txt="Track"'
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, replace
from enum import IntEnum

try:
    import serial
except ImportError:
    print("Install pyserial: pip install pyserial", file=sys.stderr)
    raise

# TJC color constants (16-bit RGB565)
GRAY = 33840
GREEN = 2016
RED = 63488
WHITE = 65535
DARK = 10508

TJC_TAIL = b"\xff\xff\xff"
DEFAULT_PAGE = "main"

WIDGET_ID_TO_NAME: dict[int, str] = {
    0: "b7 STOP (page id conflict)",
    1: "b0 AUTO",
    2: "b1 MANUAL",
    3: "b2 UP",
    4: "b3 RIGHT",
    5: "b4 LEFT",
    6: "b5 DOWN",
    7: "b6 ZERO",
    8: "b7 STOP",
    10: "b7 STOP (legacy id 10)",
}

JOG_IDS = {3, 4, 5, 6}
STOP_IDS = {0, 8, 10}


@dataclass
class SystemSnapshot:
    """RDK status bundle mapped to t0-t5 and g0 (see hmi-design.md §7.3)."""

    title: str = "Dual-Spec"
    state: str = "Ready"
    fps: float | None = None
    max_temp_c: float | None = None
    pan_deg: float | None = None
    tilt_deg: float | None = None
    log_line: str | None = None


DEFAULT_SNAPSHOT = SystemSnapshot(
    title="Dual-Spec",
    state="Ready",
    fps=18.0,
    max_temp_c=36.8,
    pan_deg=12.3,
    tilt_deg=-5.1,
    log_line="HMI test ready",
)


class TjcHmiDriver:
    """RDK -> screen driver for t0-t5 and g0 on page `main`."""

    def __init__(self, ser: serial.Serial, page: str = DEFAULT_PAGE) -> None:
        self._ser = ser
        self.page = page
        self._last: SystemSnapshot | None = None

    @staticmethod
    def _esc(text: str) -> str:
        """Keep ASCII safe for TJC txt="..." assignments."""
        return text.replace("\\", "").replace('"', "'")

    def send_cmd(self, cmd: str) -> None:
        self._ser.write(cmd.encode("ascii") + TJC_TAIL)
        self._ser.flush()

    def init_screen(self) -> None:
        self.send_cmd("bkcmd=0")

    def set_log(self, text: str) -> None:
        self.send_cmd(f'{self.page}.g0.txt="{self._esc(text)}"')

    @staticmethod
    def fmt_fps(fps: float) -> str:
        return f"FPS:{fps:.0f}"

    @staticmethod
    def fmt_temp(temp_c: float) -> str:
        return f"Temp:{temp_c:.1f}C"

    @staticmethod
    def fmt_pan(deg: float) -> str:
        return f"Pan:{deg:.1f}"

    @staticmethod
    def fmt_tilt(deg: float) -> str:
        return f"Tilt:{deg:.1f}"

    def push_snapshot(self, snap: SystemSnapshot, *, force: bool = False) -> None:
        """Push changed fields only (or all when force=True)."""
        prev = self._last
        p = self.page

        if force or prev is None or snap.title != prev.title:
            self.send_cmd(f'{p}.t0.txt="{self._esc(snap.title)}"')
        if force or prev is None or snap.state != prev.state:
            self.send_cmd(f'{p}.t1.txt="{self._esc(snap.state)}"')
        if snap.fps is not None and (force or prev is None or snap.fps != prev.fps):
            self.send_cmd(f'{p}.t2.txt="{self.fmt_fps(snap.fps)}"')
        if snap.max_temp_c is not None and (
            force or prev is None or snap.max_temp_c != prev.max_temp_c
        ):
            self.send_cmd(f'{p}.t3.txt="{self.fmt_temp(snap.max_temp_c)}"')
        if snap.pan_deg is not None and (
            force or prev is None or snap.pan_deg != prev.pan_deg
        ):
            self.send_cmd(f'{p}.t4.txt="{self.fmt_pan(snap.pan_deg)}"')
        if snap.tilt_deg is not None and (
            force or prev is None or snap.tilt_deg != prev.tilt_deg
        ):
            self.send_cmd(f'{p}.t5.txt="{self.fmt_tilt(snap.tilt_deg)}"')
        if snap.log_line is not None and (
            force or prev is None or snap.log_line != prev.log_line
        ):
            self.set_log(snap.log_line)

        self._last = snap


class TouchEvent(IntEnum):
    RELEASE = 0x00
    PRESS = 0x01


@dataclass
class KeyEvent:
    page_id: int
    component_id: int
    event: TouchEvent
    raw: bytes

    @property
    def name(self) -> str:
        return WIDGET_ID_TO_NAME.get(self.component_id, f"id={self.component_id}")

    @property
    def is_press(self) -> bool:
        return self.event == TouchEvent.PRESS

    @property
    def is_release(self) -> bool:
        return self.event == TouchEvent.RELEASE


class TjcProtocol:
    HEADER = 0x65
    TAIL = TJC_TAIL

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[KeyEvent]:
        self._buf.extend(data)
        events: list[KeyEvent] = []
        while True:
            idx = self._buf.find(self.TAIL)
            if idx < 0:
                break
            frame = bytes(self._buf[: idx + 3])
            del self._buf[: idx + 3]
            ev = self._parse_frame(frame)
            if ev is not None:
                events.append(ev)
        return events

    def _parse_frame(self, frame: bytes) -> KeyEvent | None:
        if len(frame) < 7 or frame[0] != self.HEADER:
            return None
        page_id = frame[1]
        comp_id = frame[2]
        try:
            touch = TouchEvent(frame[3])
        except ValueError:
            return None
        return KeyEvent(page_id, comp_id, touch, frame)


def send_tjc(ser: serial.Serial, cmd: str) -> None:
    """Send one TJC instruction with 3-byte terminator."""
    TjcHmiDriver(ser).send_cmd(cmd)


def demo_feedback(hmi: TjcHmiDriver, ev: KeyEvent) -> None:
    """Optional visual feedback: RDK drives button colors + t1/g0."""
    cid = ev.component_id

    if cid in (1, 2) and ev.is_release:
        if cid == 1:
            hmi.send_cmd(f"{hmi.page}.b0.bco={GREEN}")
            hmi.send_cmd(f"{hmi.page}.b1.bco={GRAY}")
            hmi.set_log("Mode: AUTO")
            hmi.push_snapshot(replace(hmi._last or DEFAULT_SNAPSHOT, state="Track"))
        else:
            hmi.send_cmd(f"{hmi.page}.b0.bco={GRAY}")
            hmi.send_cmd(f"{hmi.page}.b1.bco={GREEN}")
            hmi.set_log("Mode: MANUAL")
            hmi.push_snapshot(replace(hmi._last or DEFAULT_SNAPSHOT, state="Manual"))
        return

    if cid in JOG_IDS:
        btn = {3: "b2", 4: "b3", 5: "b4", 6: "b5"}[cid]
        color = GREEN if ev.is_press else DARK
        hmi.send_cmd(f"{hmi.page}.{btn}.bco={color}")
        if ev.is_release:
            hmi.set_log("Jog stop")
        else:
            hmi.set_log(f"Jog {ev.name}")
        return

    if cid == 7 and ev.is_release:
        hmi.set_log("Homing...")
        return

    if cid in STOP_IDS:
        if ev.is_press:
            hmi.send_cmd(f"{hmi.page}.b7.bco={RED}")
            hmi.push_snapshot(
                replace(hmi._last or DEFAULT_SNAPSHOT, state="E-STOP", log_line="E-STOP ACTIVE")
            )
        elif ev.is_release:
            hmi.set_log("E-STOP released")


def describe_action(ev: KeyEvent) -> str:
    action = "PRESS" if ev.is_press else "RELEASE"
    extra = ""
    if ev.component_id in (1, 2) and ev.is_release:
        extra = " -> would set AUTO/MANUAL"
    elif ev.component_id in JOG_IDS:
        extra = " -> jog start" if ev.is_press else " -> jog stop"
    elif ev.component_id == 7 and ev.is_release:
        extra = " -> homing"
    elif ev.component_id in STOP_IDS and ev.is_release:
        extra = " -> estop"
    elif ev.component_id in STOP_IDS and ev.is_press:
        extra = " -> estop (press)"
    return f"[{action}] page={ev.page_id} {ev.name}{extra}"


def _demo_status_tick(hmi: TjcHmiDriver, tick: int) -> None:
    base = hmi._last or DEFAULT_SNAPSHOT
    hmi.push_snapshot(
        replace(
            base,
            fps=15.0 + (tick % 10),
            max_temp_c=36.0 + (tick % 5) * 0.2,
            pan_deg=((tick * 2) % 360) - 180.0,
            tilt_deg=((tick * 3) % 90) - 45.0,
        )
    )


def run_listen(
    port: str,
    baud: int,
    demo_feedback_flag: bool,
    demo_status_flag: bool,
    init_snapshot: bool,
    once: bool,
    raw_hex: bool,
) -> None:
    parser = TjcProtocol()
    print(f"Opening {port} @ {baud} ...")
    print("RDK->screen: t0 title, t1 state, t2 FPS, t3 Temp, t4 Pan, t5 Tilt, g0 log")
    print("Screen->RDK: b0=1 .. b6=7, b7 STOP=8")
    print("Waiting for 0x65 key events (Ctrl+C to quit).\n")

    with serial.Serial(port, baud, timeout=0.05) as ser:
        hmi = TjcHmiDriver(ser)
        hmi.init_screen()
        if init_snapshot:
            hmi.push_snapshot(DEFAULT_SNAPSHOT, force=True)
        time.sleep(0.1)
        ser.reset_input_buffer()

        status_tick = 0
        last_status_ts = time.monotonic()

        try:
            while True:
                now = time.monotonic()
                if demo_status_flag and now - last_status_ts >= 0.5:
                    last_status_ts = now
                    status_tick += 1
                    _demo_status_tick(hmi, status_tick)

                chunk = ser.read(256)
                if not chunk:
                    continue
                if raw_hex:
                    print(f"RX raw: {chunk.hex(' ')}")
                events = parser.feed(chunk)
                if raw_hex and chunk and not events:
                    if parser._buf:
                        print(f"  buffer pending: {bytes(parser._buf).hex(' ')}")
                for ev in events:
                    line = describe_action(ev)
                    print(f"{time.strftime('%H:%M:%S')} {line}  raw={ev.raw.hex(' ')}")
                    if demo_feedback_flag:
                        demo_feedback(hmi, ev)
                    if once:
                        return
        except KeyboardInterrupt:
            print("\nStopped.")


def run_push_snapshot(port: str, baud: int) -> None:
    with serial.Serial(port, baud, timeout=1) as ser:
        hmi = TjcHmiDriver(ser)
        hmi.init_screen()
        hmi.push_snapshot(DEFAULT_SNAPSHOT, force=True)
        print("Pushed default snapshot to t0-t5 and g0:")
        print(f"  t0={DEFAULT_SNAPSHOT.title!r}")
        print(f"  t1={DEFAULT_SNAPSHOT.state!r}")
        print(f"  t2={TjcHmiDriver.fmt_fps(DEFAULT_SNAPSHOT.fps)}")
        print(f"  t3={TjcHmiDriver.fmt_temp(DEFAULT_SNAPSHOT.max_temp_c)}")
        print(f"  t4={TjcHmiDriver.fmt_pan(DEFAULT_SNAPSHOT.pan_deg)}")
        print(f"  t5={TjcHmiDriver.fmt_tilt(DEFAULT_SNAPSHOT.tilt_deg)}")
        print(f"  g0={DEFAULT_SNAPSHOT.log_line!r}")


def run_send(port: str, baud: int, cmd: str) -> None:
    with serial.Serial(port, baud, timeout=1) as ser:
        send_tjc(ser, cmd)
        print(f"Sent: {cmd}")
        time.sleep(0.2)
        resp = ser.read(256)
        if resp:
            print(f"RX ({len(resp)} bytes): {resp.hex(' ')}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="TJC3224 HMI bidirectional test (t0-t5/g0 + 0x65 keys)")
    p.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port (default: /dev/ttyUSB0 on RDK)",
    )
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument(
        "--demo-feedback",
        action="store_true",
        help="On key events: update bco, t1, g0 via TjcHmiDriver",
    )
    p.add_argument(
        "--demo-status",
        action="store_true",
        help="While listening: refresh t2-t5 every 0.5s with fake values",
    )
    p.add_argument(
        "--no-init",
        action="store_true",
        help="Skip initial t0-t5/g0 push on listen start",
    )
    p.add_argument(
        "--push-snapshot",
        action="store_true",
        help="Push demo t0-t5/g0 once and exit",
    )
    p.add_argument(
        "--send",
        metavar="CMD",
        help='Send one TJC command, e.g. main.t1.txt="READY"',
    )
    p.add_argument(
        "--once",
        action="store_true",
        help="Exit after first parsed key event",
    )
    p.add_argument(
        "--raw",
        action="store_true",
        help="Print all RX bytes (debug unparsed frames)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.send:
        run_send(args.port, args.baud, args.send)
    elif args.push_snapshot:
        run_push_snapshot(args.port, args.baud)
    else:
        run_listen(
            args.port,
            args.baud,
            args.demo_feedback,
            args.demo_status,
            not args.no_init,
            args.once,
            args.raw,
        )


if __name__ == "__main__":
    main()
