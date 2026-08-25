"""Production-oriented Waveshare Pico-GPS-L76K / AT6558R GNSS support.

The module separates NMEA parsing from serial transport so RoboCar can use the
same parser for direct UART, replay, or NMEA forwarded by Pico firmware.

For the Pico-GPS-L76K carrier the preferred vehicle topology is normally:
    L76K -> Pico UART0 (GP0/GP1) -> Pico firmware -> Pi host sensor stream.
``L76KGNSS`` is therefore optional direct-serial transport; ``L76KNMEAParser``
is the reusable protocol layer.

Hardware contract used here (Waveshare L76K product/module documentation):
- AT6558R; GPS/BDS/GLONASS/QZSS;
- UART + NMEA 0183 / CASIC proprietary protocol;
- default 9600 baud;
- default 1 Hz, supported PCAS02 rates 1/2/5 Hz.

The driver never fabricates coordinates, never silently enables simulation, and
never changes receiver configuration at import/startup time.
"""
from __future__ import annotations

import math
import queue
import threading
import time

from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dt_time, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from ..utils.rc_errors import *
from ..utils.rc_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("L76K GNSS")
printer = PrettyPrinter()


KNOT_TO_MPS = 0.5144444444444445
KPH_TO_MPS = 1.0 / 3.6
DEFAULT_BAUD = 9600
SUPPORTED_BAUD_RATES = (4800, 9600, 19200, 38400, 57600, 115200)
SUPPORTED_UPDATE_RATES_HZ = (1, 2, 5)
UPDATE_RATE_INTERVAL_MS = {1: 1000, 2: 500, 5: 200}
NAV_TYPES = frozenset({"RMC", "GGA", "GNS", "GSA", "VTG", "GLL", "ZDA", "GSV"})
FIX_TYPES = frozenset({"RMC", "GGA", "GNS", "GLL"})


@dataclass(frozen=True, slots=True)
class NMEASentence:
    raw: str
    talker: str
    sentence_type: str
    fields: tuple[str, ...]
    checksum: Optional[int]


@dataclass(frozen=True, slots=True)
class GNSSFix:
    sequence: int
    receipt_time: float
    receipt_monotonic: float
    utc_datetime: Optional[datetime]
    latitude_deg: Optional[float]
    longitude_deg: Optional[float]
    altitude_m: Optional[float]
    geoid_separation_m: Optional[float]
    speed_mps: Optional[float]
    track_deg: Optional[float]
    fix_quality: Optional[int]
    fix_type: Optional[int]
    satellites_used: Optional[int]
    satellites_in_view: Optional[int]
    hdop: Optional[float]
    vdop: Optional[float]
    pdop: Optional[float]
    mode: Optional[str]
    valid: bool
    talker: Optional[str]
    last_sentence_type: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["utc_datetime"] = self.utc_datetime.isoformat() if self.utc_datetime else None
        return payload

    def age_seconds(self, *, now_monotonic: Optional[float] = None) -> float:
        now = time.monotonic() if now_monotonic is None else float(now_monotonic)
        return max(0.0, now - self.receipt_monotonic)

    def is_fresh(self, max_age_s: float) -> bool:
        limit = require_finite_float(max_age_s, "gnss.max_age_s", minimum=0.0)
        return self.age_seconds() <= limit


@dataclass(slots=True)
class _State:
    sequence: int = 0
    utc_date: Optional[date] = None
    utc_time: Optional[dt_time] = None
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    geoid_separation_m: Optional[float] = None
    speed_mps: Optional[float] = None
    track_deg: Optional[float] = None
    fix_quality: Optional[int] = None
    fix_type: Optional[int] = None
    satellites_used: Optional[int] = None
    satellites_in_view: Optional[int] = None
    hdop: Optional[float] = None
    vdop: Optional[float] = None
    pdop: Optional[float] = None
    mode: Optional[str] = None
    status_valid: Optional[bool] = None
    talker: Optional[str] = None
    last_sentence_type: Optional[str] = None


def nmea_checksum(payload: str) -> int:
    value = 0
    for char in str(payload):
        value ^= ord(char)
    return value


def build_nmea_command(payload: str) -> bytes:
    body = str(payload).strip().lstrip("$").split("*", 1)[0]
    if not body:
        raise GNSSConfigurationError("empty NMEA/PCAS command")
    return f"${body}*{nmea_checksum(body):02X}\r\n".encode("ascii")


def build_pcas_update_rate_command(rate_hz: int) -> bytes:
    rate = require_int(rate_hz, "gnss.update_rate_hz", minimum=1)
    interval = UPDATE_RATE_INTERVAL_MS.get(rate)
    if interval is None:
        raise GNSSConfigurationError(
            f"L76K update rate must be one of {SUPPORTED_UPDATE_RATES_HZ}; got {rate_hz!r}"
        )
    return build_nmea_command(f"PCAS02,{interval}")


def build_pcas_baud_command(baud: int) -> bytes:
    baud = require_int(baud, "gnss.baud", minimum=1)
    command = {4800: 0, 9600: 1, 19200: 2, 38400: 3, 57600: 4, 115200: 5}.get(baud)
    if command is None:
        raise GNSSConfigurationError(f"unsupported L76K baud: {baud!r}")
    return build_nmea_command(f"PCAS01,{command}")


def parse_nmea_sentence(sentence: str | bytes, *, require_checksum: bool = True,
                        max_length: int = 256) -> NMEASentence:
    limit = require_int(max_length, "gnss.max_sentence_length", minimum=16)
    try:
        text = sentence.decode("ascii", errors="strict").strip() if isinstance(sentence, bytes) else str(sentence or "").strip()
    except UnicodeDecodeError as exc:
        raise GNSSProtocolError("NMEA sentence is not ASCII") from exc
    if not text or not text.startswith("$"):
        raise GNSSProtocolError("NMEA sentence must be non-empty and start with '$'")
    if len(text) > limit:
        raise GNSSProtocolError(f"NMEA sentence exceeds {limit} characters")

    body_checksum = text[1:]
    supplied = None
    if "*" in body_checksum:
        body, checksum_text = body_checksum.rsplit("*", 1)
        if len(checksum_text) != 2:
            raise GNSSProtocolError("NMEA checksum must be two hex digits")
        try:
            supplied = int(checksum_text, 16)
        except ValueError as exc:
            raise GNSSProtocolError("invalid NMEA checksum") from exc
        calculated = nmea_checksum(body)
        if supplied != calculated:
            raise GNSSProtocolError(
                f"NMEA checksum mismatch received={supplied:02X} calculated={calculated:02X}"
            )
    else:
        body = body_checksum
        if require_checksum:
            raise GNSSProtocolError("NMEA checksum required")

    parts = body.split(",")
    header = parts[0].upper().strip()
    if len(header) < 5:
        raise GNSSProtocolError(f"invalid NMEA header {header!r}")
    sentence_type = header[-3:]
    talker = header[:-3]
    if sentence_type not in NAV_TYPES:
        raise GNSSProtocolError(f"unsupported navigation sentence {sentence_type!r}")
    if not talker or not talker.isalnum():
        raise GNSSProtocolError(f"invalid NMEA talker {talker!r}")
    return NMEASentence(text, talker, sentence_type, tuple(parts[1:]), supplied)


def _f(fields: Sequence[str], i: int) -> Optional[str]:
    if i >= len(fields):
        return None
    value = str(fields[i]).strip()
    return value or None


def _utc_time(raw: Optional[str]) -> Optional[dt_time]:
    if raw is None or len(raw) < 6:
        return None
    try:
        hour, minute = int(raw[:2]), int(raw[2:4])
        sec_f = float(raw[4:])
        second = int(sec_f)
        microsecond = min(999999, int(round((sec_f - second) * 1_000_000)))
        return dt_time(hour, minute, second, microsecond)
    except (TypeError, ValueError, OverflowError):
        return None


def _rmc_date(raw: Optional[str]) -> Optional[date]:
    if raw is None or len(raw) != 6 or not raw.isdigit():
        return None
    try:
        yy = int(raw[4:6])
        return date(1900 + yy if yy >= 80 else 2000 + yy, int(raw[2:4]), int(raw[:2]))
    except ValueError:
        return None


def _coord(raw: Optional[str], hemi: Optional[str], *, latitude: bool) -> Optional[float]:
    if raw is None or hemi is None:
        return None
    hemi = hemi.upper()
    if hemi not in ({"N", "S"} if latitude else {"E", "W"}):
        return None
    digits = 2 if latitude else 3
    try:
        deg, minutes = int(raw[:digits]), float(raw[digits:])
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(minutes) or not 0.0 <= minutes < 60.0:
        return None
    value = deg + minutes / 60.0
    if hemi in {"S", "W"}:
        value = -value
    limit = 90.0 if latitude else 180.0
    return value if -limit <= value <= limit else None


def _course(raw: Optional[str]) -> Optional[float]:
    value = optional_finite_float(raw)
    return None if value is None else value % 360.0


def _mode_valid(mode: Optional[str]) -> Optional[bool]:
    if not mode:
        return None
    return any(c not in {"N", "-"} for c in mode.upper())


class L76KNMEAParser:
    """Checksum-validating, stateful NMEA parser for L76K navigation output."""

    def __init__(self, *, require_checksum: bool = True, max_sentence_length: int = 256):
        self.require_checksum = bool(require_checksum)
        self.max_sentence_length = require_int(max_sentence_length, "gnss.max_sentence_length", minimum=16)
        self._state = _State()
        self._latest: Optional[GNSSFix] = None
        self._lock = threading.RLock()
        self.sentences_parsed = 0
        self.sentences_ignored = 0
        self.parse_errors = 0
        self.checksum_errors = 0

    def latest(self) -> Optional[GNSSFix]:
        with self._lock:
            return self._latest

    def feed(self, sentence: str | bytes, *, receipt_time: Optional[float] = None,
             receipt_monotonic: Optional[float] = None) -> Optional[GNSSFix]:
        wall = time.time() if receipt_time is None else float(receipt_time)
        mono = time.monotonic() if receipt_monotonic is None else float(receipt_monotonic)
        try:
            text = sentence.decode("ascii", errors="strict").strip() if isinstance(sentence, bytes) else str(sentence or "").strip()
        except UnicodeDecodeError as exc:
            self.parse_errors += 1
            raise GNSSProtocolError("NMEA sentence is not ASCII") from exc
        if not text:
            return None
        header = text[1:].split(",", 1)[0].split("*", 1)[0].upper() if text.startswith("$") else ""
        if header.startswith("PCAS"):
            self.sentences_ignored += 1
            return None
        try:
            msg = parse_nmea_sentence(text, require_checksum=self.require_checksum,
                                      max_length=self.max_sentence_length)
        except GNSSProtocolError as exc:
            self.parse_errors += 1
            if "checksum" in str(exc).lower():
                self.checksum_errors += 1
            raise
        with self._lock:
            getattr(self, f"_apply_{msg.sentence_type.lower()}")(msg.fields)
            self._state.sequence += 1
            self._state.talker = msg.talker
            self._state.last_sentence_type = msg.sentence_type
            self.sentences_parsed += 1
            self._latest = self._snapshot(wall, mono)
            return self._latest

    def _snapshot(self, wall: float, mono: float) -> GNSSFix:
        s = self._state
        utc = datetime.combine(s.utc_date, s.utc_time, tzinfo=timezone.utc) if s.utc_date and s.utc_time else None
        return GNSSFix(
            s.sequence, wall, mono, utc, s.latitude_deg, s.longitude_deg,
            s.altitude_m, s.geoid_separation_m, s.speed_mps, s.track_deg,
            s.fix_quality, s.fix_type, s.satellites_used, s.satellites_in_view,
            s.hdop, s.vdop, s.pdop, s.mode,
            bool(s.status_valid is True and s.latitude_deg is not None and s.longitude_deg is not None),
            s.talker, s.last_sentence_type,
        )

    def _position(self, lat, ns, lon, ew):
        lat_v, lon_v = _coord(lat, ns, latitude=True), _coord(lon, ew, latitude=False)
        if lat_v is not None and lon_v is not None:
            self._state.latitude_deg, self._state.longitude_deg = lat_v, lon_v

    def _apply_rmc(self, x):
        s = self._state
        s.utc_time = _utc_time(_f(x, 0)) or s.utc_time
        status = (_f(x, 1) or "").upper()
        if status in {"A", "V"}:
            s.status_valid = status == "A"
        self._position(_f(x, 2), _f(x, 3), _f(x, 4), _f(x, 5))
        knots = optional_finite_float(_f(x, 6), minimum=0.0)
        if knots is not None:
            s.speed_mps = knots * KNOT_TO_MPS
        track = _course(_f(x, 7))
        if track is not None:
            s.track_deg = track
        s.utc_date = _rmc_date(_f(x, 8)) or s.utc_date
        mode = _f(x, 11)
        if mode:
            s.mode = mode.upper()
            if _mode_valid(s.mode) is False:
                s.status_valid = False

    def _apply_gga(self, x):
        s = self._state
        s.utc_time = _utc_time(_f(x, 0)) or s.utc_time
        self._position(_f(x, 1), _f(x, 2), _f(x, 3), _f(x, 4))
        q = optional_int(_f(x, 5), minimum=0)
        if q is not None:
            s.fix_quality, s.status_valid = q, q > 0
        sats = optional_int(_f(x, 6), minimum=0)
        if sats is not None:
            s.satellites_used = sats
        for attr, value in (("hdop", optional_finite_float(_f(x, 7), minimum=0.0)),
                            ("altitude_m", optional_finite_float(_f(x, 8))),
                            ("geoid_separation_m", optional_finite_float(_f(x, 10)))):
            if value is not None:
                setattr(s, attr, value)

    def _apply_gns(self, x):
        s = self._state
        s.utc_time = _utc_time(_f(x, 0)) or s.utc_time
        self._position(_f(x, 1), _f(x, 2), _f(x, 3), _f(x, 4))
        mode = _f(x, 5)
        if mode:
            mode = mode.upper()
            s.mode = mode
            valid = _mode_valid(s.mode)
            if valid is not None:
                s.status_valid = valid
        sats = optional_int(_f(x, 6), minimum=0)
        if sats is not None:
            s.satellites_used = sats
        for attr, value in (("hdop", optional_finite_float(_f(x, 7), minimum=0.0)),
                            ("altitude_m", optional_finite_float(_f(x, 8))),
                            ("geoid_separation_m", optional_finite_float(_f(x, 9)))):
            if value is not None:
                setattr(s, attr, value)

    def _apply_gll(self, x):
        s = self._state
        self._position(_f(x, 0), _f(x, 1), _f(x, 2), _f(x, 3))
        s.utc_time = _utc_time(_f(x, 4)) or s.utc_time
        status = (_f(x, 5) or "").upper()
        if status in {"A", "V"}:
            s.status_valid = status == "A"
        mode = _f(x, 6)
        if mode:
            s.mode = mode.upper()

    def _apply_gsa(self, x):
        s = self._state
        fix_type = optional_int(_f(x, 1), minimum=1, maximum=3)
        if fix_type is not None:
            s.fix_type = fix_type
            if fix_type == 1:
                s.status_valid = False
        for attr, idx in (("pdop", 14), ("hdop", 15), ("vdop", 16)):
            value = optional_finite_float(_f(x, idx), minimum=0.0)
            if value is not None:
                setattr(s, attr, value)

    def _apply_vtg(self, x):
        s = self._state
        track = _course(_f(x, 0))
        if track is not None:
            s.track_deg = track
        kph = optional_finite_float(_f(x, 6), minimum=0.0)
        knots = optional_finite_float(_f(x, 4), minimum=0.0)
        if kph is not None:
            s.speed_mps = kph * KPH_TO_MPS
        elif knots is not None:
            s.speed_mps = knots * KNOT_TO_MPS
        mode = _f(x, 8)
        if mode is not None:
            s.mode = mode.upper()

    def _apply_zda(self, x):
        s = self._state
        s.utc_time = _utc_time(_f(x, 0)) or s.utc_time
        day, month, year = optional_int(_f(x, 1), minimum=1, maximum=31), optional_int(_f(x, 2), minimum=1, maximum=12), optional_int(_f(x, 3), minimum=1, maximum=9999)
        if day and month and year:
            try:
                s.utc_date = date(year, month, day)
            except ValueError:
                pass

    def _apply_gsv(self, x):
        value = optional_int(_f(x, 2), minimum=0)
        if value is not None:
            self._state.satellites_in_view = value


class L76KGNSS:
    """Threaded direct-serial CPython transport for L76K."""

    def __init__(self, port: str = "/dev/serial0", baud: int = DEFAULT_BAUD, *,
                 timeout_s: float = 0.5, queue_size: int = 32,
                 require_checksum: bool = True, auto_reconnect: bool = True,
                 reconnect_delay_s: float = 1.0,
                 serial_factory: Optional[Callable[..., Any]] = None):
        self.port = str(port or "").strip()
        if not self.port:
            raise GNSSConfigurationError("GNSS serial port cannot be empty")
        self.baud = require_int(baud, "gnss.baud", minimum=1)
        if self.baud not in SUPPORTED_BAUD_RATES:
            raise GNSSConfigurationError(f"unsupported L76K baud {self.baud}")
        self.timeout_s = require_finite_float(timeout_s, "gnss.timeout_s", minimum=0.01)
        self.reconnect_delay_s = require_finite_float(reconnect_delay_s, "gnss.reconnect_delay_s", minimum=0.0)
        self.auto_reconnect = bool(auto_reconnect)
        self.serial_factory = serial_factory
        self.parser = L76KNMEAParser(require_checksum=require_checksum)
        self._q = queue.Queue(maxsize=require_int(queue_size, "gnss.queue_size", minimum=1))
        self._callbacks: list[Callable[[GNSSFix], None]] = []
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._ser: Any = None
        self._status = "stopped"
        self._resolved_port: Optional[str] = None
        self._last_error: Optional[str] = None
        self._last_sentence_mono: Optional[float] = None
        self._last_fix_mono: Optional[float] = None
        self.lines_received = self.valid_sentences = self.protocol_errors = 0
        self.transport_errors = self.read_timeouts = self.callback_errors = 0
        self.dropped_fixes = self.reconnects = self.commands_sent = 0

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self._status = "starting"
        self._open()
        self._thread = threading.Thread(target=self._loop, name="RoboCarL76KGNSS", daemon=True)
        self._thread.start()
        self._status = "operational"

    def stop(self) -> None:
        self._stop.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=max(1.0, self.timeout_s * 3.0))
        self._thread = None
        self._close()
        self._status = "stopped"

    close = stop

    def latest(self) -> Optional[GNSSFix]:
        return self.parser.latest()

    def poll_nowait(self) -> Optional[GNSSFix]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def subscribe(self, callback: Callable[[GNSSFix], None]):
        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)
        return callback

    def unsubscribe(self, callback: Callable[[GNSSFix], None]) -> bool:
        with self._lock:
            try:
                self._callbacks.remove(callback)
                return True
            except ValueError:
                return False

    def wait_for_fix(self, timeout_s: float, *, max_age_s: Optional[float] = None) -> Optional[GNSSFix]:
        timeout = require_finite_float(timeout_s, "gnss.wait_for_fix.timeout_s", minimum=0.0)
        deadline = time.monotonic() + timeout
        while time.monotonic() <= deadline:
            fix = self.latest()
            if fix and fix.valid and (max_age_s is None or fix.is_fresh(max_age_s)):
                return fix
            self._stop.wait(min(0.05, max(0.0, deadline - time.monotonic())))
        return None

    def set_update_rate_hz(self, rate_hz: int) -> bytes:
        command = build_pcas_update_rate_command(rate_hz)
        self._write(command)
        return command

    def send_pcas(self, payload: str) -> bytes:
        command = build_nmea_command(payload)
        self._write(command)
        return command

    def health(self) -> dict[str, Any]:
        now = time.monotonic()
        latest = self.latest()
        age = lambda stamp: None if stamp is None else max(0.0, now - stamp)
        status = self._status
        if self.running and latest is not None and not latest.valid:
            status = "no_fix"
        if self.running and self.transport_errors:
            status = "degraded"
        return {
            "status": status, "running": self.running,
            "port_requested": self.port, "port_resolved": self._resolved_port,
            "baud": self.baud, "lines_received": self.lines_received,
            "valid_sentences": self.valid_sentences, "protocol_errors": self.protocol_errors,
            "checksum_errors": self.parser.checksum_errors,
            "transport_errors": self.transport_errors, "read_timeouts": self.read_timeouts,
            "callback_errors": self.callback_errors, "dropped_fixes": self.dropped_fixes,
            "reconnects": self.reconnects, "commands_sent": self.commands_sent,
            "last_sentence_age_s": age(self._last_sentence_mono),
            "last_fix_age_s": age(self._last_fix_mono),
            "latest_fix_valid": None if latest is None else latest.valid,
            "latest_satellites_used": None if latest is None else latest.satellites_used,
            "latest_hdop": None if latest is None else latest.hdop,
            "last_error": self._last_error,
        }

    def _serial_ctor(self):
        if self.serial_factory is not None:
            return self.serial_factory
        try:
            import serial  # type: ignore
        except Exception as exc:
            raise GNSSTransportError("pyserial is required for direct GNSS UART") from exc
        return serial.Serial

    def _resolve_port(self) -> str:
        if self.port.lower() != "auto":
            return self.port
        for candidate in ("/dev/serial0", "/dev/ttyAMA0", "/dev/ttyS0"):
            if Path(candidate).exists():
                return candidate
        try:
            from serial.tools import list_ports  # type: ignore
            ports = list(list_ports.comports())
        except Exception as exc:
            raise GNSSTransportError("GNSS auto-detection unavailable") from exc
        preferred = [str(p.device) for p in ports if any(t in f"{p.device} {p.description} {p.hwid}".lower() for t in ("gps", "gnss", "l76", "cp210"))]
        if len(preferred) == 1:
            return preferred[0]
        if len(ports) == 1:
            return str(ports[0].device)
        raise GNSSTransportError("GNSS auto-detection ambiguous; configure explicit port")

    def _open(self):
        resolved = self._resolve_port()
        try:
            ser = self._serial_ctor()(resolved, baudrate=self.baud, timeout=self.timeout_s,
                                      write_timeout=self.timeout_s)
            if callable(getattr(ser, "reset_input_buffer", None)):
                ser.reset_input_buffer()
        except Exception as exc:
            self._status = "fault"
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise GNSSTransportError(f"cannot open L76K serial port {resolved!r}") from exc
        with self._lock:
            self._ser, self._resolved_port = ser, resolved
        self._last_error = None

    def _close(self):
        with self._lock:
            ser, self._ser = self._ser, None
        if ser is not None:
            try:
                ser.close()
            except Exception as exc:
                logger.warning("L76K close failed: %s", exc)

    def _write(self, payload: bytes):
        with self._lock:
            ser = self._ser
            if ser is None:
                raise GNSSTransportError("GNSS serial transport is closed")
            try:
                written = ser.write(payload)
                if callable(getattr(ser, "flush", None)):
                    ser.flush()
            except Exception as exc:
                self.transport_errors += 1
                raise GNSSTransportError("GNSS command write failed") from exc
        if written is not None and int(written) != len(payload):
            raise GNSSTransportError(f"short GNSS serial write {written}/{len(payload)}")
        self.commands_sent += 1

    @staticmethod
    def _peek_type(text: str) -> Optional[str]:
        if not text.startswith("$"):
            return None
        header = text[1:].split(",", 1)[0].split("*", 1)[0].upper()
        sentence_type = header[-3:] if len(header) >= 5 else ""
        return sentence_type if sentence_type in NAV_TYPES else None

    def _publish(self, fix: GNSSFix):
        if bounded_queue_put(self._q, fix):
            self.dropped_fixes += 1
        with self._lock:
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            try:
                callback(fix)
            except Exception as exc:
                self.callback_errors += 1
                self._last_error = f"callback: {type(exc).__name__}: {exc}"

    def _loop(self):
        while not self._stop.is_set():
            try:
                with self._lock:
                    ser = self._ser
                if ser is None:
                    raise GNSSTransportError("GNSS serial handle missing")
                raw = ser.readline()
                if not raw:
                    self.read_timeouts += 1
                    continue
                self.lines_received += 1
                mono = time.monotonic()
                self._last_sentence_mono = mono
                text = raw.decode("ascii", errors="strict").strip()
                sentence_type = self._peek_type(text)
                try:
                    fix = self.parser.feed(text, receipt_time=time.time(), receipt_monotonic=mono)
                except GNSSProtocolError as exc:
                    self.protocol_errors += 1
                    self._last_error = f"protocol: {exc}"
                    continue
                if fix is None:
                    continue
                self.valid_sentences += 1
                if sentence_type in FIX_TYPES:
                    if fix.valid:
                        self._last_fix_mono = mono
                    self._publish(fix)
            except Exception as exc:
                self.transport_errors += 1
                self._last_error = f"transport: {type(exc).__name__}: {exc}"
                if not self.auto_reconnect or self._stop.is_set():
                    self._status = "fault"
                    break
                self._status = "degraded"
                self._close()
                if self._stop.wait(self.reconnect_delay_s):
                    break
                try:
                    self._open()
                    self.reconnects += 1
                    self._status = "operational"
                except GNSSTransportError:
                    continue


__all__ = [
    "DEFAULT_BAUD","SUPPORTED_BAUD_RATES", "SUPPORTED_UPDATE_RATES_HZ",
    "GNSSFix", "NMEASentence", "L76KNMEAParser", "L76KGNSS",
    "nmea_checksum", "parse_nmea_sentence",
    "build_nmea_command", "build_pcas_update_rate_command",
    "build_pcas_baud_command",
]
