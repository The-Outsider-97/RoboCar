"""Shared helper primitives for the RoboCar subsystem.

The helpers in this module deliberately stay dependency-light.  They centralize
small deterministic operations that are reused by the serial sensor gateway,
wheel-speed estimator, motion controller, and RoboCar integration layer:

* finite numeric coercion and configuration validation;
* normalized command validation and clamping;
* low-pass filtering;
* PCA9685 pulse-width conversion for both 12-bit and 16-bit APIs;
* JSON/``KEY:VALUE`` serial-line decoding;
* bounded queue insertion without producer deadlock;
* lightweight sensor-value sanitation and freshness checks.

Hardware ownership, configuration loading, logging, threading policy, and SLAI
agent orchestration intentionally remain outside this module.
"""

from __future__ import annotations

import json
import math
import queue
import time

from collections.abc import Mapping
from numbers import Real
from typing import Any, Optional, TypeVar

from .rc_errors import *

_T = TypeVar("_T")


def _to_float(value: Any) -> Optional[float]:
    """Best-effort float conversion retained for backward compatibility.

    Unlike :func:`optional_finite_float`, this helper does not reject ``NaN`` or
    infinity; callers that consume physical measurements should normally use the
    finite variant below.
    """

    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _to_int(value: Any) -> Optional[int]:
    """Best-effort integer conversion retained for backward compatibility."""

    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def is_finite_number(value: Any) -> bool:
    """Return ``True`` for finite real scalars, excluding booleans."""

    if isinstance(value, bool):
        return False
    if isinstance(value, Real):
        return math.isfinite(float(value))
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def optional_finite_float(
    value: Any,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> Optional[float]:
    """Return a finite bounded float, or ``None`` for absent/invalid input."""

    converted = _to_float(value)
    if converted is None or not math.isfinite(converted):
        return None
    if minimum is not None and converted < float(minimum):
        return None
    if maximum is not None and converted > float(maximum):
        return None
    return converted


def optional_int(
    value: Any,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> Optional[int]:
    """Return a bounded integer, or ``None`` for absent/invalid input."""

    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, int):
            converted = value
        elif isinstance(value, Real):
            numeric = float(value)
            if not math.isfinite(numeric) or not numeric.is_integer():
                return None
            converted = int(numeric)
        elif isinstance(value, str):
            converted = int(value.strip(), 10)
        else:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    if minimum is not None and converted < int(minimum):
        return None
    if maximum is not None and converted > int(maximum):
        return None
    return converted


def optional_binary(value: Any) -> Optional[int]:
    """Return a normalized digital level (``0``/``1``), else ``None``."""

    converted = optional_int(value, minimum=0, maximum=1)
    return converted if converted in (0, 1) else None


def require_finite_float(
    value: Any,
    parameter: str,
    *,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
) -> float:
    """Return a validated configuration float or raise ``ConfigurationError``."""

    converted = optional_finite_float(value, minimum=minimum, maximum=maximum)
    if converted is None:
        raise ConfigurationError(
            parameter=parameter,
            value=value,
            valid_range=(minimum, maximum),
        )
    return converted


def require_int(
    value: Any,
    parameter: str,
    *,
    minimum: Optional[int] = None,
    maximum: Optional[int] = None,
) -> int:
    """Return a validated configuration integer or raise ``ConfigurationError``."""

    converted = optional_int(value, minimum=minimum, maximum=maximum)
    if converted is None:
        raise ConfigurationError(
            parameter=parameter,
            value=value,
            valid_range=(minimum, maximum),
        )
    return converted


def require_probability(value: Any, parameter: str) -> float:
    """Return a finite probability/low-pass coefficient in ``[0, 1]``."""

    return require_finite_float(value, parameter, minimum=0.0, maximum=1.0)


def clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a finite value to an inclusive interval.

    Invalid or inverted intervals fail explicitly rather than producing a
    plausible but incorrect control value.
    """

    candidate = optional_finite_float(value)
    lower = optional_finite_float(minimum)
    upper = optional_finite_float(maximum)
    if candidate is None or lower is None or upper is None:
        raise ValueError("clamp requires finite numeric values")
    if lower > upper:
        raise ValueError("minimum cannot be greater than maximum")
    return max(lower, min(upper, candidate))


def normalize_signed_command(value: Any, name: str) -> float:
    """Validate a normalized actuator command in ``[-1, 1]``.

    A ``ControlError`` is used here because invalid commands must never be
    silently clamped at the physical hardware boundary.  Upstream policy may
    clamp intentionally before calling this helper.
    """

    converted = optional_finite_float(value)
    if converted is None or converted < -1.0 or converted > 1.0:
        raise ControlError(
            controller=name,
            command=value,
            actual="expected finite normalized command in [-1.0, 1.0]",
        )
    return converted


def low_pass_filter(previous: Any, current: Any, alpha: Any) -> float:
    """Apply a first-order exponential low-pass filter.

    ``alpha=1`` returns the current sample and ``alpha=0`` retains the previous
    value.  All inputs must be finite.
    """

    previous_value = optional_finite_float(previous)
    current_value = optional_finite_float(current)
    alpha_value = optional_finite_float(alpha, minimum=0.0, maximum=1.0)
    if previous_value is None or current_value is None or alpha_value is None:
        raise ValueError("low_pass_filter requires finite values and alpha in [0, 1]")
    return alpha_value * current_value + (1.0 - alpha_value) * previous_value


def pulse_us_to_pca9685_counts(pulse_us: Any, frequency_hz: Any) -> int:
    """Convert a pulse width to legacy PCA9685 12-bit ``0..4095`` counts."""

    pulse = optional_finite_float(pulse_us, minimum=0.0)
    frequency = optional_finite_float(frequency_hz, minimum=1e-9)
    if pulse is None or frequency is None:
        raise ValueError("pulse_us and frequency_hz must be finite positive values")
    period_us = 1_000_000.0 / frequency
    fraction = clamp(pulse / period_us, 0.0, 1.0)
    return int(round(fraction * 4095.0))


def pulse_us_to_duty_cycle_16(pulse_us: Any, frequency_hz: Any) -> int:
    """Convert a pulse width to CircuitPython PCA9685 16-bit duty cycle."""

    pulse = optional_finite_float(pulse_us, minimum=0.0)
    frequency = optional_finite_float(frequency_hz, minimum=1e-9)
    if pulse is None or frequency is None:
        raise ValueError("pulse_us and frequency_hz must be finite positive values")
    period_us = 1_000_000.0 / frequency
    fraction = clamp(pulse / period_us, 0.0, 1.0)
    return int(round(fraction * 65535.0))


def decode_serial_payload(line: Any) -> Optional[dict[str, Any]]:
    """Decode one Pico serial line into a mapping.

    Accepted wire forms are deliberately narrow:

    * a JSON object, e.g. ``{"ultra_front_m": 0.42}``;
    * one or more comma-separated ``KEY:VALUE`` pairs, e.g. ``ULTRA_FRONT:0.42,HALL:1``.

    Empty lines, JSON arrays/scalars, and malformed key/value lines return
    ``None`` instead of being reinterpreted as another protocol.
    """

    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8", errors="strict").strip()
        except UnicodeDecodeError:
            return None
    else:
        text = str(line or "").strip()
    if not text:
        return None

    if text.startswith("{"):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        return dict(payload) if isinstance(payload, Mapping) else None

    if ":" not in text:
        return None

    payload: dict[str, Any] = {}
    for part in text.split(","):
        segment = part.strip()
        if ":" not in segment:
            continue
        key, raw_value = segment.split(":", 1)
        normalized_key = key.strip()
        normalized_value = raw_value.strip()
        if normalized_key and normalized_value:
            payload[normalized_key] = normalized_value
    return payload or None


def get_case_insensitive(mapping: Mapping[str, Any], *names: str) -> Any:
    """Return the first case-insensitive key match from ``mapping``."""

    lookup = {str(key).casefold(): value for key, value in mapping.items()}
    for name in names:
        key = str(name).casefold()
        if key in lookup:
            return lookup[key]
    return None


def bounded_queue_put(target: "queue.Queue[_T]", item: _T) -> bool:
    """Insert into a bounded queue without blocking the producer.

    If the queue is full, the oldest item is discarded and the new item is
    inserted.  The return value is ``True`` when an old item had to be dropped.
    """

    dropped = False
    try:
        target.put_nowait(item)
        return dropped
    except queue.Full:
        dropped = True

    try:
        target.get_nowait()
    except queue.Empty:
        pass

    try:
        target.put_nowait(item)
    except queue.Full:
        # Another producer raced us.  Do not block a sensor-reader thread.
        return True
    return dropped


def age_seconds(timestamp: Any, *, now: Optional[float] = None) -> Optional[float]:
    """Return non-negative age in seconds for a wall-clock timestamp."""

    ts = optional_finite_float(timestamp)
    current = time.time() if now is None else optional_finite_float(now)
    if ts is None or current is None:
        return None
    return max(0.0, current - ts)


def is_fresh(timestamp: Any, max_age_s: Any, *, now: Optional[float] = None) -> bool:
    """Return whether a timestamp is no older than ``max_age_s``."""

    limit = optional_finite_float(max_age_s, minimum=0.0)
    age = age_seconds(timestamp, now=now)
    return bool(limit is not None and age is not None and age <= limit)


__all__ = [
    "_to_float",
    "_to_int",
    "is_finite_number",
    "optional_finite_float",
    "optional_int",
    "optional_binary",
    "require_finite_float",
    "require_int",
    "require_probability",
    "clamp",
    "normalize_signed_command",
    "low_pass_filter",
    "pulse_us_to_pca9685_counts",
    "pulse_us_to_duty_cycle_16",
    "decode_serial_payload",
    "get_case_insensitive",
    "bounded_queue_put",
    "age_seconds",
    "is_fresh",
]
