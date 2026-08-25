"""Wheel-speed estimation from a monotonically increasing tick counter."""

from __future__ import annotations

import math
import time

from typing import Any, Callable, Dict, Optional

from .utils.config_loader import load_global_config, get_config_section
from .utils.rc_errors import *
from .utils.rc_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("RC Wheel Encoder")
printer = PrettyPrinter()

class WheelEncoder:
    """Estimate longitudinal wheel speed from total encoder ticks.

    ``gear_ratio`` is interpreted as input-shaft revolutions per wheel
    revolution.  If the encoder counts the wheel directly, configure ``1.0``.
    A counter decrease is treated as a reset/re-baseline event; no rollover
    modulus is guessed because the repository does not currently define one.
    """

    MAX_SPEED_MPS = 5.5555555556  # safeguard (20 km/h)

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = dict(config) if config is not None else load_global_config()
        self.wheel_config = get_config_section("encoder", self.config)
        self._clock = clock

        self.ppr = require_int(self.wheel_config.get("ppr"), "encoder.ppr", minimum=1)
        self.wheel_d = require_finite_float(
            self.wheel_config.get("wheel_d"),
            "encoder.wheel_d",
            minimum=1e-9,
        )
        self.gear_ratio = require_finite_float(
            self.wheel_config.get("gear_ratio"),
            "encoder.gear_ratio",
            minimum=1e-9,
        )
        self.alpha = require_probability(self.wheel_config.get("alpha"), "encoder.alpha")
        self.max_speed_mps = require_finite_float(
            self.wheel_config.get("max_speed_mps", self.MAX_SPEED_MPS),
            "encoder.max_speed_mps",
            minimum=1e-9,
        )

        self.wheel_c = math.pi * self.wheel_d
        self._last_ticks: Optional[int] = None
        self._last_t: Optional[float] = None
        self._speed = 0.0
        self._status = "initializing"
        self._last_error: Optional[str] = None
        self._samples = 0
        self._counter_resets = 0
        self._rejected_samples = 0
        logger.info(
            "WheelEncoder initialized: ppr=%s wheel_d=%.6fm gear_ratio=%.6f alpha=%.3f",
            self.ppr,
            self.wheel_d,
            self.gear_ratio,
            self.alpha,
        )

    def _get_validated_param(self, name: str, *,
        min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
        """Backward-compatible validated access to one encoder config value."""

        return require_finite_float(
            self.wheel_config.get(name),
            f"encoder.{name}",
            minimum=min_val,
            maximum=max_val,
        )

    def reset(self, *, ticks_total: Optional[int] = None) -> None:
        """Reset estimator state; optionally establish a new counter baseline."""

        if ticks_total is not None:
            validated = optional_int(ticks_total, minimum=0)
            if validated is None:
                raise ConfigurationError("encoder.ticks_total", ticks_total, (0, None))
            self._last_ticks = validated
            self._last_t = self._clock()
        else:
            self._last_ticks = None
            self._last_t = None
        self._speed = 0.0
        self._status = "reset"
        self._last_error = None

    def update_from_ticks(self, ticks_total: int, *,
        timestamp: Optional[float] = None) -> float:
        """Update the filtered speed estimate from an accumulated tick count.

        Invalid samples do not masquerade as ``0 m/s``.  The last valid estimate
        is retained and the health state becomes degraded, allowing the caller to
        decide whether stale velocity is acceptable for its control policy.
        """

        ticks = optional_int(ticks_total, minimum=0)
        if ticks is None:
            self._reject(SensorError("wheel_encoder_ticks", ticks_total, (0, None)))
            return self._speed

        now = self._clock() if timestamp is None else float(timestamp)
        if not math.isfinite(now):
            self._reject(SensorError("wheel_encoder_time", now, (0.0, None)))
            return self._speed

        if self._last_ticks is None or self._last_t is None:
            self._last_ticks = ticks
            self._last_t = now
            self._status = "operational"
            self._last_error = None
            return self._speed

        dt = now - self._last_t
        if dt <= 0.0:
            self._reject(SensorError("wheel_encoder_dt", dt, (0.0, None)))
            return self._speed

        delta_ticks = ticks - self._last_ticks
        if delta_ticks < 0:
            # Counter width/modulus is not defined by the repository, so guessing
            # a rollover would create fabricated distance.  Re-baseline instead.
            previous_ticks = self._last_ticks
            self._counter_resets += 1
            self._last_ticks = ticks
            self._last_t = now
            self._status = "degraded"
            self._last_error = "encoder_counter_reset_or_rollover"
            logger.warning(
                "Encoder counter decreased (%s -> %s); re-baselining without inventing rollover distance",
                previous_ticks,
                ticks,
            )
            return self._speed

        self._last_ticks = ticks
        self._last_t = now
        if delta_ticks == 0:
            instantaneous_speed = 0.0
        else:
            input_revolutions = delta_ticks / float(self.ppr)
            wheel_revolutions = input_revolutions / self.gear_ratio
            distance_m = wheel_revolutions * self.wheel_c
            instantaneous_speed = distance_m / dt

        if not math.isfinite(instantaneous_speed) or abs(instantaneous_speed) > self.max_speed_mps:
            self._reject(
                SensorError(
                    "wheel_speed",
                    instantaneous_speed,
                    (-self.max_speed_mps, self.max_speed_mps),
                )
            )
            return self._speed

        self._speed = low_pass_filter(self._speed, instantaneous_speed, self.alpha)
        self._samples += 1
        self._status = "operational"
        self._last_error = None
        return self._speed

    def _reject(self, error: SensorError) -> None:
        self._rejected_samples += 1
        self._status = "degraded"
        self._last_error = str(error)
        logger.error("WheelEncoder rejected sample: %s", error)

    def get_speed(self) -> float:
        return self._speed

    def health(self) -> Dict[str, Any]:
        age = None
        if self._last_t is not None:
            age = max(0.0, self._clock() - self._last_t)
        return {
            "status": self._status,
            "speed_mps": self._speed,
            "last_ticks": self._last_ticks,
            "last_sample_age_s": age,
            "samples": self._samples,
            "counter_resets": self._counter_resets,
            "rejected_samples": self._rejected_samples,
            "last_error": self._last_error,
            "ppr": self.ppr,
            "wheel_d_m": self.wheel_d,
            "gear_ratio": self.gear_ratio,
            "alpha": self.alpha,
            "max_speed_mps": self.max_speed_mps,
        }


__all__ = ["WheelEncoder"]

if __name__ == "__main__":
    print("\n=== Running RC Wheel Encoder ===\n")
    printer.status("TEST", "Initializing Encoder test", "info")

    encoder = WheelEncoder()
    print(encoder)
    print("\n* * * * * Phase 2 * * * * *\n")

    name = "ppr"
    type_ = int

    validate = encoder._get_validated_param(name, type_, min_val=None, max_val=None)

    # Two-tick simulation: first call primes timing, second produces a speed
    s1 = encoder.update_from_ticks(ticks_total=100000)
    time.sleep(11)  # 11s control period
    s2 = encoder.update_from_ticks(ticks_total=105000)  # +5000 ticks since last

    printer.pretty("VALIDATOR", validate, "success" if isinstance(validate, (int, float)) else "error")
    printer.pretty("UPDATER", f"{s1:.3f} -> {s2:.3f} m/s", "success" if isinstance(s2, (int, float)) else "error")

    print("\n=== Encoder Demo Completed ===\n")
