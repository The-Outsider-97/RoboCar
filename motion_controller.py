"""Steering-servo and brushed-ESC control for RoboCar.

The controller exposes one atomic normalized command surface:
``send(throttle, steer)`` with both values in ``[-1, 1]``.  Hardware PWM is
implemented against either the modern CircuitPython PCA9685 API or the bundled
legacy PCA9685 driver.  Simulation is explicit and opt-in.
"""

from __future__ import annotations

import threading
import time

from typing import Any, Dict, Optional

from .hardware.PCA9685 import PCA9685 as LegacyPCA9685
from .utils.config_loader import load_global_config, get_config_section
from .utils.rc_errors import *
from .utils.rc_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("RC Controller")
printer = PrettyPrinter()

class _SimulationPWM:
    """In-memory backend used only when simulation was explicitly requested."""

    def __init__(self) -> None:
        self.pulses_us: dict[int, int] = {}

    def write_us(self, channel: int, pulse_us: int) -> None:
        self.pulses_us[int(channel)] = int(pulse_us)


class MotionController:
    """Drive the steering servo and ESC through a PCA9685.

    ``allow_simulation=False`` is intentional: a physical vehicle must not
    silently report successful actuator commands when no PWM hardware exists.
    """

    def __init__(
        self,
        *,
        config: Optional[Dict[str, Any]] = None,
        allow_simulation: bool = False,
        pwm_backend: Any = None,
    ) -> None:
        self.config = dict(config) if config is not None else load_global_config()
        self.motion_config = get_config_section("motion", self.config)
        self.allow_simulation = bool(allow_simulation)

        self._lock = threading.RLock()
        self._throttle = 0.0
        self._steer = 0.0
        self._pwm: Any = None
        self._backend_kind = "uninitialized"
        self._simulation_mode = False
        self._hardware_status = "initializing"
        self._last_error: Optional[str] = None
        self._consecutive_errors = 0

        self._pwm_freq = require_int(
            self.motion_config.get("pwm_freq_hz"),
            "motion.pwm_freq_hz",
            minimum=1,
        )
        self._steer_ch = require_int(
            self.motion_config.get("steer_channel"),
            "motion.steer_channel",
            minimum=0,
            maximum=15,
        )
        self._throttle_ch = require_int(
            self.motion_config.get("throttle_channel"),
            "motion.throttle_channel",
            minimum=0,
            maximum=15,
        )
        if self._steer_ch == self._throttle_ch:
            raise ConfigurationError(
                parameter="motion.pwm_channels",
                value=(self._steer_ch, self._throttle_ch),
                valid_range=(0, 15),
            )

        self._esc_min = require_int(self.motion_config.get("esc_min_us"), "motion.esc_min_us", minimum=500, maximum=2500)
        self._esc_max = require_int(self.motion_config.get("esc_max_us"), "motion.esc_max_us", minimum=500, maximum=2500)
        self._esc_neutral = require_int(self.motion_config.get("esc_neutral_us"), "motion.esc_neutral_us", minimum=500, maximum=2500)
        self._servo_min = require_int(self.motion_config.get("servo_min_us"), "motion.servo_min_us", minimum=500, maximum=2500)
        self._servo_max = require_int(self.motion_config.get("servo_max_us"), "motion.servo_max_us", minimum=500, maximum=2500)
        self._servo_center = require_int(self.motion_config.get("servo_center_us"), "motion.servo_center_us", minimum=500, maximum=2500)
        self._servo_max_angle_rad = require_finite_float(
            self.motion_config.get("servo_max_angle_rad"),
            "motion.servo_max_angle_rad",
            minimum=1e-6,
        )

        if not self._esc_min <= self._esc_neutral <= self._esc_max:
            raise ConfigurationError(
                parameter="motion.esc_pulse_order",
                value=(self._esc_min, self._esc_neutral, self._esc_max),
                valid_range=(self._esc_min, self._esc_max),
            )
        if not self._servo_min <= self._servo_center <= self._servo_max:
            raise ConfigurationError(
                parameter="motion.servo_pulse_order",
                value=(self._servo_min, self._servo_center, self._servo_max),
                valid_range=(self._servo_min, self._servo_max),
            )

        self._initialize_backend(pwm_backend)
        # The constructor never arms non-neutral throttle.  Failure to reach
        # neutral is a real hardware failure and is not converted to simulation.
        self.stop()
        logger.info(
            "MotionController initialized: mode=%s backend=%s frequency=%sHz",
            "simulation" if self._simulation_mode else "hardware",
            self._backend_kind,
            self._pwm_freq,
        )

    @property
    def simulation_mode(self) -> bool:
        return self._simulation_mode

    def _initialize_backend(self, injected: Any) -> None:
        if injected is not None:
            self._pwm = injected
            if hasattr(injected, "write_us"):
                self._backend_kind = "pulse_writer"
            elif hasattr(injected, "channels"):
                self._backend_kind = "circuitpython"
                if hasattr(injected, "frequency"):
                    injected.frequency = self._pwm_freq
            elif callable(getattr(injected, "set_pwm", None)):
                self._backend_kind = "legacy"
                if callable(getattr(injected, "set_pwm_freq", None)):
                    injected.set_pwm_freq(self._pwm_freq)
            else:
                raise HardwareError(
                    device="PCA9685",
                    operation="initialize_injected_backend",
                    error_details="Unsupported injected PWM backend API",
                )
            self._hardware_status = "operational"
            return

        modern_error: Optional[BaseException] = None
        try:
            import board  # type: ignore
            import busio  # type: ignore
            from adafruit_pca9685 import PCA9685 as CircuitPythonPCA9685  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            self._pwm = CircuitPythonPCA9685(i2c)
            self._pwm.frequency = self._pwm_freq
            self._backend_kind = "circuitpython"
            self._hardware_status = "operational"
            return
        except Exception as exc:
            modern_error = exc

        legacy_error: Optional[BaseException] = None
        try:
            # The bundled class expects ``address`` as its first positional
            # argument and owns its own legacy I2C device construction.
            self._pwm = LegacyPCA9685()
            self._pwm.set_pwm_freq(self._pwm_freq)
            self._backend_kind = "legacy"
            self._hardware_status = "operational"
            return
        except Exception as exc:
            legacy_error = exc

        if self.allow_simulation:
            logger.warning(
                "No usable PCA9685 backend; explicit simulation enabled. modern=%s legacy=%s",
                modern_error,
                legacy_error,
            )
            self._pwm = _SimulationPWM()
            self._backend_kind = "simulation"
            self._simulation_mode = True
            self._hardware_status = "simulation"
            return

        details = (
            f"CircuitPython backend: {modern_error!r}; "
            f"bundled legacy backend: {legacy_error!r}"
        )
        self._hardware_status = "failed"
        self._last_error = details
        raise HardwareError("PCA9685", "initialize", details)

    def _write_us(self, channel: int, pulse_us: int) -> None:
        if not 0 <= int(channel) <= 15:
            raise ConfigurationError("pwm_channel", channel, (0, 15))
        if not 500 <= int(pulse_us) <= 2500:
            raise ControlError("PWM", pulse_us, "expected pulse width in [500, 2500] us")
        if self._pwm is None:
            raise HardwareError("PCA9685", "write_pwm", "PWM backend is not initialized")

        try:
            if self._backend_kind == "circuitpython":
                duty = pulse_us_to_duty_cycle_16(pulse_us, self._pwm_freq)
                self._pwm.channels[channel].duty_cycle = duty
            elif self._backend_kind == "legacy":
                counts = pulse_us_to_pca9685_counts(pulse_us, self._pwm_freq)
                self._pwm.set_pwm(channel, 0, counts)
            elif self._backend_kind in {"pulse_writer", "simulation"}:
                self._pwm.write_us(channel, pulse_us)
            else:
                raise RuntimeError(f"Unsupported PWM backend kind: {self._backend_kind}")
        except Exception as exc:
            self._consecutive_errors += 1
            self._hardware_status = "faulty"
            self._last_error = f"{type(exc).__name__}: {exc}"
            raise HardwareError(
                device="PCA9685",
                operation="write_pwm",
                error_details=self._last_error,
            ) from exc

        self._consecutive_errors = 0
        self._last_error = None
        self._hardware_status = "simulation" if self._simulation_mode else "operational"

    def _steering_pulse_us(self, steer: float) -> int:
        half_span = (self._servo_max - self._servo_min) / 2.0
        pulse = self._servo_center + steer * half_span
        return int(round(clamp(pulse, self._servo_min, self._servo_max)))

    def _throttle_pulse_us(self, throttle: float) -> int:
        if abs(throttle) < 1e-3:
            return self._esc_neutral
        if throttle > 0.0:
            pulse = self._esc_neutral + throttle * (self._esc_max - self._esc_neutral)
        else:
            pulse = self._esc_neutral + throttle * (self._esc_neutral - self._esc_min)
        return int(round(clamp(pulse, self._esc_min, self._esc_max)))

    def send(self, throttle: float, steer: float) -> Dict[str, Any]:
        """Atomically apply normalized throttle and steering commands."""

        throttle_value = normalize_signed_command(throttle, "Throttle")
        steer_value = normalize_signed_command(steer, "Steering")
        steer_us = self._steering_pulse_us(steer_value)
        throttle_us = self._throttle_pulse_us(throttle_value)

        with self._lock:
            try:
                self._write_us(self._steer_ch, steer_us)
                self._write_us(self._throttle_ch, throttle_us)
            except HardwareError:
                self._best_effort_neutral()
                raise
            self._throttle = throttle_value
            self._steer = steer_value

        logger.debug(
            "[Motion] throttle=%.3f (%sus) steer=%.3f (%sus)",
            throttle_value,
            throttle_us,
            steer_value,
            steer_us,
        )
        return {
            "ok": True,
            "thr": throttle_value,
            "steer": steer_value,
            "thr_us": throttle_us,
            "steer_us": steer_us,
            "mode": "simulation" if self._simulation_mode else "hardware",
        }

    def _best_effort_neutral(self) -> None:
        # Never mask the original failure.  Each neutral write is attempted
        # independently because one channel may still be controllable.
        try:
            self._write_us(self._throttle_ch, self._esc_neutral)
        except Exception as exc:
            logger.critical("Failed to command ESC neutral during recovery: %s", exc)
        try:
            self._write_us(self._steer_ch, self._servo_center)
        except Exception as exc:
            logger.error("Failed to center steering during recovery: %s", exc)
        self._throttle = 0.0
        self._steer = 0.0

    def stop(self) -> Dict[str, Any]:
        """Command ESC neutral and center steering, failing if either write fails."""

        errors: list[BaseException] = []
        with self._lock:
            try:
                self._write_us(self._throttle_ch, self._esc_neutral)
            except BaseException as exc:  # hardware boundary: retain first fault
                errors.append(exc)
            try:
                self._write_us(self._steer_ch, self._servo_center)
            except BaseException as exc:
                errors.append(exc)
            self._throttle = 0.0
            self._steer = 0.0
        if errors:
            raise errors[0]
        return {"ok": True, "thr": 0.0, "steer": 0.0}

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self._hardware_status,
            "mode": "simulation" if self._simulation_mode else "hardware",
            "backend": self._backend_kind,
            "pwm_frequency_hz": self._pwm_freq,
            "throttle": self._throttle,
            "steer": self._steer,
            "consecutive_errors": self._consecutive_errors,
            "last_error": self._last_error,
        }

    def close(self) -> None:
        try:
            self.stop()
        except Exception as exc:
            logger.critical("MotionController could not confirm neutral during close: %s", exc)
        deinit = getattr(self._pwm, "deinit", None)
        if callable(deinit):
            try:
                deinit()
            except Exception as exc:
                logger.warning("PCA9685 deinit failed: %s", exc)
        self._hardware_status = "stopped"


class PIDSpeedController:
    """Filtered PID controller for desired/measured longitudinal speed."""

    def __init__(self, *, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = dict(config) if config is not None else load_global_config()
        self.speed_config = get_config_section("speed", self.config)

        # Do not cast gains to int: the repository configuration intentionally
        # uses fractional I/D gains.
        self.kp = require_finite_float(self.speed_config.get("kp"), "speed.kp")
        self.ki = require_finite_float(self.speed_config.get("ki"), "speed.ki")
        self.kd = require_finite_float(self.speed_config.get("kd"), "speed.kd")
        self.u_min = require_finite_float(self.speed_config.get("u_min"), "speed.u_min")
        self.u_max = require_finite_float(self.speed_config.get("u_max"), "speed.u_max")
        if self.u_min >= self.u_max:
            raise ConfigurationError(
                parameter="speed.output_limits",
                value=(self.u_min, self.u_max),
                valid_range=(self.u_min, self.u_max),
            )

        smoothing = self.speed_config.get("output_smoothing", 0.2)
        self._output_smoothing = require_finite_float(
            smoothing,
            "speed.output_smoothing",
            minimum=0.0,
            maximum=1.0,
        )
        self.reset()
        logger.info("PIDSpeedController initialized")

    def reset(self) -> None:
        self._e_prev = 0.0
        self._i = 0.0
        self._t_prev: Optional[float] = None
        self._last_output = 0.0

    def update(self, v_des: float, v_meas: float) -> float:
        desired = optional_finite_float(v_des)
        measured = optional_finite_float(v_meas)
        if desired is None or measured is None:
            raise ControlError("PID", v_des, v_meas)

        now = time.monotonic()
        if self._t_prev is None:
            self._t_prev = now
            return 0.0
        dt = now - self._t_prev
        self._t_prev = now
        if dt <= 0.0:
            raise ControlError("PID.dt", dt, "monotonic interval must be positive")
        dt = max(1e-3, dt)

        error = desired - measured
        self._i += error * dt
        derivative = (error - self._e_prev) / dt
        self._e_prev = error

        raw = self.kp * error + self.ki * self._i + self.kd * derivative
        bounded = clamp(raw, self.u_min, self.u_max)
        if bounded != raw and self.ki != 0.0:
            # Conservative back-calculation anti-windup: remove half of the
            # latest integration increment when saturation occurs.
            self._i -= error * dt * 0.5

        output = low_pass_filter(
            self._last_output,
            bounded,
            self._output_smoothing,
        )
        self._last_output = output
        return output

    def safe_update(self, v_des: float, v_meas: float) -> float:
        try:
            return self.update(v_des, v_meas)
        except ControlError as exc:
            logger.warning("PID rejected invalid input: %s", exc)
            return 0.0


__all__ = ["MotionController", "PIDSpeedController"]


if __name__ == "__main__":
    print("\n=== Running RC Controller ===\n")
    printer.status("TEST", "Initializing Controller test", "info")

    motion = MotionController()
    pid = PIDSpeedController()
    print(motion)
    print(pid)
    print("\n* * * * * Phase 2 - Motion * * * * *\n")

    sender = motion.send(throttle=0.2, steer=0.0)
    status = motion.get_status()
    printer.pretty("SEND",
                   f"thr={sender['thr']:.2f}, steer={sender['steer']:.2f}, "
                   f"thr_us={sender['thr_us']}, steer_us={sender['steer_us']}",
                   "success" if sender and sender.get("ok") else "error")
    printer.pretty("STATUS", status, "success" if status["status"] != "faulty" else "error")

    print("\n* * * * * Phase 3 - Speed * * * * *\n")

    up1 = pid.update(v_des=100.0, v_meas=0.0)
    time.sleep(0.02)  # simulate 20 ms control period
    up2 = pid.update(v_des=100.0, v_meas=0.0)

    is_ok = isinstance(up2, (int, float))
    printer.pretty("UPDATE", f"{up1:.3f} -> {up2:.3f}", "success" if is_ok else "error")

    print("\n=== Controller Demo Completed ===\n")
