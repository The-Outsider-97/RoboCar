
import time

from typing import Any, Dict

from .utils.config_loader import load_global_config, get_config_section
from .utils.error_handler import ConfigurationError, HardwareError, ControlError
from .hardware.PCA9685 import PCA9685
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("RC Controller")
printer = PrettyPrinter()

# -----------------------
# Motion controller abstraction (ESC + servo)
# -----------------------
class MotionController:
    """
    Drives steering servo + ESC via PCA9685 (if available) or logs as fallback.
    Compatible with MG996 (positional) and QuicRun 1060 ESC defaults.
    """
    def __init__(self):
        self.config = load_global_config()
        self.motion_config = get_config_section('motion')
        self._simulation_mode = False

        # Hardware status tracking
        self._hardware_status = "initializing"
        self._last_error = None
        self._consecutive_errors = 0

        try:
            self._throttle = 0.0  # [-1,1] reverse..forward
            self._steer = 0.0     # [-1,1] left..right
            self._pwm = None
            self._pwm_freq = int(self.motion_config.get("pwm_freq_hz", 50))
            self._steer_ch = int(self.motion_config.get("steer_channel", 0))
            self._throttle_ch = int(self.motion_config.get("throttle_channel", 1))
            self._esc_min = int(self.motion_config.get("esc_min_us", 1000))
            self._esc_max = int(self.motion_config.get("esc_max_us", 2000))
            self._esc_neutral = int(self.motion_config.get("esc_neutral_us", 1500))
            self._servo_min = int(self.motion_config.get("servo_min_us", 1000))
            self._servo_max = int(self.motion_config.get("servo_max_us", 2000))
            self._servo_center = int(self.motion_config.get("servo_center_us", 1500))
            self._servo_max_angle_rad = float(self.motion_config.get("servo_max_angle_rad", 0.6))  # ~34° either side by default

            # Try to init PCA9685 (Adafruit) if present
            try:
                import board, busio  # type: ignore
                i2c = busio.I2C(board.SCL, board.SDA)
                self._pwm = PCA9685(i2c)
                # Use the driver’s own frequency setter if present, else attribute
                if hasattr(self._pwm, "set_pwm_freq"):
                    self._pwm.set_pwm_freq(self._pwm_freq)  # Adafruit-style API
                else:
                    self._pwm.frequency = self._pwm_freq
                logger.info(f"[PCA9685] initialized @ {self._pwm_freq} Hz")
                # Arm ESC softly at neutral
                self._write_us(self._throttle_ch, self._esc_neutral)
            
            except Exception as e_hw1:
                try:
                    # Alternative library name
                    from adafruit_servokit import ServoKit  # type: ignore
                    kit = ServoKit(channels=16)
                    self._pwm = kit
                    logger.info(f"[ServoKit] initialized (16ch)")
                    self._write_us(self._throttle_ch, self._esc_neutral)
                except Exception as e_hw2:
                    # ---- SIMULATION FALLBACK (no hardware drivers found) ----
                    logger.warning(f"[PWM] No PCA9685/ServoKit driver found; using simulation. ({e_hw1} / {e_hw2})")
                    class _SimPWM:
                        """Minimal stub to absorb PWM calls in simulation."""
                        def __init__(self):
                            self.channels = [type("Chan", (), {"duty_cycle": 0})() for _ in range(16)]
                        def set_pwm(self, ch, on, off):
                            # accept calls without hardware side-effects
                            return
                    self._pwm = _SimPWM()
                    self._simulation_mode = True
                    self._hardware_status = "simulation"

        except Exception as e:
            logger.warning(f"Hardware init failed, using simulation mode: {e}")
            self._simulation_mode = True  # Enable simulation
            self._hardware_status = "simulation"

        try:
            if not self._simulation_mode:
                self._write_us(self._throttle_ch, self._esc_neutral)
            else:
                logger.debug("[SIM] Skipping hardware arm sequence")
        except Exception as e:
            logger.error(f"Arming failed: {e}")
            self._simulation_mode = True
            self._hardware_status = "simulation"

        logger.info(f"Motion Controller succesfully initialized (Mode: {self._hardware_status})")

    def _us_to_duty(self, us: int) -> int:
        period_us = 1_000_000.0 / float(self._pwm_freq)
        return int(max(0, min(4095, (us / period_us) * 4096)))

    def _write_us(self, ch: int, us: int):
        # Simulation: always succeed without error tracking
        if self._simulation_mode:
            logger.debug(f"[SIM] ch={ch}, us={us}")
            self._hardware_status = "simulation"
            self._consecutive_errors = 0
            self._last_error = None
            return
    
        try:
            if not 0 <= ch <= 15:
                raise ConfigurationError(
                    parameter="pwm_channel",
                    value=ch,
                    valid_range=(0, 15)
                )
    
            if us < 500 or us > 2500:
                raise ControlError(
                    controller="PWM",
                    command=us,
                    actual="invalid_pulse_width"
                )
    
            # If no hardware present at runtime, auto-switch to simulation
            if self._pwm is None:
                logger.warning("[PWM] Driver not initialized at runtime; switching to simulation mode.")
                self._simulation_mode = True
                self._hardware_status = "simulation"
                logger.debug(f"[SIM] ch={ch}, us={us}")
                self._consecutive_errors = 0
                self._last_error = None
                return
    
            if hasattr(self._pwm, "channels"):
                self._pwm.channels[ch].duty_cycle = self._us_to_duty(us)
            elif hasattr(self._pwm, "set_pwm"):
                self._pwm.set_pwm(ch, 0, self._us_to_duty(us))
    
            self._consecutive_errors = 0
            self._hardware_status = "operational"
    
        except Exception as e:
            self._consecutive_errors += 1
            self._hardware_status = "faulty"
            self._last_error = str(e)
    
            if self._consecutive_errors > 3:
                raise HardwareError(
                    device="PCA9685",
                    operation="write_pwm",
                    error_details=f"Repeated failures: {self._last_error}"
                ) from e

    def send(self, throttle: float, steer: float):
        # Validate input ranges
        if not -1.0 <= throttle <= 1.0:
            raise ControlError(
                controller="Throttle",
                command=throttle,
                actual="out_of_range"
            )
            
        if not -1.0 <= steer <= 1.0:
            raise ControlError(
                controller="Steering",
                command=steer,
                actual="out_of_range"
            )
 
        self._throttle = max(-1.0, min(1.0, throttle))
        self._steer = max(-1.0, min(1.0, steer))

        # Steering mapping
        steer_angle = self._steer * self._servo_max_angle_rad
        span_us = self._servo_max - self._servo_min
        half_span = span_us / 2.0
        steer_us = int(self._servo_center + (steer_angle / max(1e-6, self._servo_max_angle_rad)) * half_span)
        steer_us = max(self._servo_min, min(self._servo_max, steer_us))

        # Throttle mapping
        if abs(self._throttle) < 1e-3:
            thr_us = self._esc_neutral
        elif self._throttle > 0:
            thr_us = int(self._esc_neutral + self._throttle * (self._esc_max - self._esc_neutral))
        else:
            thr_us = int(self._esc_neutral + self._throttle * (self._esc_neutral - self._esc_min))

        try:
            self._write_us(self._steer_ch, steer_us)
            self._write_us(self._throttle_ch, thr_us)
        except HardwareError as hw_err:
            # Graceful degradation: Stop motors on critical hardware failure
            self.stop()
            logger.critical(f"Hardware failure: {hw_err}")
            raise

        logger.debug(f"[Motion] thr={self._throttle:.2f} ({thr_us}us) steer={self._steer:.2f} ({steer_us}us)")
        return {"ok": True, "thr": self._throttle, "steer": self._steer,
                "thr_us": thr_us, "steer_us": steer_us}

    def get_status(self) -> Dict[str, Any]:
        """Return controller health status"""
        return {
            "status": self._hardware_status,
            "consecutive_errors": self._consecutive_errors,
            "last_error": self._last_error
        }

    def stop(self):
        self._write_us(self._throttle_ch, self._esc_neutral)
        self._write_us(self._steer_ch, self._servo_center)
        self._throttle = 0.0
        self._steer = 0.0


class PIDSpeedController:
    def __init__(self):
        self.config = load_global_config()
        self.speed_config = get_config_section('speed')

        self.kp = int(self.speed_config.get("kp"))
        self.ki = int(self.speed_config.get("ki"))
        self.kd = int(self.speed_config.get("kd"))
        self.u_min = int(self.speed_config.get("u_min"))
        self.u_max = int(self.speed_config.get("u_max"))

        # Validate PID parameters
        if any(not isinstance(gain, (int, float)) for gain in (self.kp, self.ki, self.kd)):
            raise ConfigurationError(
                parameter="PID_gains",
                value=(self.kp, self.ki, self.kd),
                valid_range="numeric_values"
            )

        if self.u_min >= self.u_max:
            raise ConfigurationError(
                parameter="PID_output_limits",
                value=(self.u_min, self.u_max),
                valid_range="u_min < u_max"
            )

        self.kp, self.ki, self.kd = self.kp, self.ki, self.kd
        self.u_min, self.u_max = self.u_min, self.u_max
        self.reset()

        # Smoothing factors
        self._output_smoothing = 0.2
        self._last_output = 0.0

        logger.info(f"PID Speed Controller succesfully initialized")

    def reset(self):
        """Full controller reset"""
        self._e_prev = 0.0
        self._i = 0.0
        self._t_prev = None
        self._last_output = 0.0

    def update(self, v_des: float, v_meas: float) -> float:
        # Validate inputs
        if not isinstance(v_des, (int, float)) or not isinstance(v_meas, (int, float)):
            raise ControlError(
                controller="PID",
                command=v_des,
                actual=v_meas
            )

        t = time.time()
        if self._t_prev is None:
            self._t_prev = t
            return 0.0
        dt = max(1e-3, t - self._t_prev)
        self._t_prev = t

        e = float(v_des) - float(v_meas)
        self._i += e * dt
        d = (e - self._e_prev) / dt
        self._e_prev = e

        u = self.kp * e + self.ki * self._i + self.kd * d
        # anti-windup clamp
        if u > self.u_max:
            u = self.u_max
            self._i -= e * dt * 0.5
        elif u < self.u_min:
            u = self.u_min
            self._i -= e * dt * 0.5

        # Smoothing to prevent abrupt changes
        smoothed_output = (self._output_smoothing * u + 
                          (1 - self._output_smoothing) * self._last_output)
        self._last_output = smoothed_output
        
        return smoothed_output

    def safe_update(self, v_des: float, v_meas: float) -> float:
        """Update with built-in safety checks"""
        try:
            return self.update(v_des, v_meas)
        except ControlError:
            # Graceful degradation: Return neutral on invalid input
            return 0.0

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
