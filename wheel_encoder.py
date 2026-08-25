
import math
import time

from .utils.config_loader import load_global_config, get_config_section
from .utils.error_handler import ConfigurationError, SensorError
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("RC Wheel Encoder")
printer = PrettyPrinter()

class WheelEncoder:
    """
    Monotonic ticks → m/s estimate with error handling and validation.
    Update with total ticks (e.g., from GPIO ISR).
    """
    MAX_SPEED_MPS = 5.5555555556  # in m/s = 20 km/h

    def __init__(self):
        try:
            self.config = load_global_config()
            self.wheel_config = get_config_section('encoder')

            self.pulses_per_rev = int(self.wheel_config.get("ppr"))
            self.wheel_diameter_m = float(self.wheel_config.get("wheel_d"))
            self.gear_ratio = float(self.wheel_config.get("gear_ratio"))
            self.alpha = float(self.wheel_config.get("alpha"))

            self.wheel_circ = math.pi * self.wheel_diameter_m
            self._last_ticks = None
            self._last_t = None
            self._speed = 0.0
        except KeyError as e:
            raise ConfigurationError(
                parameter=str(e).strip("'"),
                value=None,
                valid_range=(None, None)
            ) from e

        logger.info(f"Wheel Encoder succesfully initialized")

    def _get_validated_param(self, name, type_, min_val=None, max_val=None):
        """Fetch and validate configuration parameter"""
        try:
            value = type_(self.wheel_config[name])
            if min_val is not None and value < min_val:
                raise ConfigurationError(
                    parameter=name,
                    value=value,
                    valid_range=(min_val, max_val)
                )
            if max_val is not None and value > max_val:
                raise ConfigurationError(
                    parameter=name,
                    value=value,
                    valid_range=(min_val, max_val)
                )
            return value
        except KeyError:
            logger.error(f"Missing config parameter: {name}")
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"Invalid value for {name}: {self.wheel_config[name]}")
            raise ConfigurationError(
                parameter=name,
                value=self.wheel_config[name],
                valid_range=(min_val, max_val)
            ) from e

    def update_from_ticks(self, ticks_total: int) -> float:
        """Update speed estimate with new tick count"""
        try:
            ticks_total = int(ticks_total)
            now = time.time()
            
            if self._last_ticks is None:  # First reading
                self._last_ticks = ticks_total
                self._last_t = now
                return self._speed

            if self._last_t is None:
                self._last_t = now
                return self._speed

            dt = max(1e-3, now - self._last_t)
            d_ticks = ticks_total - self._last_ticks
            
            # Handle counter overflow/reset
            if d_ticks < 0:
                logger.warning("Encoder tick reset detected")
                d_ticks = ticks_total  # Treat as new baseline
            
            revs = (d_ticks / self.pulses_per_rev) / self.gear_ratio
            v = (revs * self.wheel_circ) / dt
            
            # Validate speed reading
            if abs(v) > self.MAX_SPEED_MPS:
                raise SensorError(
                    sensor_type="WheelEncoder",
                    reading=v,
                    expected_range=(-self.MAX_SPEED_MPS, self.MAX_SPEED_MPS)
                )
                
            # Apply low-pass filter
            self._speed = self.alpha * v + (1 - self.alpha) * self._speed
            self._last_ticks = ticks_total
            self._last_t = now
            return self._speed
            
        except SensorError as e:
            logger.error(f"Sensor error: {e}")
            self._reset_state()
            return 0.0
        except Exception as e:
            logger.exception("Unexpected error in speed calculation")
            return self._speed

    def get_speed(self) -> float:
        return float(self._speed)

    def _reset_state(self):
        """Reset internal state after error"""
        self._last_ticks = None
        self._last_t = None
        self._speed = 0.0

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
