"""Host-side Pico sensor and lighting gateway for RoboCar.

The Pico is the deterministic sensor/IO side of the vehicle. This module owns
the Raspberry-Pi-side serial transport, frame normalization, and the semantic
lighting policy sent to the Pico. It does not re-implement the Pico's I2C/GPIO
drivers or time-critical LED PWM/blink scheduler.

Supported sensor-frame protocols:

* one JSON object per line;
* comma-separated ``KEY:VALUE`` pairs.

Lighting commands use one compact JSON object per line. The Pico firmware must
execute the requested pattern locally so Linux scheduling and serial latency do
not determine the visible blink rate.

Simulation is intentionally opt-in. Missing pyserial, an unavailable Pico, or
an open/read failure must not silently replace physical measurements with
plausible synthetic values on a real vehicle.
"""

from __future__ import annotations

import json
import queue
import threading
import time

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional

from .utils.rc_errors import *
from .utils.rc_helpers import *
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SensorBus")
printer = PrettyPrinter()


# === Pico H Pin Map (Grove Shield v1.0) ======================================
# This mirrors the finished wiring:
# - I2C0  (GP4=SDA, GP5=SCL):  MPU6050 (IMU)
# - I2C1  (GP6=SDA, GP7=SCL):  Grove I2C Hub (6-port) with VL53L0X (GY-530) and TLV493D.
#   NOTE: The Grove I2C Hub is just a passive splitter, so ONLY I2C devices should be on it.\
#         If you use ultrasonic HC-SR04P modules (which are NOT I2C), connect them to normal\
#         GPIOs as below, or use the Grove Ultrasonic *I2C* variant instead.
# - D16  -> front white LED      (GPIO16)
# - D18  -> rear  red  LED       (GPIO18)
# - GP2  -> yellow link/status
# - GP22 -> yellow link/status
# - Hall magnetic sensor signal on physical pin 17 (GPIO13). Power = 3V3 (pin 36) + GND (pin 38).
#   ⚠︎ Pin 37 is 3V3_EN, not a power rail. Do not power sensors from pin 37.  See Pico datasheet.
#
# nRF24 PA/LNA radio (SPI) pins are reserved here for reference only; this module is handled by
# a different program. You can remap SPI to several GPIOs on RP2040; keep CE/CSN on free GPIOs.

# === Pico H Pin Map (Grove Shield v1.0) ======================================
# I2C0 bus (IMU)
PICO_I2C0_SDA = 4
PICO_I2C0_SCL = 5

# I2C1 bus (hub + ToF/magnetometer)
PICO_I2C1_SDA = 6
PICO_I2C1_SCL = 7
PICO_TOF_GPIO1 = 26
PICO_TOF_XSHUT= 27

# Ultrasonic sensors (HC-SR04P) — if present on GPIO (NOT I2C)
# Choose non-I2C pins to avoid conflicts with the Grove I2C ports (6/7).
PICO_US_FRONT_TRIG = 10
PICO_US_FRONT_ECHO = 11  # use a 5V->3V3 divider on ECHO if your board is 5V logic
PICO_US_REAR_TRIG  = 14
PICO_US_REAR_ECHO  = 15  # use a 5V->3V3 divider on ECHO

# Hall magnetic sensor (digital)
PICO_HALL = 13  # physical pin 17

# LEDs on Grove digital ports
PICO_LED_FRONT  = 16  # D16
PICO_LED_REAR   = 18  # D18
PICO_LED_LEFT_INDICATOR = 2
PICO_LED_RIGHT_INDICATOR = 22

# Optional: VBAT ADC via divider (if you add it later)
# PICO_VBAT_ADC = 26  # ADC0, through resistor divider to <=3.3V

# I2C addresses (typical; confirm on your breakout or via i2c scan)
I2C_ADDR = {
    "MPU6050": 0x68,     # 0x69 if AD0 pulled high
    "VL53L0X": 0x29,
    "TLV493D": 0x5E      # Adafruit default
}

# Structured map for programmatic use / export to JSON
PICO_PINMAP = {
    "i2c0": {"sda": PICO_I2C0_SDA, "scl": PICO_I2C0_SCL, "devices": ["MPU6050"]},
    "i2c1": {"sda": PICO_I2C1_SDA, "scl": PICO_I2C1_SCL, "hub": True, "devices": ["VL53L0X", "TLV493D"]},
    "vl53l0x": {
        "bus": 1,
        "address": ["VL53L0X"],
        "gpio1": PICO_TOF_GPIO1,
        "xshut": PICO_TOF_XSHUT,
        "gpio1_direction": "input",
        "xshut_direction": "output_active_low",
    },
    "ultrasonic": {
        "mode": "gpio",
        "front": {"trig": PICO_US_FRONT_TRIG, "echo": PICO_US_FRONT_ECHO,
                  "divider_top_ohm": 1800, "divider_bottom_ohm": 3300},
        "rear": {"trig": PICO_US_REAR_TRIG, "echo": PICO_US_REAR_ECHO,
                 "divider_top_ohm": 1800, "divider_bottom_ohm": 3300},
                 },
    "hall": {"signal": PICO_HALL},
    "led": {
        "headlights": PICO_LED_FRONT,
        "taillights": PICO_LED_REAR,
        "left_indicator": PICO_LED_LEFT_INDICATOR,
        "right_indicator": PICO_LED_RIGHT_INDICATOR,
    },
    "i2c_addr": I2C_ADDR,
    "notes": {
        "grove_shield_ports": {
            "D16": PICO_LED_FRONT,
            "D18": PICO_LED_REAR,
        },
        "direct_gpio": {
            "GP2": PICO_LED_LEFT_INDICATOR,
            "GP22": PICO_LED_RIGHT_INDICATOR,
        },
        "power_warning": (
            "Use pin 36 (3V3) for sensor power. "
            "Pin 37 is 3V3_EN, not a supply."
        ),
    },
}


def pico_pinmap_json() -> str:
    """Return the hardware map in deterministic JSON form."""

    return json.dumps(
        PICO_PINMAP,
        separators=(",", ":"),
        sort_keys=True,
    )

# =============================================================================
# Lighting types
# =============================================================================


class LightingMode(str, Enum):
    """Mutually exclusive vehicle-lighting operating modes."""

    OFF = "off"
    DRIVE = "drive"
    TURN_LEFT = "turn_left"
    TURN_RIGHT = "turn_right"
    OBSTACLE = "obstacle"
    PARKING_PULSE = "parking_pulse"


@dataclass(frozen=True, slots=True)
class LightingCommand:
    """One semantic, edge-triggered lighting command for the Pico firmware.

    The Raspberry Pi selects the required lighting policy, while the Pico
    firmware executes the requested pulse or blink timing locally.

    Attributes
    ----------
    sequence:
        Monotonically increasing host-side command sequence.
    mode:
        Effective lighting mode.
    headlights:
        Whether the GP16 headlight channel must remain on.
    taillights:
        Whether the GP18 taillight channel must remain on.
    left_pattern:
        Pattern for GP2: ``off``, ``blink``, or ``pulse``.
    right_pattern:
        Pattern for GP22: ``off``, ``blink``, or ``pulse``.
    frequency_hz_low:
        Primary or lower pattern frequency.
    frequency_hz_high:
        Optional upper alternating frequency.
    duty_cycle:
        Blink duty cycle. It is not required for the sine-wave parking pulse.
    duration_s:
        Optional finite pattern duration.
    terminal_mode:
        Mode the Pico must enter when ``duration_s`` expires.
    obstacle_distance_m:
        Nearest contributing obstacle distance for diagnostics.
    issued_at_monotonic:
        Host monotonic command-creation timestamp.
    """

    sequence: int
    mode: LightingMode
    headlights: bool
    taillights: bool
    left_pattern: str = "off"
    right_pattern: str = "off"
    frequency_hz_low: Optional[float] = None
    frequency_hz_high: Optional[float] = None
    duty_cycle: Optional[float] = None
    duration_s: Optional[float] = None
    terminal_mode: Optional[LightingMode] = None
    obstacle_distance_m: Optional[float] = None
    issued_at_monotonic: float = 0.0

    def signature(self) -> tuple[Any, ...]:
        """Return the fields that determine physical output.

        Sequence number, diagnostic distance, and timestamp are deliberately
        excluded so insignificant sensor-distance changes do not create
        duplicate serial commands when the physical pattern is unchanged.
        """

        return (
            self.mode,
            self.headlights,
            self.taillights,
            self.left_pattern,
            self.right_pattern,
            self.frequency_hz_low,
            self.frequency_hz_high,
            self.duty_cycle,
            self.duration_s,
            self.terminal_mode,
        )

    def to_payload(self) -> dict[str, Any]:
        """Serialize this command to the Pico lighting protocol."""

        def indicator_payload(pattern: str) -> dict[str, Any]:
            channel: dict[str, Any] = {
                "pattern": pattern,
            }

            if pattern not in {"blink", "pulse"}:
                return channel

            if self.frequency_hz_low is not None:
                channel["frequency_hz"] = self.frequency_hz_low

            if (
                self.frequency_hz_high is not None
                and self.frequency_hz_high != self.frequency_hz_low
            ):
                channel["frequency_hz_range"] = [
                    self.frequency_hz_low,
                    self.frequency_hz_high,
                ]
                channel["rate_selection"] = "alternate_per_cycle"

            if self.duty_cycle is not None:
                channel["duty_cycle"] = self.duty_cycle

            if pattern == "pulse":
                channel["waveform"] = "sine"

            return channel

        payload: dict[str, Any] = {
            "type": "robocar_command",
            "command": "lighting",
            "protocol_version": 1,
            "sequence": self.sequence,
            "mode": self.mode.value,
            "headlights": int(self.headlights),
            "taillights": int(self.taillights),
            "left_indicator": indicator_payload(self.left_pattern),
            "right_indicator": indicator_payload(self.right_pattern),
        }

        if self.duration_s is not None:
            payload["duration_s"] = self.duration_s

        if self.terminal_mode is not None:
            payload["terminal_mode"] = self.terminal_mode.value

        if self.obstacle_distance_m is not None:
            payload["obstacle_distance_m"] = self.obstacle_distance_m

        return payload


# =============================================================================
# Lighting controller
# =============================================================================

class VehicleLightingController:
    """Thread-safe policy controller for the four vehicle-lighting channels.

    Lighting priority
    -----------------
    1. Explicit parking acknowledgement.
    2. Obstacle warning.
    3. Approaching-turn indication.
    4. Normal driving lights.
    5. All channels off.

    Obstacle warning
    ----------------
    The nearest valid measurement from the front ultrasonic sensor, rear
    ultrasonic sensor, and VL53L0X is used.

    At 0.50 m, the indicators alternate between 2 and 3 complete blink cycles
    per second. The lower and upper rates increase linearly as the obstacle
    approaches. At 0.05 m or closer, they alternate between 12 and 18 cycles
    per second.

    Turn indication
    ---------------
    The selected left or right indicator starts blinking when the supplied
    distance to the turn becomes less than or equal to 1.0 m.

    Parking indication
    ------------------
    Both indicators receive a 1 Hz sine-wave pulse command for 30 seconds. The
    head and tail lights are off during the parking acknowledgement. After
    30 seconds, the terminal mode becomes ``off``.
    """

    _TURN_DIRECTIONS = {
        "left",
        "right",
    }

    def __init__(
        self,
        command_sink: Callable[[LightingCommand], Any],
        *,
        config: Optional[Mapping[str, Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not callable(command_sink):
            raise TypeError("command_sink must be callable")

        if not callable(clock):
            raise TypeError("clock must be callable")

        cfg = dict(config or {})

        self.obstacle_start_m = self._positive_config(cfg, "obstacle_start_m", 0.50)
        self.obstacle_full_rate_m = self._positive_config(cfg, "obstacle_full_rate_m", 0.05)

        if self.obstacle_full_rate_m >= self.obstacle_start_m:
            raise ValueError(
                "lighting.obstacle_full_rate_m must be smaller than "
                "lighting.obstacle_start_m"
            )

        self.obstacle_slow_hz_low = self._positive_config(cfg, "obstacle_slow_hz_low", 2.0)
        self.obstacle_slow_hz_high = self._positive_config(cfg, "obstacle_slow_hz_high", 3.0)
        self.obstacle_fast_hz_low = self._positive_config(cfg, "obstacle_fast_hz_low", 6.0)
        self.obstacle_fast_hz_high = self._positive_config(cfg, "obstacle_fast_hz_high", 9.0)

        if not (
            self.obstacle_slow_hz_low
            <= self.obstacle_slow_hz_high
            <= self.obstacle_fast_hz_high
        ):
            raise ValueError(
                "lighting obstacle upper-rate bounds are inconsistent"
            )

        if not (
            self.obstacle_slow_hz_low
            <= self.obstacle_fast_hz_low
            <= self.obstacle_fast_hz_high
        ):
            raise ValueError("lighting obstacle lower-rate bounds are inconsistent")

        self.turn_activation_m = self._positive_config(cfg, "turn_activation_m", 1.0)
        self.turn_blink_hz = self._positive_config(cfg, "turn_blink_hz", 1.0)
        self.parking_pulse_hz = self._positive_config(cfg, "parking_pulse_hz", 1.0)
        self.parking_duration_s = self._positive_config(cfg, "parking_duration_s", 20.0)

        self._command_sink = command_sink
        self._clock = clock
        self._lock = threading.RLock()

        self._sequence = 0
        self._drive_intent = False
        self._turn_direction: Optional[str] = None
        self._distance_to_turn_m: Optional[float] = None
        self._nearest_obstacle_m: Optional[float] = None
        self._parking_started_monotonic: Optional[float] = None

        self._last_command: Optional[LightingCommand] = None
        self._command_errors = 0
        self._last_error: Optional[str] = None

    @staticmethod
    def _positive_config(
        config: Mapping[str, Any],
        name: str,
        default: float,
    ) -> float:
        """Read and validate one strictly positive lighting value."""

        value = optional_finite_float(config.get(name, default), minimum=0.0)

        if value is None or value <= 0.0:
            raise ValueError(f"lighting.{name} must be finite and greater than zero")

        return value

    @property
    def last_command(self) -> Optional[LightingCommand]:
        """Return the latest successfully delivered lighting command."""

        with self._lock:
            return self._last_command

    def set_drive_intent(
        self,
        keep_driving: bool,
        *,
        force: bool = False,
    ) -> LightingCommand:
        """Set whether the car intends to drive or continue driving.

        Setting this to ``True`` keeps the head and tail lights on even while
        the vehicle is temporarily stopped between motion commands.

        Setting it to ``False`` clears the turn intent unless a parking
        acknowledgement is currently active.
        """

        if not isinstance(keep_driving, bool):
            raise TypeError("keep_driving must be bool")

        with self._lock:
            self._drive_intent = keep_driving

            if keep_driving:
                self._parking_started_monotonic = None
            elif self._parking_started_monotonic is None:
                self._turn_direction = None
                self._distance_to_turn_m = None

            return self._reconcile_locked(force=force)

    def set_turn_intent(
        self,
        direction: Optional[str],
        distance_to_turn_m: Optional[float],
        *,
        force: bool = False,
    ) -> LightingCommand:
        """Set or clear the next turn and its remaining distance.

        Parameters
        ----------
        direction:
            ``"left"``, ``"right"``, or ``None`` to clear the turn.
        distance_to_turn_m:
            Non-negative distance to the beginning of the turn. It is required
            when ``direction`` is not ``None``.
        force:
            Deliver the command even if its physical signature is unchanged.
        """

        normalized_direction = (
            None
            if direction is None
            else str(direction).strip().lower()
        )

        if (
            normalized_direction is not None
            and normalized_direction not in self._TURN_DIRECTIONS
        ):
            raise ValueError(
                "direction must be 'left', 'right', or None"
            )

        if normalized_direction is None:
            parsed_distance = None
        else:
            parsed_distance = optional_finite_float(
                distance_to_turn_m,
                minimum=0.0,
            )

            if parsed_distance is None:
                raise ValueError(
                    "distance_to_turn_m must be finite and non-negative "
                    "when a turn direction is supplied"
                )

        with self._lock:
            self._turn_direction = normalized_direction
            self._distance_to_turn_m = parsed_distance
            return self._reconcile_locked(force=force)

    def observe(
        self,
        reading: "SensorReading",
        *,
        force: bool = False,
    ) -> LightingCommand:
        """Update the obstacle-warning state from one sensor frame."""

        if not isinstance(reading, SensorReading):
            raise TypeError("reading must be SensorReading")

        candidates: list[float] = []

        for value in (
            reading.ultra_front_m,
            reading.ultra_rear_m,
        ):
            parsed = optional_finite_float(
                value,
                minimum=0.0,
            )
            if parsed is not None:
                candidates.append(parsed)

        tof_mm = optional_int(
            reading.tof_mm,
            minimum=0,
        )
        if tof_mm is not None:
            candidates.append(float(tof_mm) / 1000.0)

        with self._lock:
            self._nearest_obstacle_m = (
                min(candidates)
                if candidates
                else None
            )
            return self._reconcile_locked(force=force)

    def park(
        self,
        *,
        force: bool = False,
    ) -> LightingCommand:
        """Start the 30-second parking acknowledgement."""

        with self._lock:
            self._drive_intent = False
            self._turn_direction = None
            self._distance_to_turn_m = None
            self._parking_started_monotonic = self._clock()

            return self._reconcile_locked(force=force)

    def all_off(
        self,
        *,
        force: bool = False,
    ) -> LightingCommand:
        """Cancel all lighting intents and command every channel off."""

        with self._lock:
            self._drive_intent = False
            self._turn_direction = None
            self._distance_to_turn_m = None
            self._nearest_obstacle_m = None
            self._parking_started_monotonic = None

            return self._reconcile_locked(force=force)

    def refresh(
        self,
        *,
        force: bool = False,
    ) -> LightingCommand:
        """Re-evaluate time-dependent state without blocking."""

        with self._lock:
            return self._reconcile_locked(force=force)

    def health(self) -> dict[str, Any]:
        """Return the current lighting-controller health snapshot."""

        with self._lock:
            return {
                "mode": (
                    self._last_command.mode.value
                    if self._last_command is not None
                    else LightingMode.OFF.value
                ),
                "drive_intent": self._drive_intent,
                "turn_direction": self._turn_direction,
                "distance_to_turn_m": self._distance_to_turn_m,
                "nearest_obstacle_m": self._nearest_obstacle_m,
                "commands_issued": self._sequence,
                "command_errors": self._command_errors,
                "last_error": self._last_error,
            }

    def _reconcile_locked(
        self,
        *,
        force: bool,
    ) -> LightingCommand:
        """Resolve and deliver the effective lighting mode.

        The caller must hold ``self._lock``.
        """

        now = self._clock()
        desired = self._desired_command_locked(now)
        previous = self._last_command

        if (
            not force
            and previous is not None
            and desired.signature() == previous.signature()
        ):
            return previous

        self._sequence += 1

        command = LightingCommand(
            sequence=self._sequence,
            mode=desired.mode,
            headlights=desired.headlights,
            taillights=desired.taillights,
            left_pattern=desired.left_pattern,
            right_pattern=desired.right_pattern,
            frequency_hz_low=desired.frequency_hz_low,
            frequency_hz_high=desired.frequency_hz_high,
            duty_cycle=desired.duty_cycle,
            duration_s=desired.duration_s,
            terminal_mode=desired.terminal_mode,
            obstacle_distance_m=desired.obstacle_distance_m,
            issued_at_monotonic=now,
        )

        try:
            accepted = self._command_sink(command)

            if accepted is False:
                raise RuntimeError(
                    "lighting command sink rejected the command"
                )

            self._last_command = command
            self._last_error = None

        except Exception as exc:
            self._command_errors += 1
            self._last_error = (
                f"{type(exc).__name__}: {exc}"
            )
            logger.exception(
                "Lighting command delivery failed"
            )

        return command

    def _desired_command_locked(
        self,
        now: float,
    ) -> LightingCommand:
        """Build the currently required lighting command.

        The caller must hold ``self._lock``.
        """

        parked_at = self._parking_started_monotonic

        # Highest priority: explicit stationary parking.
        if parked_at is not None:
            elapsed = max(
                0.0,
                now - parked_at,
            )

            if elapsed < self.parking_duration_s:
                return LightingCommand(
                    sequence=0,
                    mode=LightingMode.PARKING_PULSE,
                    headlights=False,
                    taillights=False,
                    left_pattern="pulse",
                    right_pattern="pulse",
                    frequency_hz_low=self.parking_pulse_hz,
                    frequency_hz_high=self.parking_pulse_hz,
                    duration_s=self.parking_duration_s,
                    terminal_mode=LightingMode.OFF,
                )

            self._parking_started_monotonic = None

        # An inactive vehicle with no parking acknowledgement is dark.
        if not self._drive_intent:
            return LightingCommand(
                sequence=0,
                mode=LightingMode.OFF,
                headlights=False,
                taillights=False,
            )

        # Second priority: obstacle warning.
        obstacle = self._nearest_obstacle_m

        if (
            obstacle is not None
            and obstacle <= self.obstacle_start_m
        ):
            low_hz, high_hz = self._obstacle_frequency_range(
                obstacle
            )

            return LightingCommand(
                sequence=0,
                mode=LightingMode.OBSTACLE,
                headlights=True,
                taillights=True,
                left_pattern="blink",
                right_pattern="blink",
                frequency_hz_low=low_hz,
                frequency_hz_high=high_hz,
                duty_cycle=0.5,
                obstacle_distance_m=round(obstacle, 3),
            )

        # Third priority: approaching turn.
        turn_is_active = (
            self._turn_direction in self._TURN_DIRECTIONS
            and self._distance_to_turn_m is not None
            and self._distance_to_turn_m <= self.turn_activation_m
        )

        if turn_is_active:
            is_left = self._turn_direction == "left"

            return LightingCommand(
                sequence=0,
                mode=(
                    LightingMode.TURN_LEFT
                    if is_left
                    else LightingMode.TURN_RIGHT
                ),
                headlights=True,
                taillights=True,
                left_pattern=(
                    "blink"
                    if is_left
                    else "off"
                ),
                right_pattern=(
                    "off"
                    if is_left
                    else "blink"
                ),
                frequency_hz_low=self.turn_blink_hz,
                frequency_hz_high=self.turn_blink_hz,
                duty_cycle=0.5,
            )

        # Lowest active priority: ordinary driving lights.
        return LightingCommand(
            sequence=0,
            mode=LightingMode.DRIVE,
            headlights=True,
            taillights=True,
        )

    def _obstacle_frequency_range(
        self,
        distance_m: float,
    ) -> tuple[float, float]:
        """Return the lower and upper obstacle-warning frequencies."""

        clamped_distance = min(
            self.obstacle_start_m,
            max(
                self.obstacle_full_rate_m,
                distance_m,
            ),
        )

        closeness = (
            self.obstacle_start_m - clamped_distance
        ) / (
            self.obstacle_start_m - self.obstacle_full_rate_m
        )

        low_hz = self.obstacle_slow_hz_low + closeness * (
            self.obstacle_fast_hz_low
            - self.obstacle_slow_hz_low
        )
        high_hz = self.obstacle_slow_hz_high + closeness * (
            self.obstacle_fast_hz_high
            - self.obstacle_slow_hz_high
        )

        # Quantization prevents insignificant range noise from flooding the
        # shared serial connection with semantically equivalent commands.
        quantized_low = round(low_hz * 2.0) / 2.0
        quantized_high = round(high_hz * 2.0) / 2.0

        return quantized_low, quantized_high


# =============================================================================
# Sensor frame
# =============================================================================

@dataclass(slots=True)
class SensorReading:
    """One normalized host-side sensor frame.

    ``t`` is the Raspberry Pi wall-clock receipt timestamp.

    ``encoder_ticks_total`` is populated only when the Pico sends an
    accumulated tick counter. The digital ``hall`` level is retained
    separately and must not be treated as a complete wheel-encoder count.
    """

    t: float
    ultra_front_m: Optional[float] = None
    ultra_rear_m: Optional[float] = None
    tof_mm: Optional[int] = None
    hall: Optional[int] = None
    encoder_ticks_total: Optional[int] = None
    imu_ax: Optional[float] = None
    imu_ay: Optional[float] = None
    imu_az: Optional[float] = None
    imu_gx: Optional[float] = None
    imu_gy: Optional[float] = None
    imu_gz: Optional[float] = None
    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    led_front: Optional[int] = None
    led_rear: Optional[int] = None
    led_left_indicator: Optional[int] = None
    led_right_indicator: Optional[int] = None
    led_signal: Optional[int] = None
    vbat: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Return the frame as a detached dictionary."""
        return asdict(self)


# =============================================================================
# Sensor/IO serial gateway
# =============================================================================


class SensorBus:
    """Threaded serial gateway for Pico sensor and lighting communication.

    Parameters
    ----------
    port:
        Device path, or ``"auto"``/``None`` for USB serial autodetection.
    baud:
        Serial baud rate.
    qmax:
        Maximum number of frames retained for polling. The newest frame wins
        when the producer outruns the consumer.
    allow_simulation:
        Explicit permission to synthesize frames when physical serial transport
        cannot be started.
    lighting_config:
        Optional lighting-policy overrides. Missing values use the requirements
        defined by :class:`VehicleLightingController`.
    """

    def __init__(
        self,
        port: Optional[str] = "auto",
        baud: int = 115200,
        qmax: int = 64,
        *,
        allow_simulation: bool = False,
        lighting_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.port = port
        self.baud = require_int(
            baud,
            "sensor_bus.baud",
            minimum=1,
        )
        queue_size = require_int(
            qmax,
            "sensor_bus.qmax",
            minimum=1,
        )
        self.allow_simulation = bool(allow_simulation)

        self._ser: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

        self._q: "queue.Queue[SensorReading]" = queue.Queue(
            maxsize=queue_size
        )
        self._latest: Optional[SensorReading] = None
        self._simulation = False

        self._subscribers: list[
            Callable[[SensorReading], None]
        ] = []
        self._subscriber_lock = threading.RLock()
        self._write_lock = threading.RLock()

        self._status = "stopped"
        self._resolved_port: Optional[str] = None
        self._last_error: Optional[str] = None

        self._frames_received = 0
        self._parse_errors = 0
        self._transport_errors = 0
        self._callback_errors = 0
        self._dropped_frames = 0
        self._commands_sent = 0
        self._command_errors = 0

        self._last_frame_monotonic: Optional[float] = None

        self.lighting = VehicleLightingController(
            self._write_lighting_command,
            config=lighting_config,
        )

    @property
    def is_simulation(self) -> bool:
        """Return whether explicit simulation is active."""

        return self._simulation

    @property
    def running(self) -> bool:
        """Return whether the acquisition thread is alive."""

        return bool(
            self._thread
            and self._thread.is_alive()
        )

    def start(self) -> None:
        """Start physical serial acquisition or explicit simulation."""

        if self.running:
            return

        self._stop.clear()
        self._simulation = False
        self._last_error = None

        try:
            import serial  # type: ignore
        except Exception as exc:
            self._start_simulation_or_raise(
                "pyserial_unavailable",
                exc,
            )
            return

        resolved_port = (
            self._autodetect_port()
            if self.port in (None, "auto")
            else self.port
        )

        if not resolved_port:
            self._start_simulation_or_raise(
                "serial_port_not_found"
            )
            return

        try:
            self._ser = serial.Serial(
                resolved_port,
                self.baud,
                timeout=1,
            )
        except Exception as exc:
            self._start_simulation_or_raise(
                "serial_open_failed",
                exc,
            )
            return

        self._resolved_port = str(resolved_port)
        self._status = "operational"

        self._thread = threading.Thread(
            target=self._loop,
            name="RoboCarSensorBus",
            daemon=True,
        )
        self._thread.start()

        # Explicitly establish a known all-off state after transport startup.
        self.lighting.refresh(force=True)

        logger.info(
            "SensorBus started on %s at %s baud",
            resolved_port,
            self.baud,
        )

    def stop(self) -> None:
        """Stop acquisition and close the serial device."""

        # Attempt the safe output state before serial transport is removed.
        if self._ser is not None or self._simulation:
            self.lighting.all_off(force=True)

        self._stop.set()

        thread = self._thread
        if (
            thread is not None
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=1.5)

        self._thread = None

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception as exc:
                logger.warning(
                    "SensorBus serial close failed: %s",
                    exc,
                )

        self._ser = None
        self._status = "stopped"

    close = stop

    def subscribe(
        self,
        callback: Callable[[SensorReading], None],
    ) -> Callable[[SensorReading], None]:
        """Register a frame callback and return it for unsubscription."""

        if not callable(callback):
            raise TypeError("callback must be callable")

        with self._subscriber_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

        return callback

    def unsubscribe(
        self,
        callback: Callable[[SensorReading], None],
    ) -> bool:
        """Remove a previously registered frame callback."""

        with self._subscriber_lock:
            try:
                self._subscribers.remove(callback)
                return True
            except ValueError:
                return False

    def latest(self) -> Optional[SensorReading]:
        """Return the latest successfully parsed frame."""

        return self._latest

    def poll_nowait(self) -> Optional[SensorReading]:
        """Return one queued frame without blocking."""

        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def health(self) -> dict[str, Any]:
        """Return transport, parsing, and lighting health."""

        frame_age_s: Optional[float] = None

        if self._last_frame_monotonic is not None:
            frame_age_s = max(
                0.0,
                time.monotonic()
                - self._last_frame_monotonic,
            )

        return {
            "status": self._status,
            "running": self.running,
            "mode": (
                "simulation"
                if self._simulation
                else "hardware"
            ),
            "simulation_allowed": self.allow_simulation,
            "port_requested": self.port,
            "port_resolved": self._resolved_port,
            "baud": self.baud,
            "frames_received": self._frames_received,
            "parse_errors": self._parse_errors,
            "transport_errors": self._transport_errors,
            "callback_errors": self._callback_errors,
            "dropped_frames": self._dropped_frames,
            "commands_sent": self._commands_sent,
            "command_errors": self._command_errors,
            "last_frame_age_s": frame_age_s,
            "lighting": self.lighting.health(),
            "last_error": self._last_error,
        }

    def set_drive_intent(
        self,
        keep_driving: bool,
    ) -> LightingCommand:
        """Keep driving lights on while moving or paused to continue."""

        return self.lighting.set_drive_intent(
            keep_driving
        )

    def set_turn_intent(
        self,
        direction: Optional[str],
        distance_to_turn_m: Optional[float],
    ) -> LightingCommand:
        """Set an upcoming turn and its along-route distance."""

        return self.lighting.set_turn_intent(
            direction,
            distance_to_turn_m,
        )

    def park_lighting(self) -> LightingCommand:
        """Start the finite parking pulse pattern."""

        return self.lighting.park()

    def service_lighting(self) -> LightingCommand:
        """Advance time-dependent lighting state without blocking."""

        return self.lighting.refresh()

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    def _start_simulation_or_raise(
        self,
        reason: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        """Enter explicitly permitted simulation or fail closed."""

        details = (
            reason
            if cause is None
            else (
                f"{reason}: "
                f"{type(cause).__name__}: {cause}"
            )
        )
        self._last_error = details

        if not self.allow_simulation:
            self._status = "failed"
            logger.error(
                "SensorBus cannot start physical transport: %s",
                details,
            )
            raise CommunicationError(
                "RaspberryPi",
                "Pico",
                reason,
            ) from cause

        logger.warning(
            "SensorBus entering explicit simulation mode: %s",
            details,
        )

        self._simulation = True
        self._status = "simulation"
        self._resolved_port = None

        self._thread = threading.Thread(
            target=self._sim_loop,
            name="RoboCarSensorSimulation",
            daemon=True,
        )
        self._thread.start()

        self.lighting.refresh(force=True)

    def _sim_loop(self) -> None:
        """Produce explicit deterministic simulation frames."""

        ticks = 0

        while not self._stop.is_set():
            ticks += 1

            command = self.lighting.last_command

            reading = SensorReading(
                t=time.time(),
                ultra_front_m=(
                    0.35
                    + 0.05 * (
                        1
                        if (ticks // 25) % 2 == 0
                        else -1
                    )
                ),
                ultra_rear_m=(
                    0.50
                    + 0.07 * (
                        1
                        if (ticks // 30) % 2 == 0
                        else -1
                    )
                ),
                tof_mm=600 + (ticks % 40),
                hall=(
                    1
                    if (ticks // 10) % 2 == 0
                    else 0
                ),
                encoder_ticks_total=ticks,
                imu_ax=0.01,
                imu_ay=-0.02,
                imu_az=9.78,
                imu_gx=0.1,
                imu_gy=0.2,
                imu_gz=0.3,
                mag_x=0.5,
                mag_y=0.0,
                mag_z=-0.2,
                led_front=(
                    int(command.headlights)
                    if command is not None
                    else 0
                ),
                led_rear=(
                    int(command.taillights)
                    if command is not None
                    else 0
                ),
                led_left_indicator=(
                    int(command.left_pattern != "off")
                    if command is not None
                    else 0
                ),
                led_right_indicator=(
                    int(command.right_pattern != "off")
                    if command is not None
                    else 0
                ),
                vbat=None,
            )

            self._publish(reading)
            self._stop.wait(0.05)

    def _autodetect_port(self) -> Optional[str]:
        """Return the first recognized Pico-compatible serial port."""

        try:
            from serial.tools import list_ports  # type: ignore
        except Exception as exc:
            self._last_error = (
                f"serial_port_enumeration_failed: {exc}"
            )
            return None

        candidates: list[str] = []

        for port in list_ports.comports():
            identity = (
                f"{port.device} "
                f"{port.description} "
                f"{port.hwid}"
            ).lower()

            if any(
                token in identity
                for token in (
                    "pico",
                    "usb serial",
                    "ch340",
                    "cp210",
                )
            ):
                candidates.append(
                    str(port.device)
                )

        return (
            candidates[0]
            if candidates
            else None
        )

    def _loop(self) -> None:
        """Read and publish physical Pico sensor frames."""

        serial_device = self._ser

        if serial_device is None:
            self._status = "failed"
            return

        consecutive_transport_errors = 0

        while not self._stop.is_set():
            try:
                raw = serial_device.readline()

                if not raw:
                    continue

                if isinstance(raw, bytes):
                    line = raw.decode(
                        "utf-8",
                        errors="strict",
                    ).strip()
                else:
                    line = str(raw).strip()

                if not line:
                    continue

                reading = self._parse_line(line)

                if reading is None:
                    self._parse_errors += 1
                    self._last_error = (
                        "unrecognized_or_invalid_sensor_frame"
                    )
                    logger.debug(
                        "Dropped unrecognized Pico frame: %r",
                        line[:240],
                    )
                    continue

                consecutive_transport_errors = 0
                self._status = "operational"
                self._publish(reading)

            except UnicodeDecodeError as exc:
                self._parse_errors += 1
                self._last_error = (
                    f"serial_decode_error: {exc}"
                )
                logger.warning(
                    "SensorBus dropped non-UTF8 serial frame: %s",
                    exc,
                )

            except Exception as exc:
                consecutive_transport_errors += 1
                self._transport_errors += 1
                self._last_error = (
                    "serial_read_error: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._status = "degraded"

                logger.error(
                    "SensorBus serial read failed: %s",
                    exc,
                )

                # A live hardware failure never silently becomes simulation.
                self._stop.wait(
                    min(
                        0.25,
                        0.02 * consecutive_transport_errors,
                    )
                )

    def _write_lighting_command(
        self,
        command: LightingCommand,
    ) -> bool:
        """Write one semantic lighting command without writer interleaving."""

        if not isinstance(command, LightingCommand):
            raise TypeError(
                "command must be LightingCommand"
            )

        if self._simulation:
            self._commands_sent += 1
            return True

        serial_device = self._ser

        if serial_device is None:
            self._command_errors += 1
            self._last_error = (
                "lighting_command_without_serial_transport"
            )
            return False

        encoded = (
            json.dumps(
                command.to_payload(),
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

        try:
            with self._write_lock:
                written = serial_device.write(encoded)

            if (
                written is not None
                and int(written) != len(encoded)
            ):
                raise IOError(
                    "partial serial write: "
                    f"expected={len(encoded)} "
                    f"written={written}"
                )

            self._commands_sent += 1
            return True

        except Exception as exc:
            self._command_errors += 1
            self._last_error = (
                "lighting_serial_write_error: "
                f"{type(exc).__name__}: {exc}"
            )
            self._status = "degraded"

            logger.error(
                "SensorBus lighting command write failed: %s",
                exc,
            )
            return False

    def _publish(
        self,
        reading: SensorReading,
    ) -> None:
        """Publish one validated sensor frame."""

        if not isinstance(reading, SensorReading):
            raise TypeError(
                "reading must be SensorReading"
            )

        # Resolve obstacle lighting before subscribers consume the frame.
        # A lighting-delivery failure does not erase valid sensor data.
        self.lighting.observe(reading)

        self._latest = reading
        self._frames_received += 1
        self._last_frame_monotonic = time.monotonic()

        with self._subscriber_lock:
            subscribers = tuple(
                self._subscribers
            )

        for callback in subscribers:
            try:
                callback(reading)
            except Exception as exc:
                self._callback_errors += 1
                self._last_error = (
                    "subscriber_error: "
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception(
                    "SensorBus subscriber failed"
                )

        if bounded_queue_put(
            self._q,
            reading,
        ):
            self._dropped_frames += 1

    @staticmethod
    def _parse_line(
        line: str,
    ) -> Optional[SensorReading]:
        """Normalize one JSON or key/value Pico sensor frame."""

        payload = decode_serial_payload(line)

        if not payload:
            return None

        def get(*names: str) -> Any:
            return get_case_insensitive(
                payload,
                *names,
            )

        reading = SensorReading(
            t=time.time(),
            ultra_front_m=optional_finite_float(
                get(
                    "ultra_front_m",
                    "ULTRA_FRONT",
                ),
                minimum=0.0,
            ),
            ultra_rear_m=optional_finite_float(
                get(
                    "ultra_rear_m",
                    "ULTRA_REAR",
                ),
                minimum=0.0,
            ),
            tof_mm=optional_int(
                get(
                    "tof_mm",
                    "TOF",
                ),
                minimum=0,
            ),
            hall=optional_binary(
                get(
                    "hall",
                    "HALL",
                )
            ),
            encoder_ticks_total=optional_int(
                get(
                    "encoder_ticks_total",
                    "ENCODER_TICKS",
                    "TICKS_TOTAL",
                ),
                minimum=0,
            ),
            imu_ax=optional_finite_float(
                get(
                    "imu_ax",
                    "IMU_AX",
                )
            ),
            imu_ay=optional_finite_float(
                get(
                    "imu_ay",
                    "IMU_AY",
                )
            ),
            imu_az=optional_finite_float(
                get(
                    "imu_az",
                    "IMU_AZ",
                )
            ),
            imu_gx=optional_finite_float(
                get(
                    "imu_gx",
                    "IMU_GX",
                )
            ),
            imu_gy=optional_finite_float(
                get(
                    "imu_gy",
                    "IMU_GY",
                )
            ),
            imu_gz=optional_finite_float(
                get(
                    "imu_gz",
                    "IMU_GZ",
                )
            ),
            mag_x=optional_finite_float(
                get(
                    "mag_x",
                    "MAG_X",
                )
            ),
            mag_y=optional_finite_float(
                get(
                    "mag_y",
                    "MAG_Y",
                )
            ),
            mag_z=optional_finite_float(
                get(
                    "mag_z",
                    "MAG_Z",
                )
            ),
            led_front=optional_binary(
                get(
                    "led_front",
                    "LED_FRONT",
                )
            ),
            led_rear=optional_binary(
                get(
                    "led_rear",
                    "LED_REAR",
                )
            ),
            led_left_indicator=optional_binary(
                get(
                    "led_left_indicator",
                    "LED_LEFT_INDICATOR",
                    "LED_LEFT",
                )
            ),
            led_right_indicator=optional_binary(
                get(
                    "led_right_indicator",
                    "LED_RIGHT_INDICATOR",
                    "LED_RIGHT",
                )
            ),
            vbat=optional_finite_float(
                get(
                    "vbat",
                    "VBAT",
                ),
                minimum=0.0,
            ),
        )

        values = reading.to_dict()

        if not any(
            value is not None
            for key, value in values.items()
            if key != "t"
        ):
            return None

        return reading


__all__ = [
    "PICO_I2C0_SDA",
    "PICO_I2C0_SCL",
    "PICO_I2C1_SDA",
    "PICO_I2C1_SCL",
    "PICO_US_FRONT_TRIG",
    "PICO_US_FRONT_ECHO",
    "PICO_US_REAR_TRIG",
    "PICO_US_REAR_ECHO",
    "PICO_HALL",
    "PICO_LED_FRONT",
    "PICO_LED_REAR",
    "PICO_LED_LEFT_INDICATOR",
    "PICO_LED_RIGHT_INDICATOR",
    "I2C_ADDR",
    "PICO_PINMAP",
    "pico_pinmap_json",
    "LightingMode",
    "LightingCommand",
    "VehicleLightingController",
    "SensorReading",
    "SensorBus",
]


# =============================================================================
# Comprehensive deterministic test block
# =============================================================================


if __name__ == "__main__":
    print("\n=== Running Sensor and Lighting Gateway Tests ===\n")
    printer.status(
        "TEST",
        "SensorBus lighting policy initialized",
        "info",
    )

    class _TestClock:
        """Deterministic monotonic clock for boundary testing."""

        def __init__(self) -> None:
            self.value = 100.0

        def __call__(self) -> float:
            return self.value

    class _FakeSerial:
        """Minimal serial writer used to validate command framing."""

        def __init__(self) -> None:
            self.writes: list[bytes] = []

        def write(self, payload: bytes) -> int:
            self.writes.append(payload)
            return len(payload)

        def close(self) -> None:
            return None

    # ------------------------------------------------------------------
    # Pin-map contract
    # ------------------------------------------------------------------

    assert PICO_LED_FRONT == 16
    assert PICO_LED_REAR == 18
    assert PICO_LED_LEFT_INDICATOR == 2
    assert PICO_LED_RIGHT_INDICATOR == 22
    assert "signal" not in PICO_PINMAP["led"]

    assert PICO_PINMAP["led"] == {
        "headlights": 16,
        "taillights": 18,
        "left_indicator": 2,
        "right_indicator": 22,
    }

    printer.status(
        "TEST",
        "four-channel lighting pin map",
        "success",
    )

    # ------------------------------------------------------------------
    # Driving-light intent
    # ------------------------------------------------------------------

    test_clock = _TestClock()
    emitted: list[LightingCommand] = []

    controller = VehicleLightingController(
        lambda command: emitted.append(command),
        clock=test_clock,
    )

    drive_command = controller.set_drive_intent(True)

    assert drive_command.mode == LightingMode.DRIVE
    assert drive_command.headlights is True
    assert drive_command.taillights is True
    assert drive_command.left_pattern == "off"
    assert drive_command.right_pattern == "off"

    printer.status(
        "TEST",
        "driving and continuing-drive lighting",
        "success",
    )

    # ------------------------------------------------------------------
    # Obstacle warning boundaries
    # ------------------------------------------------------------------

    obstacle_50_cm = controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=0.50,
            ultra_rear_m=2.0,
            tof_mm=900,
        )
    )

    assert obstacle_50_cm.mode == LightingMode.OBSTACLE
    assert obstacle_50_cm.headlights is True
    assert obstacle_50_cm.taillights is True
    assert obstacle_50_cm.left_pattern == "blink"
    assert obstacle_50_cm.right_pattern == "blink"
    assert obstacle_50_cm.frequency_hz_low == 2.0
    assert obstacle_50_cm.frequency_hz_high == 3.0

    obstacle_05_cm = controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=0.40,
            ultra_rear_m=1.0,
            tof_mm=50,
        )
    )

    assert obstacle_05_cm.mode == LightingMode.OBSTACLE
    assert obstacle_05_cm.frequency_hz_low == 12.0
    assert obstacle_05_cm.frequency_hz_high == 18.0

    obstacle_below_05_cm = controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=0.04,
            ultra_rear_m=1.0,
            tof_mm=100,
        )
    )

    assert obstacle_below_05_cm.frequency_hz_low == 12.0
    assert obstacle_below_05_cm.frequency_hz_high == 18.0

    printer.status(
        "TEST",
        "distance-proportional obstacle warning",
        "success",
    )

    # ------------------------------------------------------------------
    # Sensor fusion selects the nearest valid obstacle
    # ------------------------------------------------------------------

    nearest_from_rear = controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=0.40,
            ultra_rear_m=0.10,
            tof_mm=300,
        )
    )

    assert nearest_from_rear.obstacle_distance_m == 0.10

    nearest_from_tof = controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=0.40,
            ultra_rear_m=0.30,
            tof_mm=75,
        )
    )

    assert nearest_from_tof.obstacle_distance_m == 0.075

    printer.status(
        "TEST",
        "nearest ultrasonic/ToF obstacle selection",
        "success",
    )

    # ------------------------------------------------------------------
    # Turn boundary
    # ------------------------------------------------------------------

    controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=2.0,
            ultra_rear_m=2.0,
            tof_mm=2000,
        )
    )

    before_left_turn = controller.set_turn_intent(
        "left",
        1.001,
    )
    assert before_left_turn.mode == LightingMode.DRIVE

    left_turn = controller.set_turn_intent(
        "left",
        1.0,
    )
    assert left_turn.mode == LightingMode.TURN_LEFT
    assert left_turn.headlights is True
    assert left_turn.taillights is True
    assert left_turn.left_pattern == "blink"
    assert left_turn.right_pattern == "off"
    assert left_turn.frequency_hz_low == 1.0
    assert left_turn.frequency_hz_high == 1.0

    right_turn = controller.set_turn_intent(
        "right",
        0.75,
    )
    assert right_turn.mode == LightingMode.TURN_RIGHT
    assert right_turn.left_pattern == "off"
    assert right_turn.right_pattern == "blink"
    assert right_turn.frequency_hz_low == 1.0
    assert right_turn.frequency_hz_high == 1.0

    printer.status(
        "TEST",
        "one-metre left/right turn activation",
        "success",
    )

    # ------------------------------------------------------------------
    # Obstacle warning overrides a turn
    # ------------------------------------------------------------------

    controller.set_turn_intent(
        "left",
        0.50,
    )

    obstacle_during_turn = controller.observe(
        SensorReading(
            t=time.time(),
            ultra_front_m=0.20,
            ultra_rear_m=1.0,
            tof_mm=500,
        )
    )

    assert obstacle_during_turn.mode == LightingMode.OBSTACLE
    assert obstacle_during_turn.left_pattern == "blink"
    assert obstacle_during_turn.right_pattern == "blink"

    printer.status(
        "TEST",
        "obstacle-over-turn priority",
        "success",
    )

    # ------------------------------------------------------------------
    # Parking pulse and exact expiry
    # ------------------------------------------------------------------

    parked = controller.park()

    assert parked.mode == LightingMode.PARKING_PULSE
    assert parked.headlights is False
    assert parked.taillights is False
    assert parked.left_pattern == "pulse"
    assert parked.right_pattern == "pulse"
    assert parked.frequency_hz_low == 1.0
    assert parked.frequency_hz_high == 1.0
    assert parked.duration_s == 30.0
    assert parked.terminal_mode == LightingMode.OFF

    test_clock.value = 129.999
    still_parked = controller.refresh()
    assert still_parked.mode == LightingMode.PARKING_PULSE

    test_clock.value = 130.0
    parking_expired = controller.refresh()
    assert parking_expired.mode == LightingMode.OFF
    assert parking_expired.headlights is False
    assert parking_expired.taillights is False
    assert parking_expired.left_pattern == "off"
    assert parking_expired.right_pattern == "off"

    printer.status(
        "TEST",
        "finite parking pulse and automatic expiry",
        "success",
    )

    # ------------------------------------------------------------------
    # Independent left/right telemetry normalization
    # ------------------------------------------------------------------

    parsed = SensorBus._parse_line(
        (
            '{"ULTRA_FRONT":0.4,'
            '"TOF":350,'
            '"LED_FRONT":1,'
            '"LED_REAR":1,'
            '"LED_LEFT":1,'
            '"LED_RIGHT":0}'
        )
    )

    assert parsed is not None
    assert parsed.ultra_front_m == 0.4
    assert parsed.tof_mm == 350
    assert parsed.led_front == 1
    assert parsed.led_rear == 1
    assert parsed.led_left_indicator == 1
    assert parsed.led_right_indicator == 0

    printer.status(
        "TEST",
        "four-channel telemetry normalization",
        "success",
    )

    # ------------------------------------------------------------------
    # Serial JSON command framing
    # ------------------------------------------------------------------

    serial_bus = SensorBus(
        port="/dev/test-pico",
        allow_simulation=False,
    )
    fake_serial = _FakeSerial()

    serial_bus._ser = fake_serial
    serial_bus._status = "operational"

    sent_command = serial_bus.set_drive_intent(True)

    assert sent_command.mode == LightingMode.DRIVE
    assert len(fake_serial.writes) == 1
    assert fake_serial.writes[0].endswith(b"\n")

    decoded_command = json.loads(
        fake_serial.writes[0].decode("utf-8")
    )

    assert decoded_command["type"] == "robocar_command"
    assert decoded_command["command"] == "lighting"
    assert decoded_command["protocol_version"] == 1
    assert decoded_command["headlights"] == 1
    assert decoded_command["taillights"] == 1
    assert decoded_command["left_indicator"]["pattern"] == "off"
    assert decoded_command["right_indicator"]["pattern"] == "off"

    serial_bus.lighting.all_off(force=True)

    assert len(fake_serial.writes) == 2

    decoded_off = json.loads(
        fake_serial.writes[-1].decode("utf-8")
    )

    assert decoded_off["mode"] == "off"
    assert decoded_off["headlights"] == 0
    assert decoded_off["taillights"] == 0

    printer.status(
        "TEST",
        "newline-delimited lighting command framing",
        "success",
    )

    print("\n=== Tests ran successfully ===\n")