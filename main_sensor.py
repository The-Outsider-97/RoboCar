"""Host-side Pico sensor gateway for RoboCar.

The Pico is the deterministic sensor/IO side of the vehicle.  This module owns
only the Raspberry-Pi-side serial transport and frame normalization.  It does
not re-implement the Pico's I2C/GPIO sensor drivers.

Supported line protocols are the repository's existing formats:

* one JSON object per line;
* comma-separated ``KEY:VALUE`` pairs.

Simulation is intentionally opt-in.  Missing pyserial, an unavailable Pico, or
an open/read failure must not silently replace physical measurements with
plausible synthetic values on a real vehicle.
"""

from __future__ import annotations

import queue
import threading
import time

from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

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
# - D16 -> front white LED      (GPIO16)
# - D18 -> rear  red  LED       (GPIO18)
# - D20 -> yellow link/status   (GPIO20)
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
PICO_LED_SIGNAL = 20  # D20

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
    "i2c0": {
        "sda": PICO_I2C0_SDA,
        "scl": PICO_I2C0_SCL,
        "devices": ["MPU6050"],
    },
    "i2c1": {
        "sda": PICO_I2C1_SDA,
        "scl": PICO_I2C1_SCL,
        "hub": True,
        "devices": ["VL53L0X", "TLV493D"],
    },
    "ultrasonic": {
        "mode": "gpio",
        "front": {
            "trig": PICO_US_FRONT_TRIG,
            "echo": PICO_US_FRONT_ECHO,
            "divider_top_ohm": 1800,
            "divider_bottom_ohm": 3300,
        },
        "rear": {
            "trig": PICO_US_REAR_TRIG,
            "echo": PICO_US_REAR_ECHO,
            "divider_top_ohm": 1800,
            "divider_bottom_ohm": 3300,
        },
    },
    "hall": {"signal": PICO_HALL},
    "led": {
        "front": PICO_LED_FRONT,
        "rear": PICO_LED_REAR,
        "signal": PICO_LED_SIGNAL,
    },
    "i2c_addr": I2C_ADDR,
    "notes": {
        "grove_shield_ports": {
            "D16": PICO_LED_FRONT,
            "D18": PICO_LED_REAR,
            "D20": PICO_LED_SIGNAL,
        },
        "power_warning": (
            "Use pin 36 (3V3) for sensor power. Pin 37 is 3V3_EN, not a supply."
        ),
    },
}


def pico_pinmap_json() -> str:
    """Return the hardware map in deterministic JSON form."""

    import json

    return json.dumps(PICO_PINMAP, separators=(",", ":"), sort_keys=True)


@dataclass(slots=True)
class SensorReading:
    """One normalized host-side sensor frame.

    ``t`` is the Raspberry Pi wall-clock receipt timestamp.  ``encoder_ticks_total``
    is optional and is only populated if the Pico firmware sends an accumulated
    tick counter; the digital ``hall`` level is retained separately and must not
    be treated as a complete wheel-encoder count on the host.
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
    led_signal: Optional[int] = None
    vbat: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SensorBus:
    """Threaded serial gateway for Pico sensor frames.

    Parameters
    ----------
    port:
        Device path, or ``"auto"``/``None`` for USB serial autodetection.
    baud:
        Serial baud rate.
    qmax:
        Maximum number of frames kept for polling.  The newest frame always wins
        when producers outrun consumers.
    allow_simulation:
        Explicit permission to synthesize frames when physical serial transport
        cannot be started.  This defaults to ``False`` for fail-safe operation.
    """

    def __init__(
        self,
        port: Optional[str] = "auto",
        baud: int = 115200,
        qmax: int = 64,
        *,
        allow_simulation: bool = False,
    ) -> None:
        self.port = port
        self.baud = require_int(baud, "sensor_bus.baud", minimum=1)
        queue_size = require_int(qmax, "sensor_bus.qmax", minimum=1)
        self.allow_simulation = bool(allow_simulation)

        self._ser: Any = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._q: "queue.Queue[SensorReading]" = queue.Queue(maxsize=queue_size)
        self._latest: Optional[SensorReading] = None
        self._simulation = False
        self._subscribers: list[Callable[[SensorReading], None]] = []
        self._subscriber_lock = threading.RLock()

        self._status = "stopped"
        self._resolved_port: Optional[str] = None
        self._last_error: Optional[str] = None
        self._frames_received = 0
        self._parse_errors = 0
        self._transport_errors = 0
        self._callback_errors = 0
        self._dropped_frames = 0
        self._last_frame_monotonic: Optional[float] = None

    @property
    def is_simulation(self) -> bool:
        return self._simulation

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

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
            self._start_simulation_or_raise("pyserial_unavailable", exc)
            return

        resolved = self._autodetect_port() if self.port in (None, "auto") else self.port
        if not resolved:
            self._start_simulation_or_raise("serial_port_not_found")
            return

        try:
            self._ser = serial.Serial(resolved, self.baud, timeout=1)
        except Exception as exc:
            self._start_simulation_or_raise("serial_open_failed", exc)
            return

        self._resolved_port = str(resolved)
        self._status = "operational"
        self._thread = threading.Thread(
            target=self._loop,
            name="RoboCarSensorBus",
            daemon=True,
        )
        self._thread.start()
        logger.info("SensorBus started on %s at %s baud", resolved, self.baud)

    def stop(self) -> None:
        """Stop acquisition and close the serial device."""

        self._stop.set()
        thread = self._thread
        if thread and thread is not threading.current_thread():
            thread.join(timeout=1.5)
        self._thread = None
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception as exc:
                logger.warning("SensorBus serial close failed: %s", exc)
        self._ser = None
        self._status = "stopped"

    close = stop

    def subscribe(self, callback: Callable[[SensorReading], None]) -> Callable[[SensorReading], None]:
        """Register a push callback and return it for convenient unsubscription."""

        if not callable(callback):
            raise TypeError("callback must be callable")
        with self._subscriber_lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)
        return callback

    def unsubscribe(self, callback: Callable[[SensorReading], None]) -> bool:
        with self._subscriber_lock:
            try:
                self._subscribers.remove(callback)
                return True
            except ValueError:
                return False

    def latest(self) -> Optional[SensorReading]:
        return self._latest

    def poll_nowait(self) -> Optional[SensorReading]:
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    def health(self) -> dict[str, Any]:
        age = None
        if self._last_frame_monotonic is not None:
            age = max(0.0, time.monotonic() - self._last_frame_monotonic)
        return {
            "status": self._status,
            "running": self.running,
            "mode": "simulation" if self._simulation else "hardware",
            "simulation_allowed": self.allow_simulation,
            "port_requested": self.port,
            "port_resolved": self._resolved_port,
            "baud": self.baud,
            "frames_received": self._frames_received,
            "parse_errors": self._parse_errors,
            "transport_errors": self._transport_errors,
            "callback_errors": self._callback_errors,
            "dropped_frames": self._dropped_frames,
            "last_frame_age_s": age,
            "last_error": self._last_error,
        }

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------
    def _start_simulation_or_raise(
        self,
        reason: str,
        cause: Optional[BaseException] = None,
    ) -> None:
        details = reason if cause is None else f"{reason}: {type(cause).__name__}: {cause}"
        self._last_error = details
        if not self.allow_simulation:
            self._status = "failed"
            logger.error("SensorBus cannot start physical transport: %s", details)
            raise CommunicationError("RaspberryPi", "Pico", reason) from cause

        logger.warning("SensorBus entering explicit simulation mode: %s", details)
        self._simulation = True
        self._status = "simulation"
        self._resolved_port = None
        self._thread = threading.Thread(
            target=self._sim_loop,
            name="RoboCarSensorSimulation",
            daemon=True,
        )
        self._thread.start()

    def _sim_loop(self) -> None:
        ticks = 0
        while not self._stop.is_set():
            ticks += 1
            reading = SensorReading(
                t=time.time(),
                ultra_front_m=0.35 + 0.05 * (1 if (ticks // 25) % 2 == 0 else -1),
                ultra_rear_m=0.50 + 0.07 * (1 if (ticks // 30) % 2 == 0 else -1),
                tof_mm=600 + (ticks % 40),
                hall=1 if (ticks // 10) % 2 == 0 else 0,
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
                led_front=(ticks // 20) % 2,
                led_rear=(ticks // 30) % 2,
                led_signal=(ticks // 10) % 2,
                vbat=None,
            )
            self._publish(reading)
            self._stop.wait(0.05)

    def _autodetect_port(self) -> Optional[str]:
        try:
            from serial.tools import list_ports  # type: ignore
        except Exception as exc:
            self._last_error = f"serial_port_enumeration_failed: {exc}"
            return None

        candidates: list[str] = []
        for port in list_ports.comports():
            name = f"{port.device} {port.description} {port.hwid}".lower()
            if any(token in name for token in ("pico", "usb serial", "ch340", "cp210")):
                candidates.append(str(port.device))
        return candidates[0] if candidates else None

    def _loop(self) -> None:
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
                    line = raw.decode("utf-8", errors="strict").strip()
                else:
                    line = str(raw).strip()
                if not line:
                    continue

                reading = self._parse_line(line)
                if reading is None:
                    self._parse_errors += 1
                    self._last_error = "unrecognized_or_invalid_sensor_frame"
                    logger.debug("Dropped unrecognized Pico frame: %r", line[:240])
                    continue

                consecutive_transport_errors = 0
                self._status = "operational"
                self._publish(reading)
            except UnicodeDecodeError as exc:
                self._parse_errors += 1
                self._last_error = f"serial_decode_error: {exc}"
                logger.warning("SensorBus dropped non-UTF8 serial frame: %s", exc)
            except Exception as exc:
                consecutive_transport_errors += 1
                self._transport_errors += 1
                self._last_error = f"serial_read_error: {type(exc).__name__}: {exc}"
                self._status = "degraded"
                logger.error("SensorBus serial read failed: %s", exc)
                # Do not silently convert a live hardware failure into simulation.
                self._stop.wait(min(0.25, 0.02 * consecutive_transport_errors))

    def _publish(self, reading: SensorReading) -> None:
        self._latest = reading
        self._frames_received += 1
        self._last_frame_monotonic = time.monotonic()

        with self._subscriber_lock:
            subscribers = tuple(self._subscribers)
        for callback in subscribers:
            try:
                callback(reading)
            except Exception as exc:
                self._callback_errors += 1
                self._last_error = f"subscriber_error: {type(exc).__name__}: {exc}"
                logger.exception("SensorBus subscriber failed")

        if bounded_queue_put(self._q, reading):
            self._dropped_frames += 1

    @staticmethod
    def _parse_line(line: str) -> Optional[SensorReading]:
        payload = decode_serial_payload(line)
        if not payload:
            return None

        def get(*names: str) -> Any:
            return get_case_insensitive(payload, *names)

        # Physical measurements are finite and non-negative where appropriate.
        # Invalid individual values become None; the rest of a valid frame is
        # retained so one sensor cannot erase all other observations.
        reading = SensorReading(
            t=time.time(),
            ultra_front_m=optional_finite_float(
                get("ultra_front_m", "ULTRA_FRONT"), minimum=0.0
            ),
            ultra_rear_m=optional_finite_float(
                get("ultra_rear_m", "ULTRA_REAR"), minimum=0.0
            ),
            tof_mm=optional_int(get("tof_mm", "TOF"), minimum=0),
            hall=optional_binary(get("hall", "HALL")),
            encoder_ticks_total=optional_int(
                get("encoder_ticks_total", "ENCODER_TICKS", "TICKS_TOTAL"),
                minimum=0,
            ),
            imu_ax=optional_finite_float(get("imu_ax", "IMU_AX")),
            imu_ay=optional_finite_float(get("imu_ay", "IMU_AY")),
            imu_az=optional_finite_float(get("imu_az", "IMU_AZ")),
            imu_gx=optional_finite_float(get("imu_gx", "IMU_GX")),
            imu_gy=optional_finite_float(get("imu_gy", "IMU_GY")),
            imu_gz=optional_finite_float(get("imu_gz", "IMU_GZ")),
            mag_x=optional_finite_float(get("mag_x", "MAG_X")),
            mag_y=optional_finite_float(get("mag_y", "MAG_Y")),
            mag_z=optional_finite_float(get("mag_z", "MAG_Z")),
            led_front=optional_binary(get("led_front", "LED_FRONT")),
            led_rear=optional_binary(get("led_rear", "LED_REAR")),
            led_signal=optional_binary(get("led_signal", "LED_SIGNAL")),
            vbat=optional_finite_float(get("vbat", "VBAT"), minimum=0.0),
        )
        payload_values = reading.to_dict()
        if not any(
            value is not None
            for key, value in payload_values.items()
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
    "PICO_LED_SIGNAL",
    "I2C_ADDR",
    "PICO_PINMAP",
    "pico_pinmap_json",
    "SensorReading",
    "SensorBus",
]


if __name__ == "__main__":
    print("\n=== Running Sensor Bus ===\n")
    printer.status("TEST", "Initializing SensorBus", "info")
    # Quick manual test:
    bus = SensorBus(port="auto", baud=115200)  # adjust port/baud if needed
    try:
        bus.start()
        print("SensorBus running. Ctrl+C to stop.")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        bus.stop()
    print("\n=== SensorBus Completed ===\n")
