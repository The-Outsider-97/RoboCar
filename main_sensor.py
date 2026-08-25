"""
Reads Pico H serial, parses JSON/KEY:VALUE lines into SensorReading objects,
and exposes:
  - a non-blocking queue via poll_nowait()
  - a latest() snapshot
  - a subscribe(callback) stream

Hardware map (Pico H on Grove Shield v1.0):

I2C0  (GP4 SDA, GP5 SCL)  -> MPU6050 IMU
I2C1  (GP6 SDA, GP7 SCL)  -> Grove I2C Hub:
                              - VL53L0X (GY-530)
                              - TLV493D (3D magnetometer)

LEDs on Grove Shield digital ports (documented D16/D18/D20):
  D16 -> GP16  : Front LED (white)
  D18 -> GP18  : Rear LED  (red)
  D20 -> GP20  : Link/Signal LED (yellow)
See Seeed's Grove Shield docs/examples where D16 maps to Pin(16), etc.  # ref: wiki
"""

from __future__ import annotations

import json
import threading
import time
import queue
import sys

from dataclasses import dataclass
from typing import Callable, Optional

from .utils.rc_errors import *
from .utils.rc_helpers import *
# from system.hardware.ssd1306 import SSD1306_I2C, SSD1306_SPI
from logs.logger import get_logger, PrettyPrinter

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
    "i2c0": {"sda": PICO_I2C0_SDA, "scl": PICO_I2C0_SCL, "devices": ["MPU6050"]},
    "i2c1": {"sda": PICO_I2C1_SDA, "scl": PICO_I2C1_SCL, "hub": True, "devices": ["VL53L0X","TLV493D"]},
    "ultrasonic": {
        "mode": "gpio",  # "gpio" for HC-SR04P via TRIG/ECHO; set to "i2c" if using an I2C ultrasonic ranger
        "front": {"trig": PICO_US_FRONT_TRIG, "echo": PICO_US_FRONT_ECHO, "divider_top_ohm": 1800, "divider_bottom_ohm": 3300},
        "rear":  {"trig": PICO_US_REAR_TRIG,  "echo": PICO_US_REAR_ECHO,  "divider_top_ohm": 1800, "divider_bottom_ohm": 3300}
    },
    "hall": {"signal": PICO_HALL},
    "led": {"front": PICO_LED_FRONT, "rear": PICO_LED_REAR, "signal": PICO_LED_SIGNAL},
    "i2c_addr": I2C_ADDR,
    "notes": {
        "grove_shield_ports": {"D16": PICO_LED_FRONT, "D18": PICO_LED_REAR, "D20": PICO_LED_SIGNAL},
        "power_warning": "Use pin 36 (3V3) for sensor power. Pin 37 is 3V3_EN (not a supply)."
    }
}

def pico_pinmap_json() -> str:
    """JSON representation of the Pico pin map (for dashboards or to ship to FW)."""
    try:
        import json as _json
        return _json.dumps(PICO_PINMAP, separators=(',', ':'), sort_keys=True)
    except Exception:
        # very limited environments may not have json module; fall back
        return (
            '{"i2c0":{"sda":%d,"scl":%d},"i2c1":{"sda":%d,"scl":%d}}'
            % (PICO_I2C0_SDA, PICO_I2C0_SCL, PICO_I2C1_SDA, PICO_I2C1_SCL)
        )

# ============================================================================

@dataclass
class SensorReading:
    t: float                                 # host timestamp
    ultra_front_m: Optional[float] = None
    ultra_rear_m: Optional[float] = None
    tof_mm: Optional[int] = None
    hall: Optional[int] = None               # digital level
    imu_ax: Optional[float] = None           # m/s^2
    imu_ay: Optional[float] = None
    imu_az: Optional[float] = None
    imu_gx: Optional[float] = None           # deg/s
    imu_gy: Optional[float] = None
    imu_gz: Optional[float] = None
    mag_x: Optional[float] = None            # mT (scaled)
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None
    led_front: Optional[int] = None          # 0/1
    led_rear: Optional[int] = None           # 0/1
    led_signal: Optional[int] = None         # 0/1
    vbat: Optional[float] = None

class SensorBus:
    """
    Host-side gateway for the serial stream coming from the Pico.

    The Pico firmware can send either:
      - JSON objects per line, e.g.
        {"ultra_front_m":1.23, "ultra_rear_m":0.88, "tof_mm":557, "hall":1, "imu_ax":0.1, ...}
      - or comma-separated KEY:VALUE pairs, e.g.
        ULTRA_FRONT:1.23, ULTRA_REAR:0.88, TOF:557, HALL:1, IMU_AX:0.1

    This class:
      * buffers the stream in a non-blocking queue,
      * keeps a "latest()" snapshot,
      * lets you subscribe(callback) for push updates,
      * simulates data if no serial device is found (useful on dev laptops).
    """

    def __init__(self, port: Optional[str] = "auto", baud: int = 115200, qmax: int = 64):
        self.port = port
        self.baud = baud
        self._ser = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._q: "queue.Queue[SensorReading]" = queue.Queue(maxsize=qmax)
        self._latest: Optional[SensorReading] = None
        self._simulation = False
        self._subscribers: list[Callable[[SensorReading], None]] = []

    # ---- public API ---------------------------------------------------------

    def start(self):
        """Begin reading serial (or simulation if no port found)."""
        if self._thread and self._thread.is_alive():
            return
        try:
            import serial  # pyserial
        except Exception:
            serial = None

        if serial is None:
            # pyserial not installed -> simulation
            self._start_sim()
            return
        try:
            resolved = None
            if self.port in (None, "auto"):
                resolved = self._autodetect_port()
            else:
                resolved = self.port
            if not resolved:
                # no ports found -> simulation
                self._start_sim()
                return
            self._ser = serial.Serial(
                resolved, self.baud, timeout=1)
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception:
            # open failed -> simulation
            self._start_sim()

    def _start_sim(self):
        self._simulation = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._thread.start()
    
    def _sim_loop(self):
        ticks = 0
        while not self._stop.is_set():
            ticks += 1
            # make up some plausible values
            r = SensorReading(
                t=time.time(),
                ultra_front_m=0.35 + 0.05 * (1 if (ticks // 25) % 2 == 0 else -1),
                ultra_rear_m=0.50 + 0.07 * (1 if (ticks // 30) % 2 == 0 else -1),
                tof_mm=600 + (ticks % 40),
                hall=1 if (ticks // 10) % 2 == 0 else 0,
                imu_ax=0.01, imu_ay=-0.02, imu_az=9.78,
                imu_gx=0.1, imu_gy=0.2, imu_gz=0.3,
                mag_x=0.5, mag_y=0.0, mag_z=-0.2,
                led_front=(ticks // 20) % 2,
                led_rear=(ticks // 30) % 2,
                led_signal=(ticks // 10) % 2,
            )
            self._publish(r)
            time.sleep(0.05)

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self._thread = None
        try:
            if self._ser:
                self._ser.close()
        except Exception:
            pass
        self._ser = None

    def subscribe(self, callback: Callable[[SensorReading], None]):
        """Call callback(reading) for each parsed frame."""
        self._subscribers.append(callback)

    def latest(self) -> Optional[SensorReading]:
        """Return latest parsed frame (if any)."""
        return self._latest

    def poll_nowait(self) -> Optional[SensorReading]:
        """Pop one reading from the queue, if available (non-blocking)."""
        try:
            return self._q.get_nowait()
        except queue.Empty:
            return None

    # ---- internal -----------------------------------------------------------

    def _publish(self, r: SensorReading):
        self._latest = r
        for cb in self._subscribers:
            try:
                cb(r)
            except Exception:
                pass
        try:
            self._q.put_nowait(r)
        except queue.Full:
            # drop oldest to make room
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(r)

    def _autodetect_port(self) -> Optional[str]:
        try:
            from serial.tools import list_ports
        except Exception:
            return None
        candidates = []
        for p in list_ports.comports():
            # heuristic: look for Pico or USB serial adapters
            name = f"{p.device} {p.description} {p.hwid}".lower()
            if "pico" in name or "usb serial" in name or "ch340" in name or "cp210" in name:
                candidates.append(p.device)
        return candidates[0] if candidates else None

    def _loop(self):
        assert self._ser is not None
        ser = self._ser
        while not self._stop.is_set():
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                r = self._parse_line(line)
                if r:
                    self._publish(r)
            except Exception:
                time.sleep(0.02)

    @staticmethod
    def _parse_line(line: str) -> Optional[SensorReading]:
        # JSON first
        try:
            obj = json.loads(line)
            return SensorReading(
                t=time.time(),
                ultra_front_m=_to_float(obj.get("ultra_front_m")),
                ultra_rear_m=_to_float(obj.get("ultra_rear_m")),
                tof_mm=_to_int(obj.get("tof_mm")),
                hall=_to_int(obj.get("hall")),
                imu_ax=_to_float(obj.get("imu_ax")),
                imu_ay=_to_float(obj.get("imu_ay")),
                imu_az=_to_float(obj.get("imu_az")),
                imu_gx=_to_float(obj.get("imu_gx")),
                imu_gy=_to_float(obj.get("imu_gy")),
                imu_gz=_to_float(obj.get("imu_gz")),
                mag_x=_to_float(obj.get("mag_x")),
                mag_y=_to_float(obj.get("mag_y")),
                mag_z=_to_float(obj.get("mag_z")),
                led_front=_to_int(obj.get("led_front")),
                led_rear=_to_int(obj.get("led_rear")),
                led_signal=_to_int(obj.get("led_signal")),
                vbat=_to_float(obj.get("vbat")),
            )
        except Exception:
            pass
        # KEY:VALUE fallback (comma-separated)
        kv = {}
        for part in line.split(","):
            part = part.strip()
            if ":" in part:
                k, v = part.split(":", 1)
                kv[k.strip().upper()] = v.strip()
        if kv:
            return SensorReading(
                ultra_front_m=_to_float(kv.get("ULTRA_FRONT")),
                ultra_rear_m=_to_float(kv.get("ULTRA_REAR")),
                tof_mm=_to_int(kv.get("TOF")),
                hall=_to_int(kv.get("HALL")),
                imu_ax=_to_float(kv.get("IMU_AX")),
                imu_ay=_to_float(kv.get("IMU_AY")),
                imu_az=_to_float(kv.get("IMU_AZ")),
                imu_gx=_to_float(kv.get("IMU_GX")),
                imu_gy=_to_float(kv.get("IMU_GY")),
                imu_gz=_to_float(kv.get("IMU_GZ")),
                mag_x=_to_float(kv.get("MAG_X")),
                mag_y=_to_float(kv.get("MAG_Y")),
                mag_z=_to_float(kv.get("MAG_Z")),
                led_front=_to_int(kv.get("LED_FRONT")),
                led_rear=_to_int(kv.get("LED_REAR")),
                led_signal=_to_int(kv.get("LED_SIGNAL")),
                vbat=_to_float(kv.get("VBAT")),
                t=time.time(),
            )
        return None


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


# === quick smoke test =========================================================
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
