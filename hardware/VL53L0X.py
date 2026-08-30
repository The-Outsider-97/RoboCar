"""VL53L0X time-of-flight sensor driver for RoboCar.

This module controls the VL53L0X directly through its I2C registers. It does
not require the former ``vl53l0x_python`` shared library, ``ctypes``, or a
second SMBus implementation. The bundled ``machine.py`` adapter owns the bus
API, while RoboCar's shared validation, error, and logging modules provide the
same behaviour used by the rest of the vehicle stack.

The initialization and timing calculations follow ST's public VL53L0X API
sequence and the established Pololu/Adafruit register implementations.
Distances are reported in millimetres. I2C addresses are always 7-bit.

Expected project location::

    RoboCar/hardware/VL53L0X.py

Run the hardware test from the project root with::

    python -m RoboCar.hardware.VL53L0X
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import time

from ..modules.machine import I2C, Pin
from ..utils.rc_errors import ConfigurationError, HardwareError, SensorError
from ..utils.rc_helpers import require_finite_float, require_int
from logs.logger import PrettyPrinter, get_logger # pyright: ignore[reportMissingImports]

logger = get_logger("VL53L0X")
printer = PrettyPrinter()


class Vl53l0xAccuracyMode:
    """Backward-compatible ranging-profile constants."""

    GOOD = 0
    BETTER = 1
    BEST = 2
    LONG_RANGE = 3
    HIGH_SPEED = 4


class Vl53l0xDeviceMode:
    """Backward-compatible mode constants used by older RoboCar callers."""

    SINGLE_RANGING = 0
    CONTINUOUS_RANGING = 1
    SINGLE_HISTOGRAM = 2
    CONTINUOUS_TIMED_RANGING = 3
    SINGLE_ALS = 10
    GPIO_DRIVE = 20
    GPIO_OSC = 21


class Vl53l0xGpioAlarmType:
    """Values accepted by ``SYSTEM_INTERRUPT_CONFIG_GPIO``."""

    OFF = 0
    THRESHOLD_CROSSED_LOW = 1
    THRESHOLD_CROSSED_HIGH = 2
    THRESHOLD_CROSSED_OUT = 3
    NEW_MEASUREMENT_READY = 4


class Vl53l0xInterruptPolarity:
    LOW = 0
    HIGH = 1


@dataclass(frozen=True)
class RangeMeasurement:
    """One immutable VL53L0X measurement and its quality metadata."""

    distance_mm: int
    range_status: int
    status_text: str
    valid: bool
    monotonic_s: float

    @property
    def distance_m(self) -> float:
        """Distance in metres without discarding the millimetre source value."""

        return self.distance_mm / 1000.0


class VL53L0X:
    """Direct-register driver for one VL53L0X ToF sensor.

    Args:
        i2c_bus: ``machine.I2C`` bus identifier. RoboCar uses I2C1.
        i2c_address: Sensor's current 7-bit address; power-on default is 0x29.
        sda_pin: SDA pin identifier passed to ``machine.Pin``; RoboCar uses GP6.
        scl_pin: SCL pin identifier passed to ``machine.Pin``; RoboCar uses GP7.
        i2c_frequency_hz: Requested bus frequency, at most 400 kHz.
        io_timeout_s: Upper bound for every sensor polling operation.
        poll_interval_s: Delay between register polls to prevent busy-spinning.
        i2c: Optional injected ``machine.I2C``-compatible object. Injected
            buses are never closed by this class.
        auto_initialize: Initialize and calibrate during construction when true.
        strict_identity: Require all three documented identification bytes when
            true. When false, only the model ID (0xEE) is mandatory.
    """

    DEFAULT_I2C_BUS = 1
    DEFAULT_I2C_ADDRESS = 0x29
    DEFAULT_SDA_PIN = 6
    DEFAULT_SCL_PIN = 7
    DEFAULT_I2C_FREQUENCY_HZ = 400_000
    DEFAULT_IO_TIMEOUT_S = 0.5
    DEFAULT_POLL_INTERVAL_S = 0.001

    EXPECTED_MODEL_ID = 0xEE
    EXPECTED_MODULE_TYPE = 0xAA
    EXPECTED_REVISION_ID = 0x10
    MIN_TIMING_BUDGET_US = 20_000
    MAX_RANGE_MM = 8_190

    PROFILE_GOOD = "good"
    PROFILE_BETTER = "better"
    PROFILE_BEST = "best"
    PROFILE_LONG_RANGE = "long_range"
    PROFILE_HIGH_SPEED = "high_speed"

    _PROFILE_NAMES = {
        Vl53l0xAccuracyMode.GOOD: PROFILE_GOOD,
        Vl53l0xAccuracyMode.BETTER: PROFILE_BETTER,
        Vl53l0xAccuracyMode.BEST: PROFILE_BEST,
        Vl53l0xAccuracyMode.LONG_RANGE: PROFILE_LONG_RANGE,
        Vl53l0xAccuracyMode.HIGH_SPEED: PROFILE_HIGH_SPEED,
    }
    _PROFILE_SETTINGS = {
        PROFILE_GOOD: (0.25, 14, 10, 33_000),
        PROFILE_BETTER: (0.25, 14, 10, 66_000),
        PROFILE_BEST: (0.25, 14, 10, 200_000),
        PROFILE_LONG_RANGE: (0.10, 18, 14, 33_000),
        PROFILE_HIGH_SPEED: (0.25, 14, 10, 20_000),
    }

    SYSRANGE_START = 0x00
    SYSTEM_SEQUENCE_CONFIG = 0x01
    SYSTEM_INTERMEASUREMENT_PERIOD = 0x04
    SYSTEM_INTERRUPT_CONFIG_GPIO = 0x0A
    SYSTEM_INTERRUPT_CLEAR = 0x0B
    SYSTEM_THRESH_HIGH = 0x0C
    SYSTEM_THRESH_LOW = 0x0E
    RESULT_INTERRUPT_STATUS = 0x13
    RESULT_RANGE_STATUS = 0x14
    MSRC_CONFIG_CONTROL = 0x60
    FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT = 0x44
    PRE_RANGE_CONFIG_VCSEL_PERIOD = 0x50
    PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x51
    PRE_RANGE_CONFIG_VALID_PHASE_LOW = 0x56
    PRE_RANGE_CONFIG_VALID_PHASE_HIGH = 0x57
    MSRC_CONFIG_TIMEOUT_MACROP = 0x46
    FINAL_RANGE_CONFIG_VALID_PHASE_LOW = 0x47
    FINAL_RANGE_CONFIG_VALID_PHASE_HIGH = 0x48
    FINAL_RANGE_CONFIG_VCSEL_PERIOD = 0x70
    FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI = 0x71
    GLOBAL_CONFIG_VCSEL_WIDTH = 0x32
    GLOBAL_CONFIG_SPAD_ENABLES_REF_0 = 0xB0
    GLOBAL_CONFIG_REF_EN_START_SELECT = 0xB6
    DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD = 0x4E
    DYNAMIC_SPAD_REF_EN_START_OFFSET = 0x4F
    GPIO_HV_MUX_ACTIVE_HIGH = 0x84
    I2C_SLAVE_DEVICE_ADDRESS = 0x8A
    VHV_CONFIG_PAD_SCL_SDA_EXTSUP_HV = 0x89
    IDENTIFICATION_MODEL_ID = 0xC0
    IDENTIFICATION_MODULE_TYPE = 0xC1
    IDENTIFICATION_REVISION_ID = 0xC2
    OSC_CALIBRATE_VAL = 0xF8

    VCSEL_PERIOD_PRE_RANGE = 0
    VCSEL_PERIOD_FINAL_RANGE = 1

    RANGE_STATUS_TEXT = {
        0: "range_valid",
        1: "sigma_fail",
        2: "signal_fail",
        3: "minimum_range_fail",
        4: "phase_fail",
        5: "hardware_fail",
        6: "no_update",
        7: "wrapped_target_fail",
        8: "processing_fail",
        9: "crosstalk_signal_fail",
        10: "synchronization_interrupt",
        11: "merged_pulse",
        12: "target_present_insufficient_signal",
        13: "minimum_range_clipped",
        14: "range_complete",
        15: "unknown_error",
    }

    # Mandatory private tuning sequence distributed with ST's reference API.
    _TUNING_SETTINGS = (
        (0xFF, 0x01), (0x00, 0x00), (0xFF, 0x00), (0x09, 0x00),
        (0x10, 0x00), (0x11, 0x00), (0x24, 0x01), (0x25, 0xFF),
        (0x75, 0x00), (0xFF, 0x01), (0x4E, 0x2C), (0x48, 0x00),
        (0x30, 0x20), (0xFF, 0x00), (0x30, 0x09), (0x54, 0x00),
        (0x31, 0x04), (0x32, 0x03), (0x40, 0x83), (0x46, 0x25),
        (0x60, 0x00), (0x27, 0x00), (0x50, 0x06), (0x51, 0x00),
        (0x52, 0x96), (0x56, 0x08), (0x57, 0x30), (0x61, 0x00),
        (0x62, 0x00), (0x64, 0x00), (0x65, 0x00), (0x66, 0xA0),
        (0xFF, 0x01), (0x22, 0x32), (0x47, 0x14), (0x49, 0xFF),
        (0x4A, 0x00), (0xFF, 0x00), (0x7A, 0x0A), (0x7B, 0x00),
        (0x78, 0x21), (0xFF, 0x01), (0x23, 0x34), (0x42, 0x00),
        (0x44, 0xFF), (0x45, 0x26), (0x46, 0x05), (0x40, 0x40),
        (0x0E, 0x06), (0x20, 0x1A), (0x43, 0x40), (0xFF, 0x00),
        (0x34, 0x03), (0x35, 0x44), (0xFF, 0x01), (0x31, 0x04),
        (0x4B, 0x09), (0x4C, 0x05), (0x4D, 0x04), (0xFF, 0x00),
        (0x44, 0x00), (0x45, 0x20), (0x47, 0x08), (0x48, 0x28),
        (0x67, 0x00), (0x70, 0x04), (0x71, 0x01), (0x72, 0xFE),
        (0x76, 0x00), (0x77, 0x00), (0xFF, 0x01), (0x0D, 0x01),
        (0xFF, 0x00), (0x80, 0x01), (0x01, 0xF8), (0xFF, 0x01),
        (0x8E, 0x01), (0x00, 0x01), (0xFF, 0x00), (0x80, 0x00),
    )

    def __init__(
        self,
        i2c_bus: int = DEFAULT_I2C_BUS,
        i2c_address: int = DEFAULT_I2C_ADDRESS,
        *,
        sda_pin: int = DEFAULT_SDA_PIN,
        scl_pin: int = DEFAULT_SCL_PIN,
        i2c_frequency_hz: int = DEFAULT_I2C_FREQUENCY_HZ,
        io_timeout_s: float = DEFAULT_IO_TIMEOUT_S,
        poll_interval_s: float = DEFAULT_POLL_INTERVAL_S,
        i2c: Any = None,
        auto_initialize: bool = False,
        strict_identity: bool = True,
        tca9548a_num: Optional[int] = None,
        tca9548a_addr: Optional[int] = None,
    ) -> None:
        # Multiplexing belongs in an injected I2C adapter. Accept legacy neutral
        # values so old construction still works, but reject an ignored mux.
        if tca9548a_num not in (None, 255) or tca9548a_addr not in (None, 0):
            raise ConfigurationError(
                parameter="VL53L0X.tca9548a",
                value=(tca9548a_num, tca9548a_addr),
                valid_range="inject a mux-aware machine.I2C-compatible bus",
            )

        self._i2c_bus = require_int(i2c_bus, "VL53L0X.i2c_bus", minimum=0)
        self._address = self._validate_address(i2c_address, "VL53L0X.i2c_address")
        self._sda_pin = require_int(sda_pin, "VL53L0X.sda_pin", minimum=0)
        self._scl_pin = require_int(scl_pin, "VL53L0X.scl_pin", minimum=0)
        self._i2c_frequency_hz = require_int(
            i2c_frequency_hz,
            "VL53L0X.i2c_frequency_hz",
            minimum=10_000,
            maximum=400_000,
        )
        self._io_timeout_s = require_finite_float(
            io_timeout_s,
            "VL53L0X.io_timeout_s",
            minimum=0.001,
            maximum=60.0,
        )
        self._poll_interval_s = require_finite_float(
            poll_interval_s,
            "VL53L0X.poll_interval_s",
            minimum=0.0001,
            maximum=self._io_timeout_s,
        )

        self._i2c = i2c
        self._owns_i2c = i2c is None
        self._strict_identity = bool(strict_identity)
        self._initialized = False
        self._continuous_mode = False
        self._continuous_period_ms = 0
        self._stop_variable = 0
        self._profile = self.PROFILE_GOOD
        self._measurement_timing_budget_us = 0
        self._model_id: Optional[int] = None
        self._module_type: Optional[int] = None
        self._revision_id: Optional[int] = None
        self._last_measurement: Optional[RangeMeasurement] = None
        self._last_error: Optional[str] = None
        self._read_count = 0
        self._invalid_read_count = 0
        self._timeout_count = 0
        self._consecutive_errors = 0

        if auto_initialize:
            self.open()

    @staticmethod
    def _validate_address(value: Any, parameter: str) -> int:
        return require_int(value, parameter, minimum=0x08, maximum=0x77)

    @property
    def address(self) -> int:
        return self._address

    @property
    def i2c_address(self) -> int:
        """Legacy address attribute, now changed only through ``set_address``."""

        return self._address

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def is_continuous_mode(self) -> bool:
        return self._continuous_mode

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def last_measurement(self) -> Optional[RangeMeasurement]:
        return self._last_measurement

    @property
    def data_ready(self) -> bool:
        self._require_initialized("check_data_ready")
        return (self._read_u8(self.RESULT_INTERRUPT_STATUS) & 0x07) != 0

    def _ensure_bus(self) -> None:
        if self._i2c is None:
            try:
                self._i2c = I2C(
                    self._i2c_bus,
                    sda=Pin(self._sda_pin),
                    scl=Pin(self._scl_pin),
                    freq=self._i2c_frequency_hz,
                )
            except Exception as exc:
                self._record_error(exc)
                raise HardwareError(
                    device="VL53L0X",
                    operation="open_i2c",
                    error_details=(
                        "bus=%d sda=%d scl=%d address=0x%02X: %s: %s"
                        % (
                            self._i2c_bus,
                            self._sda_pin,
                            self._scl_pin,
                            self._address,
                            type(exc).__name__,
                            exc,
                        )
                    ),
                ) from exc

    def _record_error(self, exc: BaseException) -> None:
        self._last_error = "%s: %s" % (type(exc).__name__, exc)
        self._consecutive_errors += 1

    def _record_success(self) -> None:
        self._last_error = None
        self._consecutive_errors = 0

    def _release_owned_bus(self) -> None:
        """Close and forget the bus only when this driver created it."""

        if not self._owns_i2c or self._i2c is None:
            return
        try:
            close = getattr(self._i2c, "close", None)
            if callable(close):
                close()
        finally:
            self._i2c = None

    def _write_multi(self, register: int, data: bytes) -> None:
        self._ensure_bus()
        try:
            assert self._i2c is not None
            self._i2c.writeto_mem(self._address, register & 0xFF, bytes(data))
        except Exception as exc:
            self._record_error(exc)
            raise HardwareError(
                device="VL53L0X",
                operation="i2c_write",
                error_details="address=0x%02X register=0x%02X length=%d: %s: %s"
                % (self._address, register & 0xFF, len(data), type(exc).__name__, exc),
            ) from exc

    def _read_multi(self, register: int, length: int) -> bytes:
        self._ensure_bus()
        try:
            read_length = require_int(
                length, "VL53L0X.read_length", minimum=1, maximum=32
            )
            assert self._i2c is not None
            data = bytes(
                self._i2c.readfrom_mem(
                    self._address,
                    register & 0xFF,
                    read_length,
                )
            )
        except Exception as exc:
            self._record_error(exc)
            raise HardwareError(
                device="VL53L0X",
                operation="i2c_read",
                error_details="address=0x%02X register=0x%02X length=%d: %s: %s"
                % (self._address, register & 0xFF, length, type(exc).__name__, exc),
            ) from exc
        if len(data) != length:
            exc = IOError("expected %d bytes, received %d" % (length, len(data)))
            self._record_error(exc)
            raise HardwareError(
                device="VL53L0X",
                operation="i2c_read",
                error_details=str(exc),
            )
        return data

    def _write_u8(self, register: int, value: int) -> None:
        self._write_multi(register, bytes((value & 0xFF,)))

    def _write_u16(self, register: int, value: int) -> None:
        value &= 0xFFFF
        self._write_multi(register, bytes(((value >> 8) & 0xFF, value & 0xFF)))

    def _write_u32(self, register: int, value: int) -> None:
        value &= 0xFFFFFFFF
        self._write_multi(
            register,
            bytes(
                (
                    (value >> 24) & 0xFF,
                    (value >> 16) & 0xFF,
                    (value >> 8) & 0xFF,
                    value & 0xFF,
                )
            ),
        )

    def _read_u8(self, register: int) -> int:
        return self._read_multi(register, 1)[0]

    def _read_u16(self, register: int) -> int:
        data = self._read_multi(register, 2)
        return (data[0] << 8) | data[1]

    def _wait_for(
        self,
        predicate: Callable[[], bool],
        operation: str,
        *,
        timeout_s: Optional[float] = None,
    ) -> None:
        limit = self._io_timeout_s if timeout_s is None else require_finite_float(
            timeout_s,
            "VL53L0X.wait_timeout_s",
            minimum=0.001,
            maximum=60.0,
        )
        deadline = time.monotonic() + limit
        while not predicate():
            if time.monotonic() >= deadline:
                self._timeout_count += 1
                exc = TimeoutError("timed out after %.3f s" % limit)
                self._record_error(exc)
                raise HardwareError(
                    device="VL53L0X",
                    operation=operation,
                    error_details=str(exc),
                ) from exc
            time.sleep(self._poll_interval_s)

    def _require_initialized(self, operation: str) -> None:
        if not self._initialized:
            raise HardwareError(
                device="VL53L0X",
                operation=operation,
                error_details="sensor is not initialized; call open() first",
            )

    def open(self) -> "VL53L0X":
        """Open the bus and execute identity checks, setup, and calibration."""

        if self._initialized:
            return self
        self._ensure_bus()
        try:
            self._initialize_sensor()
        except Exception as exc:
            if self._last_error is None:
                self._record_error(exc)
            try:
                self._release_owned_bus()
            except Exception as close_exc:
                logger.error(
                    "VL53L0X bus cleanup after initialization failure failed: %s",
                    close_exc,
                )
            if not isinstance(exc, HardwareError):
                raise HardwareError(
                    device="VL53L0X",
                    operation="initialize",
                    error_details="%s: %s" % (type(exc).__name__, exc),
                ) from exc
            raise
        self._initialized = True
        self._record_success()
        logger.info(
            "VL53L0X initialized: bus=%d address=0x%02X model=0x%02X "
            "revision=0x%02X budget=%dus",
            self._i2c_bus,
            self._address,
            self._model_id,
            self._revision_id,
            self._measurement_timing_budget_us,
        )
        return self

    initialize = open

    def _initialize_sensor(self) -> None:
        self._wait_for(
            lambda: self._read_u8(self.IDENTIFICATION_MODEL_ID)
            == self.EXPECTED_MODEL_ID,
            "wait_for_boot",
        )
        self._model_id = self._read_u8(self.IDENTIFICATION_MODEL_ID)
        self._module_type = self._read_u8(self.IDENTIFICATION_MODULE_TYPE)
        self._revision_id = self._read_u8(self.IDENTIFICATION_REVISION_ID)
        expected = (
            self.EXPECTED_MODEL_ID,
            self.EXPECTED_MODULE_TYPE,
            self.EXPECTED_REVISION_ID,
        )
        actual = (self._model_id, self._module_type, self._revision_id)
        if self._model_id != self.EXPECTED_MODEL_ID or (
            self._strict_identity and actual != expected
        ):
            raise HardwareError(
                device="VL53L0X",
                operation="verify_identity",
                error_details="expected %r, received %r at address 0x%02X"
                % (expected, actual, self._address),
            )
        if actual != expected:
            logger.warning(
                "Non-standard VL53L0X identity bytes: expected=%r actual=%r",
                expected,
                actual,
            )

        self._write_u8(
            self.VHV_CONFIG_PAD_SCL_SDA_EXTSUP_HV,
            self._read_u8(self.VHV_CONFIG_PAD_SCL_SDA_EXTSUP_HV) | 0x01,
        )
        self._write_u8(0x88, 0x00)
        self._write_sequence(((0x80, 0x01), (0xFF, 0x01), (0x00, 0x00)))
        self._stop_variable = self._read_u8(0x91)
        self._write_sequence(((0x00, 0x01), (0xFF, 0x00), (0x80, 0x00)))

        self._write_u8(
            self.MSRC_CONFIG_CONTROL,
            self._read_u8(self.MSRC_CONFIG_CONTROL) | 0x12,
        )
        self.set_signal_rate_limit(0.25, require_initialized=False)
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, 0xFF)

        spad_count, spad_is_aperture = self._get_spad_info()
        ref_spad_map = bytearray(
            self._read_multi(self.GLOBAL_CONFIG_SPAD_ENABLES_REF_0, 6)
        )
        self._write_sequence(
            (
                (0xFF, 0x01),
                (self.DYNAMIC_SPAD_REF_EN_START_OFFSET, 0x00),
                (self.DYNAMIC_SPAD_NUM_REQUESTED_REF_SPAD, 0x2C),
                (0xFF, 0x00),
                (self.GLOBAL_CONFIG_REF_EN_START_SELECT, 0xB4),
            )
        )
        first_spad = 12 if spad_is_aperture else 0
        enabled = 0
        for index in range(48):
            byte_index = index // 8
            mask = 1 << (index % 8)
            if index < first_spad or enabled >= spad_count:
                ref_spad_map[byte_index] &= ~mask
            elif ref_spad_map[byte_index] & mask:
                enabled += 1
        self._write_multi(
            self.GLOBAL_CONFIG_SPAD_ENABLES_REF_0, bytes(ref_spad_map)
        )

        self._write_sequence(self._TUNING_SETTINGS)
        self._write_u8(self.SYSTEM_INTERRUPT_CONFIG_GPIO, 0x04)
        self._write_u8(
            self.GPIO_HV_MUX_ACTIVE_HIGH,
            self._read_u8(self.GPIO_HV_MUX_ACTIVE_HIGH) & ~0x10,
        )
        self.clear_interrupt(require_initialized=False)

        initial_budget = self._get_measurement_timing_budget_us()
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, 0xE8)
        self._set_measurement_timing_budget_us(initial_budget)
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, 0x01)
        self._perform_single_ref_calibration(0x40)
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, 0x02)
        self._perform_single_ref_calibration(0x00)
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, 0xE8)
        self._profile = self.PROFILE_GOOD

    def _write_sequence(self, pairs: Tuple[Tuple[int, int], ...]) -> None:
        for register, value in pairs:
            self._write_u8(register, value)

    def _get_spad_info(self) -> Tuple[int, bool]:
        self._write_sequence(
            ((0x80, 0x01), (0xFF, 0x01), (0x00, 0x00), (0xFF, 0x06))
        )
        self._write_u8(0x83, self._read_u8(0x83) | 0x04)
        self._write_sequence(
            ((0xFF, 0x07), (0x81, 0x01), (0x80, 0x01), (0x94, 0x6B), (0x83, 0x00))
        )
        self._wait_for(lambda: self._read_u8(0x83) != 0x00, "read_spad_info")
        self._write_u8(0x83, 0x01)
        value = self._read_u8(0x92)
        count = value & 0x7F
        is_aperture = ((value >> 7) & 0x01) == 1

        self._write_sequence(((0x81, 0x00), (0xFF, 0x06)))
        self._write_u8(0x83, self._read_u8(0x83) & ~0x04)
        self._write_sequence(
            ((0xFF, 0x01), (0x00, 0x01), (0xFF, 0x00), (0x80, 0x00))
        )
        if count <= 0 or count > 48:
            raise HardwareError(
                device="VL53L0X",
                operation="read_spad_info",
                error_details="invalid SPAD count %d" % count,
            )
        return count, is_aperture

    def _perform_single_ref_calibration(self, vhv_init_byte: int) -> None:
        self._write_u8(self.SYSRANGE_START, 0x01 | (vhv_init_byte & 0xFF))
        self._wait_for(
            lambda: (self._read_u8(self.RESULT_INTERRUPT_STATUS) & 0x07) != 0,
            "reference_calibration",
        )
        self.clear_interrupt(require_initialized=False)
        self._write_u8(self.SYSRANGE_START, 0x00)

    @staticmethod
    def _decode_timeout(value: int) -> int:
        return ((value & 0xFF) << ((value >> 8) & 0xFF)) + 1

    @staticmethod
    def _encode_timeout(timeout_mclks: int) -> int:
        timeout = max(0, int(timeout_mclks))
        if timeout == 0:
            return 0
        least = timeout - 1
        exponent = 0
        while least > 255:
            least >>= 1
            exponent += 1
        return ((exponent << 8) | (least & 0xFF)) & 0xFFFF

    @staticmethod
    def _timeout_mclks_to_microseconds(
        timeout_mclks: int, vcsel_period_pclks: int
    ) -> int:
        macro_period_ns = ((2304 * vcsel_period_pclks * 1655) + 500) // 1000
        return (
            (int(timeout_mclks) * macro_period_ns) + (macro_period_ns // 2)
        ) // 1000

    @staticmethod
    def _timeout_microseconds_to_mclks(
        timeout_us: int, vcsel_period_pclks: int
    ) -> int:
        macro_period_ns = ((2304 * vcsel_period_pclks * 1655) + 500) // 1000
        return (
            (int(timeout_us) * 1000) + (macro_period_ns // 2)
        ) // macro_period_ns

    @staticmethod
    def _encode_vcsel_period(period_pclks: int) -> int:
        return (int(period_pclks) >> 1) - 1

    @staticmethod
    def _decode_vcsel_period(register_value: int) -> int:
        return ((int(register_value) + 1) & 0xFF) << 1

    def get_vcsel_pulse_period(self, period_type: int) -> int:
        self._require_initialized("get_vcsel_pulse_period")
        return self._get_vcsel_pulse_period(period_type)

    def _get_vcsel_pulse_period(self, period_type: int) -> int:
        if period_type == self.VCSEL_PERIOD_PRE_RANGE:
            return self._decode_vcsel_period(
                self._read_u8(self.PRE_RANGE_CONFIG_VCSEL_PERIOD)
            )
        if period_type == self.VCSEL_PERIOD_FINAL_RANGE:
            return self._decode_vcsel_period(
                self._read_u8(self.FINAL_RANGE_CONFIG_VCSEL_PERIOD)
            )
        raise ConfigurationError(
            parameter="VL53L0X.vcsel_period_type",
            value=period_type,
            valid_range=(
                self.VCSEL_PERIOD_PRE_RANGE,
                self.VCSEL_PERIOD_FINAL_RANGE,
            ),
        )

    def _get_sequence_step_enables(self) -> Dict[str, bool]:
        sequence = self._read_u8(self.SYSTEM_SEQUENCE_CONFIG)
        return {
            "tcc": bool((sequence >> 4) & 0x01),
            "dss": bool((sequence >> 3) & 0x01),
            "msrc": bool((sequence >> 2) & 0x01),
            "pre_range": bool((sequence >> 6) & 0x01),
            "final_range": bool((sequence >> 7) & 0x01),
        }

    def _get_sequence_step_timeouts(
        self, pre_range_enabled: bool
    ) -> Dict[str, int]:
        pre_vcsel = self._get_vcsel_pulse_period(self.VCSEL_PERIOD_PRE_RANGE)
        msrc_mclks = self._read_u8(self.MSRC_CONFIG_TIMEOUT_MACROP) + 1
        msrc_us = self._timeout_mclks_to_microseconds(msrc_mclks, pre_vcsel)
        pre_mclks = self._decode_timeout(
            self._read_u16(self.PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI)
        )
        pre_us = self._timeout_mclks_to_microseconds(pre_mclks, pre_vcsel)

        final_vcsel = self._get_vcsel_pulse_period(self.VCSEL_PERIOD_FINAL_RANGE)
        final_mclks = self._decode_timeout(
            self._read_u16(self.FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI)
        )
        if pre_range_enabled:
            final_mclks = max(0, final_mclks - pre_mclks)
        final_us = self._timeout_mclks_to_microseconds(final_mclks, final_vcsel)
        return {
            "msrc_us": msrc_us,
            "pre_mclks": pre_mclks,
            "pre_us": pre_us,
            "final_us": final_us,
            "final_vcsel": final_vcsel,
        }

    def _get_measurement_timing_budget_us(self) -> int:
        enables = self._get_sequence_step_enables()
        timeouts = self._get_sequence_step_timeouts(enables["pre_range"])
        budget = 1910 + 960
        if enables["tcc"]:
            budget += timeouts["msrc_us"] + 590
        if enables["dss"]:
            budget += 2 * (timeouts["msrc_us"] + 690)
        elif enables["msrc"]:
            budget += timeouts["msrc_us"] + 660
        if enables["pre_range"]:
            budget += timeouts["pre_us"] + 660
        if enables["final_range"]:
            budget += timeouts["final_us"] + 550
        self._measurement_timing_budget_us = int(budget)
        return self._measurement_timing_budget_us

    @property
    def measurement_timing_budget_us(self) -> int:
        self._require_initialized("get_measurement_timing_budget")
        return self._get_measurement_timing_budget_us()

    @measurement_timing_budget_us.setter
    def measurement_timing_budget_us(self, budget_us: int) -> None:
        self.set_measurement_timing_budget_us(budget_us)

    def set_measurement_timing_budget_us(self, budget_us: int) -> int:
        self._require_initialized("set_measurement_timing_budget")
        return self._set_measurement_timing_budget_us(budget_us)

    def _set_measurement_timing_budget_us(self, budget_us: int) -> int:
        requested = require_int(
            budget_us,
            "VL53L0X.measurement_timing_budget_us",
            minimum=self.MIN_TIMING_BUDGET_US,
            maximum=1_000_000,
        )
        enables = self._get_sequence_step_enables()
        timeouts = self._get_sequence_step_timeouts(enables["pre_range"])
        used = 1320 + 960
        if enables["tcc"]:
            used += timeouts["msrc_us"] + 590
        if enables["dss"]:
            used += 2 * (timeouts["msrc_us"] + 690)
        elif enables["msrc"]:
            used += timeouts["msrc_us"] + 660
        if enables["pre_range"]:
            used += timeouts["pre_us"] + 660
        if not enables["final_range"]:
            raise HardwareError(
                device="VL53L0X",
                operation="set_measurement_timing_budget",
                error_details="final-range sequence step is disabled",
            )
        used += 550
        if used > requested:
            raise ConfigurationError(
                parameter="VL53L0X.measurement_timing_budget_us",
                value=requested,
                valid_range=(used, 1_000_000),
            )
        final_us = requested - used
        final_mclks = self._timeout_microseconds_to_mclks(
            final_us, timeouts["final_vcsel"]
        )
        if enables["pre_range"]:
            final_mclks += timeouts["pre_mclks"]
        self._write_u16(
            self.FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI,
            self._encode_timeout(final_mclks),
        )
        self._measurement_timing_budget_us = requested
        return requested

    @property
    def signal_rate_limit_mcps(self) -> float:
        self._require_initialized("get_signal_rate_limit")
        return (
            self._read_u16(self.FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT)
            / 128.0
        )

    def set_signal_rate_limit(
        self,
        limit_mcps: float,
        *,
        require_initialized: bool = True,
    ) -> float:
        if require_initialized:
            self._require_initialized("set_signal_rate_limit")
        limit = require_finite_float(
            limit_mcps,
            "VL53L0X.signal_rate_limit_mcps",
            minimum=0.0,
            maximum=511.99,
        )
        self._write_u16(
            self.FINAL_RANGE_CONFIG_MIN_COUNT_RATE_RTN_LIMIT,
            int(round(limit * 128.0)),
        )
        return limit

    def set_vcsel_pulse_period(
        self, period_type: int, period_pclks: int
    ) -> int:
        """Set a supported pre-range or final-range laser pulse period."""

        self._require_initialized("set_vcsel_pulse_period")
        period = require_int(
            period_pclks,
            "VL53L0X.vcsel_period_pclks",
            minimum=8,
            maximum=18,
        )
        if period_type == self.VCSEL_PERIOD_PRE_RANGE:
            phase_high = {12: 0x18, 14: 0x30, 16: 0x40, 18: 0x50}.get(period)
            if phase_high is None:
                raise ConfigurationError(
                    "VL53L0X.pre_range_vcsel_period",
                    period,
                    (12, 14, 16, 18),
                )
        elif period_type == self.VCSEL_PERIOD_FINAL_RANGE:
            final_settings = {
                8: (0x10, 0x08, 0x02, 0x0C, 0x30),
                10: (0x28, 0x08, 0x03, 0x09, 0x20),
                12: (0x38, 0x08, 0x03, 0x08, 0x20),
                14: (0x48, 0x08, 0x03, 0x07, 0x20),
            }.get(period)
            if final_settings is None:
                raise ConfigurationError(
                    "VL53L0X.final_range_vcsel_period",
                    period,
                    (8, 10, 12, 14),
                )
        else:
            raise ConfigurationError(
                "VL53L0X.vcsel_period_type",
                period_type,
                (
                    self.VCSEL_PERIOD_PRE_RANGE,
                    self.VCSEL_PERIOD_FINAL_RANGE,
                ),
            )

        if self._get_vcsel_pulse_period(period_type) == period:
            return period

        budget = (
            self._measurement_timing_budget_us
            or self._get_measurement_timing_budget_us()
        )
        enables = self._get_sequence_step_enables()
        timeouts = self._get_sequence_step_timeouts(enables["pre_range"])
        encoded = self._encode_vcsel_period(period)

        if period_type == self.VCSEL_PERIOD_PRE_RANGE:
            self._write_u8(self.PRE_RANGE_CONFIG_VALID_PHASE_HIGH, phase_high)
            self._write_u8(self.PRE_RANGE_CONFIG_VALID_PHASE_LOW, 0x08)
            self._write_u8(self.PRE_RANGE_CONFIG_VCSEL_PERIOD, encoded)
            pre_mclks = self._timeout_microseconds_to_mclks(
                timeouts["pre_us"], period
            )
            self._write_u16(
                self.PRE_RANGE_CONFIG_TIMEOUT_MACROP_HI,
                self._encode_timeout(pre_mclks),
            )
            msrc_mclks = self._timeout_microseconds_to_mclks(
                timeouts["msrc_us"], period
            )
            self._write_u8(
                self.MSRC_CONFIG_TIMEOUT_MACROP,
                min(255, max(0, msrc_mclks - 1)),
            )
        else:
            phase_high, phase_low, width, phasecal_timeout, phasecal_limit = (
                final_settings
            )
            self._write_u8(self.FINAL_RANGE_CONFIG_VALID_PHASE_HIGH, phase_high)
            self._write_u8(self.FINAL_RANGE_CONFIG_VALID_PHASE_LOW, phase_low)
            self._write_u8(self.GLOBAL_CONFIG_VCSEL_WIDTH, width)
            # ALGO_PHASECAL_CONFIG_TIMEOUT is register 0x30 on page 0;
            # ALGO_PHASECAL_LIM uses the same address on page 1.
            self._write_u8(0x30, phasecal_timeout)
            self._write_u8(0xFF, 0x01)
            self._write_u8(0x30, phasecal_limit)
            self._write_u8(0xFF, 0x00)
            self._write_u8(self.FINAL_RANGE_CONFIG_VCSEL_PERIOD, encoded)
            new_final_mclks = self._timeout_microseconds_to_mclks(
                timeouts["final_us"], period
            )
            if enables["pre_range"]:
                new_final_mclks += timeouts["pre_mclks"]
            self._write_u16(
                self.FINAL_RANGE_CONFIG_TIMEOUT_MACROP_HI,
                self._encode_timeout(new_final_mclks),
            )

        self._set_measurement_timing_budget_us(budget)
        sequence = self._read_u8(self.SYSTEM_SEQUENCE_CONFIG)
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, 0x02)
        self._perform_single_ref_calibration(0x00)
        self._write_u8(self.SYSTEM_SEQUENCE_CONFIG, sequence)
        return period

    def set_profile(self, profile: Any) -> str:
        """Apply one coherent speed/accuracy/range profile.

        Accepted names are ``good``, ``better``, ``best``, ``long_range``, and
        ``high_speed``. Legacy ``Vl53l0xAccuracyMode`` integers are accepted.
        """

        self._require_initialized("set_profile")
        if isinstance(profile, bool):
            normalized = None
        elif isinstance(profile, int):
            normalized = self._PROFILE_NAMES.get(profile)
        else:
            normalized = (
                str(profile)
                .strip()
                .lower()
                .replace("-", "_")
                .replace(" ", "_")
            )
        if normalized not in self._PROFILE_SETTINGS:
            raise ConfigurationError(
                parameter="VL53L0X.profile",
                value=profile,
                valid_range=tuple(self._PROFILE_SETTINGS),
            )
        if self._continuous_mode:
            raise HardwareError(
                device="VL53L0X",
                operation="set_profile",
                error_details="stop continuous ranging before changing profile",
            )
        signal_rate, pre_period, final_period, budget = self._PROFILE_SETTINGS[
            normalized
        ]
        self.set_signal_rate_limit(signal_rate)
        self.set_vcsel_pulse_period(
            self.VCSEL_PERIOD_PRE_RANGE, pre_period
        )
        self.set_vcsel_pulse_period(
            self.VCSEL_PERIOD_FINAL_RANGE, final_period
        )
        self.set_measurement_timing_budget_us(budget)
        self._profile = normalized
        logger.info(
            "VL53L0X profile=%s signal_rate=%.2fMCPS pre=%dPCLK "
            "final=%dPCLK budget=%dus",
            normalized,
            signal_rate,
            pre_period,
            final_period,
            budget,
        )
        return normalized

    def _restore_stop_variable(self) -> None:
        self._write_sequence(
            (
                (0x80, 0x01),
                (0xFF, 0x01),
                (0x00, 0x00),
                (0x91, self._stop_variable),
                (0x00, 0x01),
                (0xFF, 0x00),
                (0x80, 0x00),
            )
        )

    def start_continuous(self, period_ms: int = 0) -> None:
        """Start back-to-back (0 ms) or timed continuous measurements."""

        self._require_initialized("start_continuous")
        if self._continuous_mode:
            return
        period = require_int(
            period_ms,
            "VL53L0X.continuous_period_ms",
            minimum=0,
            maximum=60_000,
        )
        if period:
            minimum_period_ms = (
                self._measurement_timing_budget_us + 999
            ) // 1000
            if period < minimum_period_ms:
                raise ConfigurationError(
                    "VL53L0X.continuous_period_ms",
                    period,
                    (minimum_period_ms, 60_000),
                )
        self._restore_stop_variable()
        if period > 0:
            calibrated = period
            oscillator_calibration = self._read_u16(self.OSC_CALIBRATE_VAL)
            if oscillator_calibration:
                calibrated *= oscillator_calibration
            if calibrated > 0xFFFFFFFF:
                maximum = (
                    0xFFFFFFFF // oscillator_calibration
                    if oscillator_calibration
                    else 60_000
                )
                raise ConfigurationError(
                    "VL53L0X.continuous_period_ms", period, (1, maximum)
                )
            self._write_u32(self.SYSTEM_INTERMEASUREMENT_PERIOD, calibrated)
            self._write_u8(self.SYSRANGE_START, 0x04)
        else:
            self._write_u8(self.SYSRANGE_START, 0x02)
        self._continuous_mode = True
        self._continuous_period_ms = period
        logger.info("VL53L0X continuous ranging started: period_ms=%d", period)

    def stop_continuous(self) -> None:
        """Stop continuous ranging without initiating another measurement."""

        if not self._initialized or not self._continuous_mode:
            return
        self._write_u8(self.SYSRANGE_START, 0x01)
        self._write_sequence(
            (
                (0xFF, 0x01),
                (0x00, 0x00),
                (0x91, 0x00),
                (0x00, 0x01),
                (0xFF, 0x00),
            )
        )
        self._continuous_mode = False
        self._continuous_period_ms = 0
        logger.info("VL53L0X continuous ranging stopped")

    def _start_single_measurement(self) -> None:
        self._restore_stop_variable()
        self._write_u8(self.SYSRANGE_START, 0x01)
        self._wait_for(
            lambda: (self._read_u8(self.SYSRANGE_START) & 0x01) == 0,
            "start_single_measurement",
        )

    def read_measurement(
        self, *, require_valid: bool = False
    ) -> RangeMeasurement:
        """Read one result, starting a single-shot measurement when necessary.

        ``valid`` is true only when the sensor's range-status field is zero.
        Invalid optical results are returned with their status by default; set
        ``require_valid=True`` when the caller explicitly wants a SensorError.
        I2C failures and timeouts always raise ``HardwareError``.
        """

        self._require_initialized("read_measurement")
        if not self._continuous_mode:
            self._start_single_measurement()
        self._wait_for(lambda: self.data_ready, "wait_for_range")

        status = (self._read_u8(self.RESULT_RANGE_STATUS) & 0x78) >> 3
        distance_mm = self._read_u16(self.RESULT_RANGE_STATUS + 10)
        self.clear_interrupt()
        status_text = self.RANGE_STATUS_TEXT.get(
            status, "unknown_status_%d" % status
        )
        measurement = RangeMeasurement(
            distance_mm=distance_mm,
            range_status=status,
            status_text=status_text,
            valid=status == 0,
            monotonic_s=time.monotonic(),
        )
        self._last_measurement = measurement
        self._read_count += 1
        if not measurement.valid:
            self._invalid_read_count += 1
            logger.warning(
                "VL53L0X invalid range: distance=%dmm status=%d (%s)",
                distance_mm,
                status,
                status_text,
            )
            if require_valid:
                raise SensorError(
                    "VL53L0X.range",
                    {"distance_mm": distance_mm, "status": status_text},
                    ("range_valid", "range_valid"),
                )
        self._record_success()
        return measurement

    def read_range_single_mm(self, *, require_valid: bool = False) -> int:
        if self._continuous_mode:
            raise HardwareError(
                "VL53L0X",
                "read_range_single",
                "stop continuous ranging before a single-shot read",
            )
        return self.read_measurement(require_valid=require_valid).distance_mm

    def read_range_continuous_mm(self, *, require_valid: bool = False) -> int:
        if not self._continuous_mode:
            raise HardwareError(
                "VL53L0X",
                "read_range_continuous",
                "call start_continuous() first",
            )
        return self.read_measurement(require_valid=require_valid).distance_mm

    @property
    def range(self) -> int:
        """Distance in millimetres, using the active single/continuous mode."""

        return self.read_measurement().distance_mm

    @property
    def distance(self) -> float:
        """Distance in centimetres for compatibility with common Python APIs."""

        return self.range / 10.0

    def get_distance(self) -> int:
        """Legacy method returning the next distance in millimetres."""

        return self.range

    def start_ranging(
        self, mode: Any = Vl53l0xAccuracyMode.GOOD
    ) -> None:
        """Legacy wrapper: apply a profile and start continuous ranging."""

        self.set_profile(mode)
        self.start_continuous()

    def stop_ranging(self) -> None:
        self.stop_continuous()

    def get_timing(self) -> int:
        """Legacy method returning the actual timing budget in microseconds."""

        return self.measurement_timing_budget_us

    def configure_gpio_interrupt(
        self,
        proximity_alarm_type: int = Vl53l0xGpioAlarmType.NEW_MEASUREMENT_READY,
        interrupt_polarity: int = Vl53l0xInterruptPolarity.LOW,
        threshold_low_mm: int = 250,
        threshold_high_mm: int = 500,
    ) -> None:
        """Configure GPIO1 for range-ready or threshold interrupts."""

        self._require_initialized("configure_gpio_interrupt")
        functionality = require_int(
            proximity_alarm_type,
            "VL53L0X.gpio_alarm_type",
            minimum=Vl53l0xGpioAlarmType.OFF,
            maximum=Vl53l0xGpioAlarmType.NEW_MEASUREMENT_READY,
        )
        polarity = require_int(
            interrupt_polarity,
            "VL53L0X.interrupt_polarity",
            minimum=Vl53l0xInterruptPolarity.LOW,
            maximum=Vl53l0xInterruptPolarity.HIGH,
        )
        low = require_int(
            threshold_low_mm,
            "VL53L0X.threshold_low_mm",
            minimum=0,
            maximum=self.MAX_RANGE_MM,
        )
        high = require_int(
            threshold_high_mm,
            "VL53L0X.threshold_high_mm",
            minimum=0,
            maximum=self.MAX_RANGE_MM,
        )
        if low > high:
            raise ConfigurationError(
                "VL53L0X.interrupt_thresholds",
                (low, high),
                (0, self.MAX_RANGE_MM),
            )
        # ST's API stores half of each requested millimetre threshold because
        # the sensor firmware applies a factor of two when it evaluates them.
        self._write_u16(self.SYSTEM_THRESH_HIGH, (high >> 1) & 0x0FFF)
        self._write_u16(self.SYSTEM_THRESH_LOW, (low >> 1) & 0x0FFF)
        self._write_u8(
            self.SYSTEM_INTERRUPT_CONFIG_GPIO, functionality & 0x07
        )
        mux = self._read_u8(self.GPIO_HV_MUX_ACTIVE_HIGH)
        mux = (
            (mux | 0x10)
            if polarity == Vl53l0xInterruptPolarity.HIGH
            else (mux & ~0x10)
        )
        self._write_u8(self.GPIO_HV_MUX_ACTIVE_HIGH, mux)
        self.clear_interrupt()

    def clear_interrupt(self, *, require_initialized: bool = True) -> None:
        if require_initialized:
            self._require_initialized("clear_interrupt")
        self._write_u8(self.SYSTEM_INTERRUPT_CLEAR, 0x01)

    def set_address(self, new_address: int) -> int:
        """Assign a volatile 7-bit address until the sensor loses power.

        With multiple VL53L0X sensors, hold every other device's XSHUT pin low
        while changing one sensor from the shared 0x29 power-on address.
        """

        self._require_initialized("set_address")
        if self._continuous_mode:
            raise HardwareError(
                "VL53L0X",
                "set_address",
                "stop continuous ranging before changing address",
            )
        address = self._validate_address(
            new_address, "VL53L0X.new_address"
        )
        if address == self._address:
            return address
        old_address = self._address
        self._write_u8(self.I2C_SLAVE_DEVICE_ADDRESS, address & 0x7F)
        self._address = address
        identity = self._read_u8(self.IDENTIFICATION_MODEL_ID)
        if identity != self.EXPECTED_MODEL_ID:
            raise HardwareError(
                "VL53L0X",
                "set_address",
                "new address responded with model ID 0x%02X, expected 0xEE; "
                "driver retains 0x%02X because the address write completed"
                % (identity, address),
            )
        logger.info(
            "VL53L0X address changed: 0x%02X -> 0x%02X",
            old_address,
            address,
        )
        return address

    change_address = set_address

    def health(self) -> Dict[str, Any]:
        """Return a side-effect-free diagnostic snapshot for RoboCar telemetry."""

        last = self._last_measurement
        return {
            "initialized": self._initialized,
            "continuous": self._continuous_mode,
            "continuous_period_ms": self._continuous_period_ms,
            "address": self._address,
            "address_hex": "0x%02X" % self._address,
            "model_id": self._model_id,
            "module_type": self._module_type,
            "revision_id": self._revision_id,
            "profile": self._profile,
            "timing_budget_us": self._measurement_timing_budget_us,
            "read_count": self._read_count,
            "invalid_read_count": self._invalid_read_count,
            "timeout_count": self._timeout_count,
            "consecutive_errors": self._consecutive_errors,
            "last_error": self._last_error,
            "last_distance_mm": None if last is None else last.distance_mm,
            "last_range_status": None if last is None else last.status_text,
            "last_read_monotonic_s": (
                None if last is None else last.monotonic_s
            ),
        }

    def close(self) -> None:
        """Stop ranging and close only a bus created by this instance."""

        stop_error: Optional[BaseException] = None
        if self._continuous_mode:
            try:
                self.stop_continuous()
            except Exception as exc:
                stop_error = exc
                logger.error("VL53L0X stop during close failed: %s", exc)
        if self._owns_i2c and self._i2c is not None:
            try:
                self._release_owned_bus()
            except Exception as exc:
                self._record_error(exc)
                raise HardwareError(
                    "VL53L0X",
                    "close_i2c",
                    "%s: %s" % (type(exc).__name__, exc),
                ) from exc
        self._initialized = False
        self._continuous_mode = False
        if stop_error is not None:
            raise stop_error

    def __enter__(self) -> "VL53L0X":
        return self.open()

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


__all__ = [
    "RangeMeasurement",
    "VL53L0X",
    "Vl53l0xAccuracyMode",
    "Vl53l0xDeviceMode",
    "Vl53l0xGpioAlarmType",
    "Vl53l0xInterruptPolarity",
]


if __name__ == "__main__":
    print("\n=== Running VL53L0X Hardware Test ===\n")
    printer.status(
        "TEST", "Starting VL53L0X I2C1 / GP6-GP7 hardware test", "info"
    )

    sensor: Optional[VL53L0X] = None
    test_succeeded = False
    try:
        sensor = VL53L0X(
            i2c_bus=1,
            i2c_address=0x29,
            sda_pin=6,
            scl_pin=7,
            i2c_frequency_hz=400_000,
            io_timeout_s=0.75,
            poll_interval_s=0.001,
        )
        sensor.open()
        assert sensor.initialized
        assert sensor.address == 0x29
        identity = sensor.health()
        assert identity["model_id"] == VL53L0X.EXPECTED_MODEL_ID
        printer.status(
            "TEST",
            "Identity passed: model=0x%02X module=0x%02X revision=0x%02X"
            % (
                identity["model_id"],
                identity["module_type"],
                identity["revision_id"],
            ),
            "success",
        )

        applied_profile = sensor.set_profile(VL53L0X.PROFILE_GOOD)
        timing_budget_us = sensor.get_timing()
        assert applied_profile == VL53L0X.PROFILE_GOOD
        assert timing_budget_us >= VL53L0X.MIN_TIMING_BUDGET_US
        printer.status(
            "TEST",
            "Profile=%s timing_budget=%dus signal_limit=%.3fMCPS"
            % (
                applied_profile,
                timing_budget_us,
                sensor.signal_rate_limit_mcps,
            ),
            "success",
        )

        single = sensor.read_measurement()
        assert isinstance(single.distance_mm, int)
        assert 0 <= single.range_status <= 15
        printer.status(
            "TEST",
            "Single shot: %d mm, status=%s, valid=%s"
            % (single.distance_mm, single.status_text, single.valid),
            "success" if single.valid else "info",
        )

        sensor.start_continuous(period_ms=50)
        assert sensor.is_continuous_mode
        valid_samples = 0
        for sample_number in range(1, 6):
            sample = sensor.read_measurement()
            valid_samples += int(sample.valid)
            printer.status(
                "TEST",
                "Continuous sample %d/5: %d mm, status=%s"
                % (sample_number, sample.distance_mm, sample.status_text),
                "success" if sample.valid else "info",
            )
        sensor.stop_continuous()
        assert not sensor.is_continuous_mode

        health = sensor.health()
        assert health["read_count"] == 6
        assert health["timeout_count"] == 0
        printer.status(
            "TEST",
            "Read path passed: 6 samples, %d valid, %d sensor-status rejections"
            % (
                valid_samples + int(single.valid),
                health["invalid_read_count"],
            ),
            "success",
        )
        test_succeeded = True

    except Exception as exc:
        printer.status(
            "TEST",
            "VL53L0X hardware test failed: %s: %s"
            % (type(exc).__name__, exc),
            "error",
        )
        logger.exception("VL53L0X hardware test failed")
        raise

    finally:
        if sensor is not None:
            try:
                sensor.close()
                printer.status(
                    "TEST", "Sensor stopped and I2C resource closed", "info"
                )
            except Exception as exc:
                printer.status(
                    "TEST",
                    "Cleanup failed: %s: %s" % (type(exc).__name__, exc),
                    "error",
                )

    if test_succeeded:
        printer.status("TEST", "VL53L0X hardware test completed", "success")
        print("\n=== Test ran successfully ===\n")
