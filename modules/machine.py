"""RoboCar Linux compatibility layer for a narrow subset of MicroPython ``machine``.

This module exists for reusable RoboCar hardware/device drivers that run on the
Raspberry Pi Linux host but were written against the small ``machine.Pin`` /
``machine.I2C`` surface commonly used by MicroPython drivers.

Important boundary
------------------
This is deliberately *not* a general replacement for MicroPython's built-in
``machine`` module.

Implemented:
    - ``Pin`` as a validated descriptor.
    - Linux I2C through ``smbus2``.
    - raw I2C read/write.
    - register/memory read/write with 8- or 16-bit register addresses.
    - ``scan()``, ``readfrom_into()``, ``readfrom_mem_into()``, ``writevto()``.
    - deterministic close/deinit and context-manager support.
    - health/error counters.
    - the historical ``get_i2c_device()`` compatibility helper.

Not implemented:
    - direct Linux GPIO via ``Pin.value()``, IRQ, PWM, ADC, UART, SPI, timers,
      reset, sleep, or other MCU-specific facilities.

RoboCar currently keeps deterministic GPIO/sensor acquisition on the Raspberry
Pi Pico.  Therefore this module must not silently interpret a Pico ``GP<n>``
identifier as a Raspberry Pi BCM GPIO identifier.

All physical I/O failures are fail-closed and surfaced through RoboCar's
``HardwareError``.
"""

from __future__ import annotations

import errno
import threading
from collections.abc import Iterable
from typing import Any, Optional

from ..utils.rc_errors import *
from ..utils.rc_helpers import *
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("Machine")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Optional Linux backend
# ---------------------------------------------------------------------------

try:
    import smbus2 as _smbus2  # type: ignore
except ImportError as exc:
    # Keep the module importable in CI/development.  Construction of I2C fails
    # explicitly if the physical backend is actually requested.
    _smbus2 = None
    _SMBUS2_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _SMBUS2_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_I2C_MIN_ADDRESS = 0x00
_I2C_MAX_ADDRESS = 0x7F
_I2C_SCAN_MIN_ADDRESS = 0x08
_I2C_SCAN_MAX_ADDRESS = 0x77
_SUPPORTED_ADDR_SIZES = (8, 16)

# These are normal "no responder" results during a diagnostic bus scan.
_SCAN_MISS_ERRNOS = {
    errno.ENXIO,
    errno.EREMOTEIO,
    errno.EIO,
}


def _hardware_error(device: str, operation: str, exc: BaseException) -> HardwareError:
    return HardwareError(
        device=device,
        operation=operation,
        error_details=f"{type(exc).__name__}: {exc}",
    )


def _require_smbus2() -> Any:
    if _smbus2 is None:
        cause = _SMBUS2_IMPORT_ERROR or ImportError("smbus2 is unavailable")
        error = HardwareError(
            device="Linux I2C",
            operation="load_smbus2",
            error_details=(
                "smbus2 is required for RoboCar.modules.machine.I2C on Linux"
            ),
        )
        raise error from cause
    return _smbus2


def _i2c_address(value: Any, parameter: str = "machine.i2c.address") -> int:
    address = optional_int(
        value,
        minimum=_I2C_MIN_ADDRESS,
        maximum=_I2C_MAX_ADDRESS,
    )
    if address is None:
        raise ConfigurationError(
            parameter=parameter,
            value=value,
            valid_range=(_I2C_MIN_ADDRESS, _I2C_MAX_ADDRESS),
        )
    return address


def _length(value: Any, parameter: str) -> int:
    converted = optional_int(value, minimum=0)
    if converted is None:
        raise ConfigurationError(
            parameter=parameter,
            value=value,
            valid_range=(0, None),
        )
    return converted


def _addrsize(value: Any) -> int:
    converted = optional_int(value)
    if converted not in _SUPPORTED_ADDR_SIZES:
        raise ConfigurationError(
            parameter="machine.i2c.addrsize",
            value=value,
            valid_range=_SUPPORTED_ADDR_SIZES,
        )
    return converted


def _memaddr(value: Any, addrsize: int) -> int:
    maximum = (1 << addrsize) - 1
    converted = optional_int(value, minimum=0, maximum=maximum)
    if converted is None:
        raise ConfigurationError(
            parameter="machine.i2c.memaddr",
            value=value,
            valid_range=(0, maximum),
        )
    return converted


def _as_bytes(value: Any, parameter: str) -> bytes:
    # ``bytes(5)`` is valid Python but almost never means "write five zero
    # bytes" when a device driver accidentally supplies an integer.
    if value is None or isinstance(value, (bool, int, float)):
        raise TypeError(f"{parameter} must be bytes-like")
    try:
        return bytes(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{parameter} must be bytes-like") from exc


def _writable_bytes_view(value: Any, parameter: str) -> memoryview:
    try:
        view = memoryview(value)
    except TypeError as exc:
        raise TypeError(f"{parameter} must be a writable buffer") from exc

    if view.readonly:
        raise TypeError(f"{parameter} must be writable")

    try:
        return view.cast("B")
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{parameter} must be byte-addressable") from exc


# ---------------------------------------------------------------------------
# Pin
# ---------------------------------------------------------------------------


class Pin:
    """Validated pin descriptor.

    ``Pin`` intentionally does not claim a Raspberry Pi GPIO line.  It is safe
    to use as constructor metadata for bus-oriented drivers, e.g.::

        I2C(1, sda=Pin(2), scl=Pin(3))

    Direct GPIO access fails explicitly because Pico GP numbering and Raspberry
    Pi BCM numbering are not interchangeable.
    """

    IN = 0
    OUT = 1
    OPEN_DRAIN = 2

    PULL_UP = 3
    PULL_DOWN = 4

    IRQ_FALLING = 0x04
    IRQ_RISING = 0x08

    def __init__(
        self,
        pin: Any,
        mode: Optional[int] = None,
        pull: Optional[int] = None,
        *,
        value: Optional[int] = None,
    ) -> None:
        pin_id = optional_int(pin, minimum=0)
        if pin_id is None:
            raise ConfigurationError(
                parameter="machine.pin.id",
                value=pin,
                valid_range=(0, None),
            )

        self.id = pin_id
        self.mode = mode
        self.pull = pull
        self.initial_value: Optional[int] = None

        if value is not None:
            initial = optional_binary(value)
            if initial is None:
                raise ConfigurationError(
                    parameter="machine.pin.value",
                    value=value,
                    valid_range=(0, 1),
                )
            self.initial_value = initial

    def init(
        self,
        mode: Optional[int] = None,
        pull: Optional[int] = None,
        *,
        value: Optional[int] = None,
    ) -> None:
        """Update descriptor metadata only."""

        if mode is not None:
            self.mode = mode
        if pull is not None:
            self.pull = pull
        if value is not None:
            initial = optional_binary(value)
            if initial is None:
                raise ConfigurationError(
                    parameter="machine.pin.value",
                    value=value,
                    valid_range=(0, 1),
                )
            self.initial_value = initial

    def _unsupported_gpio(self, operation: str) -> HardwareError:
        return HardwareError(
            device=f"Pin({self.id})",
            operation=operation,
            error_details=(
                "RoboCar.modules.machine.Pin is descriptor-only. "
                "No implicit Pico-GP-to-Raspberry-Pi-BCM mapping is permitted."
            ),
        )

    def value(self, value: Optional[int] = None) -> int:
        raise self._unsupported_gpio(
            "write_gpio" if value is not None else "read_gpio"
        )

    __call__ = value

    def on(self) -> None:
        raise self._unsupported_gpio("write_gpio")

    def off(self) -> None:
        raise self._unsupported_gpio("write_gpio")

    def toggle(self) -> None:
        raise self._unsupported_gpio("toggle_gpio")

    def irq(self, *args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise self._unsupported_gpio("configure_irq")

    def __int__(self) -> int:
        return self.id

    def __index__(self) -> int:
        return self.id

    def __repr__(self) -> str:
        return (
            f"Pin({self.id}, mode={self.mode!r}, pull={self.pull!r}, "
            f"value={self.initial_value!r})"
        )


# ---------------------------------------------------------------------------
# I2C
# ---------------------------------------------------------------------------


class I2C:
    """MicroPython-shaped I2C adapter backed by Linux ``smbus2``.

    ``id`` maps to ``/dev/i2c-<id>``.

    ``sda``, ``scl``, and ``freq`` are retained for source compatibility and
    diagnostics.  On Linux, pin muxing and I2C clock configuration remain an OS
    responsibility; this class does not silently reconfigure them.
    """

    def __init__(
        self,
        id: Any = 1,
        *,
        sda: Optional[Pin] = None,
        scl: Optional[Pin] = None,
        freq: Any = 100_000,
    ) -> None:
        self.id = require_int(id, "machine.i2c.id", minimum=0)
        self.freq = require_int(freq, "machine.i2c.freq", minimum=1)
        self.sda = sda
        self.scl = scl

        self._lock = threading.RLock()
        self._bus: Optional[Any] = None
        self._closed = True
        self._operations = 0
        self._errors = 0
        self._last_error: Optional[str] = None

        backend = _require_smbus2()

        try:
            self._bus = backend.SMBus(self.id)
        except Exception as exc:
            self._errors += 1
            self._last_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Failed to open /dev/i2c-%s: %s",
                self.id,
                self._last_error,
            )
            raise _hardware_error(f"I2C({self.id})", "open", exc) from exc

        self._closed = False
        logger.info(
            "I2C bus %s opened (requested=%sHz, sda=%r, scl=%r)",
            self.id,
            self.freq,
            self.sda,
            self.scl,
        )

    # ---------------------------- internals ----------------------------

    def _require_open(self) -> Any:
        if self._closed or self._bus is None:
            raise HardwareError(
                device=f"I2C({self.id})",
                operation="access_closed_bus",
                error_details="I2C bus is closed",
            )
        return self._bus

    def _execute(self, operation: str, callback: Any) -> Any:
        with self._lock:
            bus = self._require_open()
            try:
                result = callback(bus)
            except HardwareError:
                raise
            except Exception as exc:
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.error(
                    "I2C(%s) %s failed: %s",
                    self.id,
                    operation,
                    self._last_error,
                )
                raise _hardware_error(
                    f"I2C({self.id})",
                    operation,
                    exc,
                ) from exc

            self._operations += 1
            self._last_error = None
            return result

    # ---------------------------- lifecycle ----------------------------

    def close(self) -> None:
        """Close the Linux I2C handle.  Repeated calls are harmless."""

        with self._lock:
            if self._closed:
                return

            bus = self._bus
            self._bus = None
            self._closed = True

            if bus is None:
                return

            try:
                bus.close()
            except Exception as exc:
                self._errors += 1
                self._last_error = f"{type(exc).__name__}: {exc}"
                raise _hardware_error(
                    f"I2C({self.id})",
                    "close",
                    exc,
                ) from exc

            logger.info("I2C bus %s closed", self.id)

    deinit = close

    def __enter__(self) -> "I2C":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        self.close()
        return False

    def __del__(self) -> None:
        # Never allow interpreter-shutdown cleanup to mask another exception.
        try:
            if not getattr(self, "_closed", True):
                bus = getattr(self, "_bus", None)
                if bus is not None:
                    bus.close()
        except Exception:
            pass

    # ---------------------------- diagnostics --------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    def health(self) -> dict[str, Any]:
        """Return adapter health without performing another hardware access."""

        return {
            "bus_id": self.id,
            "device": f"/dev/i2c-{self.id}",
            "closed": self._closed,
            "operations": self._operations,
            "errors": self._errors,
            "last_error": self._last_error,
            "requested_freq_hz": self.freq,
            "sda": int(self.sda) if isinstance(self.sda, Pin) else self.sda,
            "scl": int(self.scl) if isinstance(self.scl, Pin) else self.scl,
        }

    # ---------------------------- raw I/O ------------------------------

    def scan(self) -> list[int]:
        """Return responding 7-bit addresses.

        This is a diagnostic/startup operation, not something to run in a
        control loop.
        """

        def _scan(bus: Any) -> list[int]:
            found: list[int] = []
            for address in range(
                _I2C_SCAN_MIN_ADDRESS,
                _I2C_SCAN_MAX_ADDRESS + 1,
            ):
                try:
                    bus.write_quick(address)
                except OSError as exc:
                    if exc.errno in _SCAN_MISS_ERRNOS:
                        continue
                    raise
                found.append(address)
            return found

        found = list(self._execute("scan", _scan))
        logger.debug(
            "I2C(%s) scan: %s",
            self.id,
            [f"0x{address:02X}" for address in found],
        )
        return found

    def writeto(self, addr: Any, buf: Any, stop: bool = True) -> int:
        address = _i2c_address(addr)
        payload = _as_bytes(buf, "machine.i2c.writeto.buf")

        if stop is not True:
            raise HardwareError(
                device=f"I2C({self.id})",
                operation="writeto",
                error_details=(
                    "stop=False cannot be preserved across separate Linux SMBus "
                    "calls; use an explicit combined transaction instead"
                ),
            )

        def _write(bus: Any) -> int:
            backend = _require_smbus2()
            message = backend.i2c_msg.write(address, payload)
            bus.i2c_rdwr(message)
            return len(payload)

        return int(self._execute("writeto", _write))

    def writevto(
        self,
        addr: Any,
        vector: Iterable[Any],
        stop: bool = True,
    ) -> int:
        payload = b"".join(
            _as_bytes(part, "machine.i2c.writevto.item")
            for part in vector
        )
        return self.writeto(addr, payload, stop=stop)

    def readfrom(
        self,
        addr: Any,
        nbytes: Any,
        stop: bool = True,
    ) -> bytes:
        address = _i2c_address(addr)
        size = _length(nbytes, "machine.i2c.readfrom.nbytes")

        if stop is not True:
            raise HardwareError(
                device=f"I2C({self.id})",
                operation="readfrom",
                error_details=(
                    "stop=False cannot be represented as a persistent Linux "
                    "transaction between separate method calls"
                ),
            )

        if size == 0:
            return b""

        def _read(bus: Any) -> bytes:
            backend = _require_smbus2()
            message = backend.i2c_msg.read(address, size)
            bus.i2c_rdwr(message)
            return bytes(message)

        return bytes(self._execute("readfrom", _read))

    def readfrom_into(
        self,
        addr: Any,
        buf: Any,
        stop: bool = True,
    ) -> int:
        target = _writable_bytes_view(
            buf,
            "machine.i2c.readfrom_into.buf",
        )
        data = self.readfrom(addr, len(target), stop=stop)
        target[:] = data
        return len(data)

    # ---------------------------- register I/O -------------------------

    def writeto_mem(
        self,
        addr: Any,
        memaddr: Any,
        data: Any,
        *,
        addrsize: Any = 8,
    ) -> int:
        address = _i2c_address(addr)
        width = _addrsize(addrsize)
        register = _memaddr(memaddr, width)
        payload = _as_bytes(data, "machine.i2c.writeto_mem.data")
        prefix = register.to_bytes(width // 8, "big")

        def _write_mem(bus: Any) -> int:
            backend = _require_smbus2()
            message = backend.i2c_msg.write(
                address,
                prefix + payload,
            )
            bus.i2c_rdwr(message)
            return len(payload)

        return int(self._execute("writeto_mem", _write_mem))

    def readfrom_mem(
        self,
        addr: Any,
        memaddr: Any,
        nbytes: Any,
        *,
        addrsize: Any = 8,
    ) -> bytes:
        address = _i2c_address(addr)
        width = _addrsize(addrsize)
        register = _memaddr(memaddr, width)
        size = _length(
            nbytes,
            "machine.i2c.readfrom_mem.nbytes",
        )

        if size == 0:
            return b""

        prefix = register.to_bytes(width // 8, "big")

        def _read_mem(bus: Any) -> bytes:
            backend = _require_smbus2()
            set_register = backend.i2c_msg.write(address, prefix)
            read_data = backend.i2c_msg.read(address, size)
            # One I2C_RDWR ioctl: write register pointer, repeated START, read.
            bus.i2c_rdwr(set_register, read_data)
            return bytes(read_data)

        return bytes(self._execute("readfrom_mem", _read_mem))

    def readfrom_mem_into(
        self,
        addr: Any,
        memaddr: Any,
        buf: Any,
        *,
        addrsize: Any = 8,
    ) -> int:
        target = _writable_bytes_view(
            buf,
            "machine.i2c.readfrom_mem_into.buf",
        )
        data = self.readfrom_mem(
            addr,
            memaddr,
            len(target),
            addrsize=addrsize,
        )
        target[:] = data
        return len(data)

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return (
            f"I2C({self.id}, freq={self.freq}, sda={self.sda!r}, "
            f"scl={self.scl!r}, state={state!r})"
        )


# ---------------------------------------------------------------------------
# Legacy device helper
# ---------------------------------------------------------------------------


class _LegacyI2CDevice:
    """Compatibility wrapper for historical RoboCar/Adafruit-style drivers."""

    def __init__(self, addr: Any, busnum: Any = 1) -> None:
        self.address = _i2c_address(
            addr,
            "machine.legacy_i2c.address",
        )
        self.busnum = require_int(
            busnum,
            "machine.legacy_i2c.busnum",
            minimum=0,
        )
        self._i2c = I2C(self.busnum)
        self._closed = False

    def _require_open(self) -> None:
        if self._closed:
            raise HardwareError(
                device=f"I2CDevice(0x{self.address:02X})",
                operation="access_closed_device",
                error_details="Legacy I2C device is closed",
            )

    def writeRaw8(self, value: Any) -> None:
        self._require_open()
        byte = require_int(
            value,
            "machine.legacy_i2c.value",
            minimum=0,
            maximum=0xFF,
        )
        self._i2c.writeto(
            self.address,
            bytes((byte,)),
        )

    def write8(self, reg: Any, value: Any) -> None:
        self._require_open()
        byte = require_int(
            value,
            "machine.legacy_i2c.value",
            minimum=0,
            maximum=0xFF,
        )
        self._i2c.writeto_mem(
            self.address,
            reg,
            bytes((byte,)),
        )

    def readU8(self, reg: Any) -> int:
        self._require_open()
        return self._i2c.readfrom_mem(
            self.address,
            reg,
            1,
        )[0]

    def readList(self, reg: Any, length: Any) -> list[int]:
        self._require_open()
        size = _length(
            length,
            "machine.legacy_i2c.length",
        )
        return list(
            self._i2c.readfrom_mem(
                self.address,
                reg,
                size,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._i2c.close()
        self._closed = True

    deinit = close

    def __enter__(self) -> "_LegacyI2CDevice":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        self.close()
        return False

    def __repr__(self) -> str:
        return (
            f"_LegacyI2CDevice(address=0x{self.address:02X}, "
            f"busnum={self.busnum}, closed={self._closed})"
        )


def get_i2c_device(addr: Any, busnum: Any = 1) -> _LegacyI2CDevice:
    """Return the legacy register-oriented I2C adapter."""

    return _LegacyI2CDevice(addr, busnum)


__all__ = [
    "Pin",
    "I2C",
    "get_i2c_device",
]


if __name__ == "__main__":
    print("\n=== Running Machine Compatibility Adapter Tests ===\n")
    printer.status(
        "TEST",
        "Machine compatibility adapter initialized",
        "info",
    )

    pin = Pin(2)
    assert int(pin) == 2
    assert pin.id == 2
    printer.status(
        "TEST",
        "Pin descriptor validation",
        "success",
    )

    try:
        pin.value()
    except HardwareError:
        printer.status(
            "TEST",
            "Pin GPIO fail-closed behavior",
            "success",
        )
    else:
        raise AssertionError(
            "Pin.value() must fail closed without a GPIO backend"
        )

    assert _i2c_address(0x29) == 0x29
    assert _memaddr(0x1234, 16) == 0x1234
    assert _as_bytes(
        bytearray((1, 2, 3)),
        "test",
    ) == b"\x01\x02\x03"
    printer.status(
        "TEST",
        "I2C argument validation",
        "success",
    )

    print("\n=== Test ran successfully ===\n")
