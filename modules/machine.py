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
#  Simulated I2C bus backend
# ---------------------------------------------------------------------------
#
# ``_SimulatedI2CBus`` and ``_SimulatedSMBus`` stand in for ``smbus2.SMBus``
# when a physical bus cannot be opened (see ``I2C.__init__``).  Both classes
# expose exactly the surface ``I2C`` actually drives -- ``write_quick``,
# ``i2c_rdwr``, and ``close`` -- so a driver written against ``machine.I2C``
# behaves identically under simulation and under real hardware.
#
# Register-pointer model
# -----------------------
# I2C itself has no concept of "registers"; it is a bus for exchanging raw
# byte strings with an address.  Register-oriented access (``writeto_mem`` /
# ``readfrom_mem``) is a *software convention* layered on top by device
# drivers: a write whose payload begins with one or more address bytes
# followed by data, and a read that resumes from wherever the device's
# internal pointer was last left.  ``I2C`` in this module encodes that
# convention in exactly two shapes, both reproduced here:
#
#   1. Combined write (``writeto_mem``): a single ``i2c_msg.write`` whose
#      payload is ``<register-prefix><data...>``, sent alone.  The simulator
#      must know how many leading bytes are address versus data; this is
#      configurable per device via ``set_register_width`` (default 1 byte,
#      i.e. ``addrsize=8``) because the bus cannot infer it from the wire.
#
#   2. Split write+read (``readfrom_mem``): a pointer-only ``i2c_msg.write``
#      (register prefix, zero data bytes) immediately followed by an
#      ``i2c_msg.read`` in the *same* ``i2c_rdwr()`` call.  Because the write
#      payload here is unambiguously "the whole thing is the pointer",
#      addrsize is irrelevant and 8-/16-bit register addressing both work
#      without configuration.
#
# After either shape, the device's internal pointer auto-increments past the
# bytes touched, mirroring the auto-increment behavior of essentially every
# real I2C peripheral register file (EEPROMs, IMUs, PWM drivers, etc.), so
# that a driver issuing several small reads/writes in sequence observes the
# same "walk forward through the register map" behavior it would on hardware.
#
# Fault injection
# ----------------
# Real buses fail: a device may be absent (ENXIO), wedged (EIO/EREMOTEIO), or
# the fd may be invalid (EBADF).  Tests that exercise ``I2C.scan()`` or
# ``HardwareError`` fallback paths need to provoke those failures
# deterministically without physical hardware.  Both classes therefore accept
# an explicit "present address" allow-list and per-address fault injection;
# by default every address ACKs (permissive, matching a bare in-memory model)
# so existing callers that never configure the simulator keep working
# unchanged.

class _SimulatedI2CError(OSError):
    """``OSError`` raised by the simulator with Linux-shaped ``errno`` values.

    Real ``smbus2``/``ioctl`` failures surface as :class:`OSError` with a
    POSIX ``errno`` (``ENXIO`` for "no such device", ``EIO``/``EREMOTEIO``
    for a wedged responder, ``EBADF`` for an already-closed file descriptor).
    ``I2C._execute`` wraps *any* exception into ``HardwareError``, and
    ``I2C.scan()`` specifically inspects ``exc.errno`` against
    ``_SCAN_MISS_ERRNOS`` to decide whether a missing responder is a normal
    scan miss or a genuine bus fault.  Raising plain ``OSError`` here (rather
    than a bespoke exception type) keeps both call sites working exactly as
    they would against a real kernel I2C driver.
    """

    def __init__(self, errno_value: int, message: str) -> None:
        super().__init__(errno_value, message)


class _SimulatedBusMemory:
    """Shared in-memory transport for the simulated bus backends.

    This holds everything that is *not* part of the public ``smbus2``-shaped
    API: per-address byte storage, pointer tracking, presence/fault
    configuration, and locking.  ``_SimulatedI2CBus`` and ``_SimulatedSMBus``
    both derive from it so the register-pointer semantics described above are
    implemented exactly once and cannot drift between the two backends.

    Not thread-safe by omission: every public entry point below acquires
    ``self._lock``, matching the coarse-grained locking ``I2C`` itself applies
    via its own ``threading.RLock``. Holding a second, independent lock here
    keeps the simulator safe to drive directly in tests that bypass ``I2C``
    and call the bus methods concurrently from multiple threads.
    """

    def __init__(self, bus_id: Any) -> None:
        self.bus_id = bus_id
        self._lock = threading.RLock()
        self._closed = False

        # address -> {register_offset: 0..255}; sparse, so 8-bit and 16-bit
        # register spaces are both represented without pre-allocating memory.
        self._memory: dict[int, dict[int, int]] = {}
        # address -> current register pointer (auto-incrementing).
        self._pointers: dict[int, int] = {}
        # address -> number of leading payload bytes treated as a register
        # prefix for *combined* write+data messages (see module note above).
        self._register_width: dict[int, int] = {}
        # None => every 7-bit address ACKs (default/permissive). A concrete
        # set restricts ACKs to those addresses.
        self._present: Optional[set[int]] = None
        # Addresses that should fail every transaction (EIO), independent of
        # presence, to simulate a wedged/misbehaving responder.
        self._faulted: set[int] = set()

    # ------------------------------------------------------------------
    # Test/fixture configuration (not part of the smbus2 surface)
    # ------------------------------------------------------------------

    def register_device(
        self,
        address: int,
        *,
        initial: Optional[dict[int, int]] = None,
        register_width: int = 1,
    ) -> None:
        """Pre-seed a device's memory and declare it present on the bus.

        ``initial`` maps register offsets to starting byte values (any
        offsets omitted default to ``0x00``, matching freshly powered-on
        hardware with unspecified reset state treated as zeroed). Calling
        this method for at least one address switches the bus out of its
        default "every address ACKs" mode: once any device is explicitly
        registered, ``scan()``/``write_quick()`` only ACK registered
        addresses, so tests that want a realistic scan must register every
        simulated peripheral they expect to see.
        """

        with self._lock:
            self._present = self._present or set()
            self._present.add(address)
            cells = self._memory.setdefault(address, {})
            if initial:
                cells.update(initial)
            self._pointers.setdefault(address, 0)
            self.set_register_width(address, register_width)

    def set_register_width(self, address: int, width: int) -> None:
        """Set the register-prefix width (in bytes) used for combined
        write+data transactions targeting ``address`` (see module note
        above). MicroPython/CircuitPython ``addrsize`` is either 8 or 16
        bits, i.e. a 1- or 2-byte prefix; this must match whatever the driver
        under test configures via ``I2C.writeto_mem(..., addrsize=...)``.
        """

        if width < 1:
            raise ValueError("register_width must be at least 1 byte")
        with self._lock:
            self._register_width[address] = width

    def set_present_addresses(self, addresses: Optional[Iterable[int]]) -> None:
        """Restrict which 7-bit addresses ACK. ``None`` restores the
        permissive default where every address ACKs."""

        with self._lock:
            self._present = None if addresses is None else set(addresses)

    def inject_fault(self, address: int) -> None:
        """Make every subsequent transaction against ``address`` fail with
        ``EIO``, simulating a wedged or misbehaving responder."""

        with self._lock:
            self._faulted.add(address)

    def clear_fault(self, address: int) -> None:
        """Undo :meth:`inject_fault` for ``address``."""

        with self._lock:
            self._faulted.discard(address)

    def dump(self, address: int) -> dict[int, int]:
        """Return a snapshot copy of ``address``'s simulated register file,
        for test assertions."""

        with self._lock:
            return dict(self._memory.get(address, {}))

    def reset(self) -> None:
        """Clear all simulated state, as if every device were power-cycled
        and the fault/presence configuration reset to defaults."""

        with self._lock:
            self._memory.clear()
            self._pointers.clear()
            self._register_width.clear()
            self._present = None
            self._faulted.clear()

    # ------------------------------------------------------------------
    # Internals shared by the smbus2-shaped surface
    # ------------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise _SimulatedI2CError(
                errno.EBADF,
                f"simulated I2C bus {self.bus_id} is closed",
            )

    def _require_ack(self, address: int) -> None:
        if self._present is not None and address not in self._present:
            raise _SimulatedI2CError(
                errno.ENXIO,
                f"simulated I2C bus {self.bus_id}: no responder at "
                f"0x{address:02X}",
            )
        if address in self._faulted:
            raise _SimulatedI2CError(
                errno.EIO,
                f"simulated I2C bus {self.bus_id}: responder at "
                f"0x{address:02X} is faulted",
            )

    def _cells_for(self, address: int) -> dict[int, int]:
        return self._memory.setdefault(address, {})

    def _do_write_quick(self, address: int) -> None:
        with self._lock:
            self._require_open()
            self._require_ack(address)
            # A quick-write is purely a presence probe; it deliberately does
            # not touch stored register contents or the pointer, matching
            # SMBus_write_quick() semantics on real hardware.
            self._cells_for(address)

    def _do_i2c_rdwr(self, *messages: Any) -> None:
        with self._lock:
            self._require_open()

            # Detect the split write(pointer-only)+read shape described in
            # the module note: a zero-data write immediately followed by a
            # read for the same address, both within this call. When present,
            # the write's entire payload is the pointer, unambiguously (no
            # register_width guess needed), and the read must not be treated
            # as a fresh, independently-addressed transaction.
            index = 0
            while index < len(messages):
                message = messages[index]
                address = self._require_message_address(message)
                self._require_ack(address)
                is_read = bool(message.flags & 0x01)

                if not is_read and index + 1 < len(messages):
                    next_message = messages[index + 1]
                    next_address = self._require_message_address(next_message)
                    next_is_read = bool(next_message.flags & 0x01)
                    if next_is_read and next_address == address:
                        # Split write(pointer-only)+read shape: the write's
                        # entire payload is the register pointer, unambiguous
                        # regardless of configured register_width (see the
                        # module note above ``_SimulatedBusMemory``).
                        payload = bytes(message)
                        if payload:
                            self._pointers[address] = int.from_bytes(
                                payload, "big"
                            )
                        self._service_read(next_address, next_message)
                        index += 2
                        continue

                if is_read:
                    self._service_read(address, message)
                else:
                    self._service_write(address, message)
                index += 1

    @staticmethod
    def _require_message_address(message: Any) -> int:
        address = getattr(message, "addr", None)
        if address is None:
            raise TypeError(
                "simulated i2c_rdwr() requires smbus2.i2c_msg-shaped "
                "objects exposing '.addr'"
            )
        return int(address)

    def _service_write(self, address: int, message: Any) -> None:
        payload = bytes(message)
        if not payload:
            return
        width = self._register_width.get(address, 1)
        cells = self._cells_for(address)
        if len(payload) <= width:
            # Pointer-only write with no trailing data (e.g. a bare
            # register-select with the split shape not detected above
            # because no read followed in this call): just move the pointer.
            self._pointers[address] = int.from_bytes(payload, "big")
            return
        pointer = int.from_bytes(payload[:width], "big")
        data = payload[width:]
        for offset, byte_value in enumerate(data):
            cells[pointer + offset] = byte_value
        self._pointers[address] = pointer + len(data)

    def _service_read(self, address: int, message: Any) -> None:
        cells = self._cells_for(address)
        pointer = self._pointers.get(address, 0)
        length = len(message)
        data = bytes(cells.get(pointer + i, 0x00) for i in range(length))
        self._deliver(message, data)
        self._pointers[address] = pointer + length

    @staticmethod
    def _deliver(message: Any, data: bytes) -> None:
        """Copy ``data`` into a real ``smbus2.i2c_msg`` read buffer.

        ``smbus2.i2c_msg.buf`` is a ctypes ``c_char`` array; each element is
        assigned as a length-1 ``bytes`` object. Falling back to whole-buffer
        slice assignment covers alternate ``i2c_msg``-compatible shims (e.g.
        a lightweight test double) that expose a plain mutable byte buffer
        instead of the ctypes array smbus2 uses.
        """

        try:
            for offset, byte_value in enumerate(data):
                message.buf[offset] = bytes((byte_value,))
        except (TypeError, ValueError, IndexError):
            try:
                message.buf[: len(data)] = data
            except Exception as exc:
                raise _SimulatedI2CError(
                    errno.EPROTO,
                    f"cannot deliver {len(data)} simulated read byte(s) "
                    f"into {type(message).__name__}.buf",
                ) from exc

    def _do_close(self) -> None:
        with self._lock:
            self._closed = True

    def __enter__(self) -> "_SimulatedBusMemory":
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        del exc_type, exc, tb
        self._do_close()
        return False

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return (
            f"{type(self).__name__}(bus_id={self.bus_id!r}, state={state!r}, "
            f"devices={sorted(self._memory)!r})"
        )


class _SimulatedI2CBus(_SimulatedBusMemory):
    """In-memory I2C bus for testing when smbus2 is unavailable.

    Implements only the raw, protocol-agnostic surface a bare I2C bus offers
    -- address probing (``write_quick``) and message-level transactions
    (``i2c_rdwr``) -- with no higher-level "SMBus command" convenience
    methods. Use this backend when simulating access patterns that only ever
    go through :class:`I2C`'s raw ``writeto``/``readfrom``/``writeto_mem``/
    ``readfrom_mem`` family, which is how this module drives the bus.
    """

    def write_quick(self, addr: int) -> None:
        """Probe ``addr`` without transferring data (used by ``I2C.scan()``).

        Raises ``OSError(errno.ENXIO)`` if ``addr`` is not present and
        ``OSError(errno.EIO)`` if it has been fault-injected, exactly as
        ``smbus2.SMBus.write_quick`` does against unresponsive hardware.
        """

        self._do_write_quick(addr)

    def i2c_rdwr(self, *messages: Any) -> None:
        """Execute one or more ``smbus2.i2c_msg`` objects as a single,
        repeated-START transaction, exactly as ``smbus2.SMBus.i2c_rdwr``
        does. Write messages deposit bytes into the addressed device's
        simulated memory and advance its register pointer; read messages are
        filled in place (via ``message.buf``) from that memory starting at
        the current pointer. See the module-level note above for how the
        register-pointer convention is inferred from message shape.
        """

        self._do_i2c_rdwr(*messages)

    def close(self) -> None:
        """Idempotently mark the bus closed. Further operations raise
        ``OSError(errno.EBADF)``, matching a closed Linux file descriptor."""

        self._do_close()


class _SimulatedSMBus(_SimulatedI2CBus):
    """In-memory stand-in for ``smbus2.SMBus``.

    Extends :class:`_SimulatedI2CBus` with the classic SMBus "command code"
    convenience methods (``read_byte_data``, ``write_i2c_block_data``, etc.).
    ``machine.I2C`` itself only ever calls ``write_quick``/``i2c_rdwr``/
    ``close`` (inherited unchanged from :class:`_SimulatedI2CBus`) -- these
    additions exist so that test code or drivers written directly against
    the ``smbus2.SMBus`` API (rather than through ``machine.I2C``) can be
    pointed at this simulator as a drop-in substitute without gaps.

    All command-code methods below are implemented in terms of the same
    register-pointer memory model used by ``i2c_rdwr``, so a value written
    via ``write_byte_data`` is visible to a subsequent ``I2C.readfrom_mem``
    against the same offset, and vice versa -- there is exactly one source
    of truth for simulated device state.
    """

    def read_byte(self, address: int) -> int:
        """Read one byte from the device's current pointer without first
        selecting a register (``SMBus.read_byte``)."""

        with self._lock:
            self._require_open()
            self._require_ack(address)
            cells = self._cells_for(address)
            pointer = self._pointers.get(address, 0)
            value = cells.get(pointer, 0x00)
            self._pointers[address] = pointer + 1
            return value

    def write_byte(self, address: int, value: int) -> None:
        """Write one byte at the device's current pointer with no register
        select (``SMBus.write_byte``)."""

        with self._lock:
            self._require_open()
            self._require_ack(address)
            cells = self._cells_for(address)
            pointer = self._pointers.get(address, 0)
            cells[pointer] = value & 0xFF
            self._pointers[address] = pointer + 1

    def read_byte_data(self, address: int, register: int) -> int:
        """Select ``register`` then read one byte (``SMBus.read_byte_data``)."""

        with self._lock:
            self._require_open()
            self._require_ack(address)
            cells = self._cells_for(address)
            value = cells.get(register, 0x00)
            self._pointers[address] = register + 1
            return value

    def write_byte_data(self, address: int, register: int, value: int) -> None:
        """Select ``register`` then write one byte (``SMBus.write_byte_data``)."""

        with self._lock:
            self._require_open()
            self._require_ack(address)
            cells = self._cells_for(address)
            cells[register] = value & 0xFF
            self._pointers[address] = register + 1

    def read_word_data(self, address: int, register: int) -> int:
        """Select ``register`` then read two little-endian bytes
        (``SMBus.read_word_data``)."""

        low = self.read_byte_data(address, register)
        high = self.read_byte_data(address, register + 1)
        return (high << 8) | low

    def write_word_data(self, address: int, register: int, value: int) -> None:
        """Select ``register`` then write two little-endian bytes
        (``SMBus.write_word_data``)."""

        value &= 0xFFFF
        self.write_byte_data(address, register, value & 0xFF)
        self.write_byte_data(address, register + 1, (value >> 8) & 0xFF)

    def read_i2c_block_data(
        self, address: int, register: int, length: int
    ) -> list[int]:
        """Select ``register`` then read ``length`` sequential bytes
        (``SMBus.read_i2c_block_data``)."""

        if length < 0:
            raise ValueError("length must be non-negative")
        with self._lock:
            self._require_open()
            self._require_ack(address)
            cells = self._cells_for(address)
            data = [cells.get(register + i, 0x00) for i in range(length)]
            self._pointers[address] = register + length
            return data

    def write_i2c_block_data(
        self, address: int, register: int, data: Iterable[int]
    ) -> None:
        """Select ``register`` then write a sequence of bytes
        (``SMBus.write_i2c_block_data``)."""

        payload = list(data)
        with self._lock:
            self._require_open()
            self._require_ack(address)
            cells = self._cells_for(address)
            for offset, byte_value in enumerate(payload):
                cells[register + offset] = byte_value & 0xFF
            self._pointers[address] = register + len(payload)


class _SimulatedI2CMessage:
    """Minimal stand-in for smbus2.i2c_msg when smbus2 is unavailable."""
    def __init__(self, addr: int, flags: int, buf: Any) -> None:
        self.addr = addr
        self.flags = flags          # 0 = write, 1 = read (same as smbus2)
        self.buf = buf              # bytes for write, bytearray for read

    def __len__(self) -> int:
        return len(self.buf)

    def __bytes__(self) -> bytes:
        return bytes(self.buf)


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
    errno.EREMOTE,
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
            error_details="smbus2 is required for RoboCar.modules.machine.I2C on Linux",
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
# ADC
# ---------------------------------------------------------------------------

class ADC:
    """Stub ADC for Linux compatibility. Real ADC is not supported.

    The reading value can be set via the constructor or the ``value`` property.
    """
    def __init__(self, pin: Pin, value: int = 32768) -> None:
        self.pin = pin
        self._value = value

    def read_u16(self) -> int:
        """Return the current simulated 16‑bit ADC reading (0‑65535)."""
        return self._value

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, new_value: int) -> None:
        if not 0 <= new_value <= 65535:
            raise ValueError("ADC value must be between 0 and 65535")
        self._value = new_value

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

    _simulate_gpio: bool = False

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
        self._simulated_state: Optional[int] = None

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
        if self._simulate_gpio:
            if value is not None:
                self._simulated_state = 1 if value else 0
            return self._simulated_state if self._simulated_state is not None else 0
        raise self._unsupported_gpio(
            "write_gpio" if value is not None else "read_gpio"
        )

    # Add low() and high() as aliases to value():
    def low(self) -> None:
        self.value(0)

    def high(self) -> None:
        self.value(1)

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

        self._simulated = False
        if _smbus2 is not None:
            try:
                self._bus = _smbus2.SMBus(self.id)
                logger.info("I2C bus %s opened via smbus2", self.id)
            except Exception as exc:
                logger.warning(
                    "Failed to open real I2C bus %s: %s; falling back to simulated",
                    self.id, exc
                )
                self._bus = _SimulatedSMBus(self.id)
                self._simulated = True
        else:
            logger.warning("smbus2 not available; using simulated I2C bus %s", self.id)
            self._bus = _SimulatedSMBus(self.id)
            self._simulated = True

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
        def _write(bus):
            if _smbus2 is not None:
                message = _smbus2.i2c_msg.write(address, payload)
            else:
                message = _SimulatedI2CMessage(address, 0, payload)
            bus.i2c_rdwr(message)
            return len(payload)
        return self._execute("writeto", _write)

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

        def _read(bus):
            if _smbus2 is not None:
                message = _smbus2.i2c_msg.read(address, size)
            else:
                message = _SimulatedI2CMessage(address, 1, bytearray(size))
            bus.i2c_rdwr(message)
            return bytes(message.buf)
        return self._execute("readfrom", _read)

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

        def _write_mem(bus):
            full = prefix + payload
            if _smbus2 is not None:
                message = _smbus2.i2c_msg.write(address, full)
            else:
                message = _SimulatedI2CMessage(address, 0, full)
            bus.i2c_rdwr(message)
            return len(payload)
        return self._execute("writeto_mem", _write_mem)

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

        def _read_mem(bus):
            if _smbus2 is not None:
                set_reg = _smbus2.i2c_msg.write(address, prefix)
                read_msg = _smbus2.i2c_msg.read(address, size)
            else:
                set_reg = _SimulatedI2CMessage(address, 0, prefix)
                read_msg = _SimulatedI2CMessage(address, 1, bytearray(size))
            bus.i2c_rdwr(set_reg, read_msg)
            return bytes(read_msg.buf)
        return self._execute("readfrom_mem", _read_mem)

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
    "ADC",
    "Pin",
    "I2C",
    "get_i2c_device",
]


if __name__ == "__main__":
    print("\n=== Running Machine Compatibility Adapter Tests ===\n")
    printer.status("TEST", "Machine compatibility adapter initialized", "info")

    pin = Pin(2)
    assert int(pin) == 2
    assert pin.id == 2
    printer.status("TEST", "Pin descriptor validation", "success")

    try:
        pin.value()
    except HardwareError:
        printer.status("TEST", "Pin GPIO fail-closed behavior", "success")
    else:
        raise AssertionError("Pin.value() must fail closed without a GPIO backend")

    assert _i2c_address(0x29) == 0x29
    assert _memaddr(0x1234, 16) == 0x1234
    assert _as_bytes(
        bytearray((1, 2, 3)),
        "test",
    ) == b"\x01\x02\x03"
    printer.status("TEST", "I2C argument validation", "success")

    print("\n=== Test ran successfully ===\n")