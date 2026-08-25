"""Synchronous deterministic watchdogs for RoboCar.

Critical watchdog detection never routes through a reasoner/planner.  The
caller receives an explicit :class:`WatchdogReport` and must immediately invoke
the hardware-safe stop callback before delegating recovery to SLAI HandlerAgent.

The module itself owns no thread.  Call :meth:`VehicleWatchdog.check` from the
fast vehicle loop at a deterministic cadence.
"""

from __future__ import annotations

import math
import threading
import time

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple


class WatchdogSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True, slots=True)
class WatchdogThresholds:
    sensor_frame_timeout_s: Optional[float] = None
    pico_heartbeat_timeout_s: Optional[float] = None
    control_cycle_deadline_s: Optional[float] = None
    gnss_timeout_s: Optional[float] = None
    planner_timeout_s: Optional[float] = None

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if value is not None:
                numeric = float(value)
                if not math.isfinite(numeric) or numeric <= 0.0:
                    raise ValueError(f"{name} must be finite and > 0 when enabled")


@dataclass(frozen=True, slots=True)
class WatchdogInputs:
    now_monotonic: float
    last_sensor_frame_monotonic: Optional[float] = None
    last_pico_heartbeat_monotonic: Optional[float] = None
    last_control_cycle_duration_s: Optional[float] = None
    actuator_status: Optional[str] = None
    actuator_fault: bool = False
    last_gnss_fix_monotonic: Optional[float] = None
    gnss_required: bool = False
    last_plan_monotonic: Optional[float] = None
    planner_required: bool = False


@dataclass(frozen=True, slots=True)
class WatchdogEvent:
    code: str
    severity: WatchdogSeverity
    message: str
    age_or_value: Optional[float] = None
    threshold: Optional[float] = None


@dataclass(frozen=True, slots=True)
class WatchdogReport:
    timestamp_monotonic: float
    safe: bool
    requires_stop: bool
    events: Tuple[WatchdogEvent, ...]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        for item in payload["events"]:
            item["severity"] = item["severity"].value
        return payload


class VehicleWatchdog:
    """Evaluate freshness/deadline/hardware faults without side effects."""

    CRITICAL_ACTUATOR_STATES = {"failed", "fault", "faulty", "critical"}

    def __init__(self, thresholds: WatchdogThresholds) -> None:
        self.thresholds = thresholds
        self._lock = threading.RLock()
        self._checks = 0
        self._critical_reports = 0
        self._last_report: Optional[WatchdogReport] = None

    @property
    def last_report(self) -> Optional[WatchdogReport]:
        with self._lock:
            return self._last_report

    def check(self, inputs: WatchdogInputs) -> WatchdogReport:
        now = float(inputs.now_monotonic)
        if not math.isfinite(now):
            raise ValueError("now_monotonic must be finite")

        events: list[WatchdogEvent] = []

        self._freshness_event(
            events,
            code="sensor_frame_stale",
            label="Pico sensor frame",
            now=now,
            last=inputs.last_sensor_frame_monotonic,
            threshold=self.thresholds.sensor_frame_timeout_s,
            required=True,
            severity=WatchdogSeverity.CRITICAL,
        )
        self._freshness_event(
            events,
            code="pico_heartbeat_stale",
            label="Pico heartbeat",
            now=now,
            last=inputs.last_pico_heartbeat_monotonic,
            threshold=self.thresholds.pico_heartbeat_timeout_s,
            required=True,
            severity=WatchdogSeverity.CRITICAL,
        )

        if self.thresholds.control_cycle_deadline_s is not None:
            duration = inputs.last_control_cycle_duration_s
            if duration is not None and float(duration) > self.thresholds.control_cycle_deadline_s:
                events.append(
                    WatchdogEvent(
                        code="control_cycle_deadline_miss",
                        severity=WatchdogSeverity.CRITICAL,
                        message="Control loop exceeded configured deadline",
                        age_or_value=float(duration),
                        threshold=self.thresholds.control_cycle_deadline_s,
                    )
                )

        actuator_status = str(inputs.actuator_status or "").strip().lower()
        if inputs.actuator_fault or actuator_status in self.CRITICAL_ACTUATOR_STATES:
            events.append(
                WatchdogEvent(
                    code="actuator_fault",
                    severity=WatchdogSeverity.CRITICAL,
                    message=f"Actuator/PWM boundary reported fault state {actuator_status!r}",
                )
            )

        self._freshness_event(
            events,
            code="gnss_stale",
            label="GNSS fix",
            now=now,
            last=inputs.last_gnss_fix_monotonic,
            threshold=self.thresholds.gnss_timeout_s,
            required=inputs.gnss_required,
            severity=(
                WatchdogSeverity.CRITICAL
                if inputs.gnss_required
                else WatchdogSeverity.WARNING
            ),
        )
        self._freshness_event(
            events,
            code="planner_result_stale",
            label="Planner result",
            now=now,
            last=inputs.last_plan_monotonic,
            threshold=self.thresholds.planner_timeout_s,
            required=inputs.planner_required,
            severity=(
                WatchdogSeverity.CRITICAL
                if inputs.planner_required
                else WatchdogSeverity.WARNING
            ),
        )

        requires_stop = any(
            event.severity == WatchdogSeverity.CRITICAL for event in events
        )
        report = WatchdogReport(
            timestamp_monotonic=now,
            safe=not requires_stop,
            requires_stop=requires_stop,
            events=tuple(events),
        )
        with self._lock:
            self._checks += 1
            if requires_stop:
                self._critical_reports += 1
            self._last_report = report
        return report

    @staticmethod
    def enforce(
        report: WatchdogReport,
        *,
        stop_callback: Callable[[], Any],
        event_callback: Optional[Callable[[WatchdogEvent], Any]] = None,
        recovery_callback: Optional[Callable[[WatchdogReport], Any]] = None,
    ) -> bool:
        """Apply the required ordering: STOP -> record -> recovery.

        Returns ``True`` if a critical report caused a stop.
        """

        if not report.requires_stop:
            if event_callback is not None:
                for event in report.events:
                    event_callback(event)
            return False

        # Hard boundary first.  Never invoke HandlerAgent/reasoning before this.
        stop_callback()

        if event_callback is not None:
            for event in report.events:
                event_callback(event)

        if recovery_callback is not None:
            recovery_callback(report)

        return True

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "checks": self._checks,
                "critical_reports": self._critical_reports,
                "last_report": (
                    None if self._last_report is None else self._last_report.to_dict()
                ),
            }

    @staticmethod
    def _freshness_event(
        events: list[WatchdogEvent],
        *,
        code: str,
        label: str,
        now: float,
        last: Optional[float],
        threshold: Optional[float],
        required: bool,
        severity: WatchdogSeverity,
    ) -> None:
        if threshold is None:
            return
        if last is None:
            if required:
                events.append(
                    WatchdogEvent(
                        code=code,
                        severity=severity,
                        message=f"{label} has never been observed",
                        threshold=threshold,
                    )
                )
            return
        age = max(0.0, now - float(last))
        if age > threshold:
            events.append(
                WatchdogEvent(
                    code=code,
                    severity=severity,
                    message=f"{label} exceeded freshness/deadline threshold",
                    age_or_value=age,
                    threshold=threshold,
                )
            )


__all__ = [
    "WatchdogSeverity",
    "WatchdogThresholds",
    "WatchdogInputs",
    "WatchdogEvent",
    "WatchdogReport",
    "VehicleWatchdog",
]
