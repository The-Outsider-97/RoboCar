"""Typed authoritative world-state model for RoboCar.

This module intentionally contains no SLAI imports and no hardware ownership.
The in-process :class:`WorldModel` is the authoritative domain state.  SLAI
SharedMemory (or any other key/value store) may mirror snapshots for agent
visibility, but arbitrary shared-memory keys are not the semantic source of
truth.

Design principles
-----------------
* immutable snapshots for readers;
* monotonic revision numbers;
* explicit timestamps and freshness;
* bounded event history;
* typed sub-states for vehicle, obstacles, route, health and actuation;
* no hidden background threads;
* no invented sensor covariance or confidence;
* optional mirroring through a tiny structural ``set(key, value)`` protocol.

Coordinates
-----------
All metric pose/path coordinates are expressed in one caller-defined local
Cartesian frame in metres.  ``yaw_rad`` follows the mathematical convention:
0 rad points along +X and positive rotation is counter-clockwise.  WGS-84 GNSS
coordinates are stored separately and must be transformed by the localization/
geodesy layer before being used by metric path/control algorithms.
"""

from __future__ import annotations

import math
import threading
import time

from collections import deque
from dataclasses import dataclass, field, fields, is_dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import Any, Deque, Dict, Iterable, Mapping, Optional, Protocol, Sequence, Tuple


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be numeric, not bool")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _optional_finite(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    return _finite(value, name)


def _probability(value: Any, name: str) -> float:
    result = _finite(value, name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return result


def _normalize_yaw(yaw_rad: float) -> float:
    value = _finite(yaw_rad, "yaw_rad")
    return math.atan2(math.sin(value), math.cos(value))


def _safe_mapping(value: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


class WorldModelError(RuntimeError):
    """Base error for invalid world-state transitions."""


class OperatingMode(str, Enum):
    STOPPED = "stopped"
    MANUAL = "manual"
    AUTONOMOUS = "autonomous"
    DEGRADED = "degraded"
    EMERGENCY_STOP = "emergency_stop"


class HealthLevel(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


class Mirror(Protocol):
    """Minimal protocol implemented by SLAI SharedMemory."""

    def set(self, key: str, value: Any, *args: Any, **kwargs: Any) -> Any:
        ...


@dataclass(frozen=True, slots=True)
class PoseState:
    x_m: float
    y_m: float
    yaw_rad: float
    speed_mps: float = 0.0
    yaw_rate_rad_s: Optional[float] = None
    confidence: float = 1.0
    timestamp_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite(self.x_m, "x_m"))
        object.__setattr__(self, "y_m", _finite(self.y_m, "y_m"))
        object.__setattr__(self, "yaw_rad", _normalize_yaw(self.yaw_rad))
        object.__setattr__(self, "speed_mps", _finite(self.speed_mps, "speed_mps"))
        object.__setattr__(
            self,
            "yaw_rate_rad_s",
            _optional_finite(self.yaw_rate_rad_s, "yaw_rate_rad_s"),
        )
        object.__setattr__(self, "confidence", _probability(self.confidence, "confidence"))
        object.__setattr__(
            self,
            "timestamp_monotonic",
            _finite(self.timestamp_monotonic, "timestamp_monotonic"),
        )


@dataclass(frozen=True, slots=True)
class GNSSState:
    latitude_deg: Optional[float] = None
    longitude_deg: Optional[float] = None
    altitude_m: Optional[float] = None
    speed_mps: Optional[float] = None
    track_deg: Optional[float] = None
    hdop: Optional[float] = None
    satellites_used: Optional[int] = None
    fix_quality: Optional[int] = None
    valid: bool = False
    timestamp_monotonic: Optional[float] = None

    def __post_init__(self) -> None:
        lat = _optional_finite(self.latitude_deg, "latitude_deg")
        lon = _optional_finite(self.longitude_deg, "longitude_deg")
        if lat is not None and not -90.0 <= lat <= 90.0:
            raise ValueError("latitude_deg must be in [-90, 90]")
        if lon is not None and not -180.0 <= lon <= 180.0:
            raise ValueError("longitude_deg must be in [-180, 180]")
        if self.satellites_used is not None and int(self.satellites_used) < 0:
            raise ValueError("satellites_used must be non-negative")
        if self.fix_quality is not None and int(self.fix_quality) < 0:
            raise ValueError("fix_quality must be non-negative")
        if self.valid and (lat is None or lon is None):
            raise ValueError("valid GNSSState requires latitude_deg and longitude_deg")
        object.__setattr__(self, "latitude_deg", lat)
        object.__setattr__(self, "longitude_deg", lon)
        object.__setattr__(self, "altitude_m", _optional_finite(self.altitude_m, "altitude_m"))

        speed = _optional_finite(self.speed_mps, "gnss.speed_mps")
        if speed is not None and speed < 0.0:
            raise ValueError("gnss.speed_mps must be non-negative")
        object.__setattr__(self, "speed_mps", speed)

        track = _optional_finite(self.track_deg, "track_deg")
        if track is not None and not 0.0 <= track <= 360.0:
            raise ValueError("track_deg must be in [0, 360]")
        object.__setattr__(self, "track_deg", track)

        hdop = _optional_finite(self.hdop, "hdop")
        if hdop is not None and hdop < 0.0:
            raise ValueError("hdop must be non-negative")
        object.__setattr__(self, "hdop", hdop)
        object.__setattr__(
            self,
            "timestamp_monotonic",
            _optional_finite(self.timestamp_monotonic, "gnss.timestamp_monotonic"),
        )


@dataclass(frozen=True, slots=True)
class ObstacleState:
    front_distance_m: Optional[float] = None
    rear_distance_m: Optional[float] = None
    nearest_distance_m: Optional[float] = None
    front_confidence: Optional[float] = None
    rear_confidence: Optional[float] = None
    disagreement: bool = False
    source_health: Mapping[str, str] = field(default_factory=dict)
    timestamp_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        for name in ("front_distance_m", "rear_distance_m", "nearest_distance_m"):
            value = _optional_finite(getattr(self, name), name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in ("front_confidence", "rear_confidence"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _probability(value, name))
        object.__setattr__(self, "source_health", _safe_mapping(self.source_health))
        object.__setattr__(
            self,
            "timestamp_monotonic",
            _finite(self.timestamp_monotonic, "obstacle.timestamp_monotonic"),
        )


@dataclass(frozen=True, slots=True)
class WaypointState:
    x_m: float
    y_m: float
    target_speed_mps: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "x_m", _finite(self.x_m, "waypoint.x_m"))
        object.__setattr__(self, "y_m", _finite(self.y_m, "waypoint.y_m"))
        if self.target_speed_mps is not None:
            speed = _finite(self.target_speed_mps, "waypoint.target_speed_mps")
            if speed < 0.0:
                raise ValueError("waypoint.target_speed_mps must be non-negative")
            object.__setattr__(self, "target_speed_mps", speed)


@dataclass(frozen=True, slots=True)
class RouteState:
    route_id: Optional[str] = None
    waypoints: Tuple[WaypointState, ...] = ()
    active_index: int = 0
    completed: bool = False
    planner_name: Optional[str] = None
    planned_at_monotonic: Optional[float] = None
    goal_tolerance_m: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        points = tuple(self.waypoints)
        object.__setattr__(self, "waypoints", points)
        if self.active_index < 0:
            raise ValueError("active_index must be non-negative")
        if points and self.active_index >= len(points):
            raise ValueError("active_index must refer to an existing waypoint")
        if not points and self.active_index != 0:
            raise ValueError("empty routes require active_index=0")
        if self.planned_at_monotonic is not None:
            object.__setattr__(
                self,
                "planned_at_monotonic",
                _finite(self.planned_at_monotonic, "planned_at_monotonic"),
            )
        if self.goal_tolerance_m is not None:
            tolerance = _finite(self.goal_tolerance_m, "goal_tolerance_m")
            if tolerance < 0.0:
                raise ValueError("goal_tolerance_m must be non-negative")
            object.__setattr__(self, "goal_tolerance_m", tolerance)
        object.__setattr__(self, "metadata", _safe_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class SensorHealthState:
    level: HealthLevel = HealthLevel.UNKNOWN
    available: Tuple[str, ...] = ()
    unavailable: Tuple[str, ...] = ()
    stale: Tuple[str, ...] = ()
    score: Optional[float] = None
    dropped_frames: int = 0
    parse_errors: int = 0
    transport_errors: int = 0
    last_sensor_frame_monotonic: Optional[float] = None
    last_pico_heartbeat_monotonic: Optional[float] = None
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "level", HealthLevel(self.level))
        for name in ("dropped_frames", "parse_errors", "transport_errors"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.score is not None:
            object.__setattr__(self, "score", _probability(self.score, "sensor_health.score"))
        for name in ("last_sensor_frame_monotonic", "last_pico_heartbeat_monotonic"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        object.__setattr__(self, "available", tuple(self.available))
        object.__setattr__(self, "unavailable", tuple(self.unavailable))
        object.__setattr__(self, "stale", tuple(self.stale))
        object.__setattr__(self, "details", _safe_mapping(self.details))


@dataclass(frozen=True, slots=True)
class SafetyState:
    estop_latched: bool = False
    allowed_to_move: bool = False
    degraded: bool = False
    speed_cap_mps: Optional[float] = None
    reasons: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    timestamp_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.speed_cap_mps is not None:
            speed = _finite(self.speed_cap_mps, "speed_cap_mps")
            if speed < 0.0:
                raise ValueError("speed_cap_mps must be non-negative")
            object.__setattr__(self, "speed_cap_mps", speed)
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(
            self,
            "timestamp_monotonic",
            _finite(self.timestamp_monotonic, "safety.timestamp_monotonic"),
        )


@dataclass(frozen=True, slots=True)
class ActuationState:
    throttle: float = 0.0
    steering: float = 0.0
    requested_speed_mps: Optional[float] = None
    source: str = "none"
    status: str = "neutral"
    timestamp_monotonic: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        throttle = _finite(self.throttle, "throttle")
        steering = _finite(self.steering, "steering")
        if not -1.0 <= throttle <= 1.0:
            raise ValueError("throttle must be in [-1, 1]")
        if not -1.0 <= steering <= 1.0:
            raise ValueError("steering must be in [-1, 1]")
        object.__setattr__(self, "throttle", throttle)
        object.__setattr__(self, "steering", steering)
        if self.requested_speed_mps is not None:
            object.__setattr__(
                self,
                "requested_speed_mps",
                _finite(self.requested_speed_mps, "requested_speed_mps"),
            )
        object.__setattr__(
            self,
            "timestamp_monotonic",
            _finite(self.timestamp_monotonic, "actuation.timestamp_monotonic"),
        )


@dataclass(frozen=True, slots=True)
class AutonomyState:
    mode: OperatingMode = OperatingMode.STOPPED
    run_id: Optional[str] = None
    goal_id: Optional[str] = None
    cycle: Optional[int] = None
    planner_status: Optional[str] = None
    last_plan_monotonic: Optional[float] = None
    last_control_cycle_monotonic: Optional[float] = None
    last_recovery_monotonic: Optional[float] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", OperatingMode(self.mode))
        if self.cycle is not None and int(self.cycle) < 0:
            raise ValueError("cycle must be non-negative")
        for name in (
            "last_plan_monotonic",
            "last_control_cycle_monotonic",
            "last_recovery_monotonic",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name))
        object.__setattr__(self, "metadata", _safe_mapping(self.metadata))


@dataclass(frozen=True, slots=True)
class WorldEvent:
    sequence: int
    event_type: str
    timestamp_wall: float
    timestamp_monotonic: float
    severity: str = "info"
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("event sequence must be >= 1")
        if not self.event_type.strip():
            raise ValueError("event_type must be non-empty")
        object.__setattr__(self, "payload", _safe_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    revision: int
    timestamp_wall: float
    timestamp_monotonic: float
    pose: Optional[PoseState] = None
    gnss: GNSSState = field(default_factory=GNSSState)
    obstacles: ObstacleState = field(default_factory=ObstacleState)
    route: RouteState = field(default_factory=RouteState)
    sensor_health: SensorHealthState = field(default_factory=SensorHealthState)
    safety: SafetyState = field(default_factory=SafetyState)
    actuation: ActuationState = field(default_factory=ActuationState)
    autonomy: AutonomyState = field(default_factory=AutonomyState)
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.revision < 0:
            raise ValueError("revision must be non-negative")
        object.__setattr__(self, "context", _safe_mapping(self.context))

    def to_dict(self) -> Dict[str, Any]:
        return _serialize(self)

    def age_seconds(self, now_monotonic: Optional[float] = None) -> float:
        now = time.monotonic() if now_monotonic is None else _finite(
            now_monotonic, "now_monotonic"
        )
        return max(0.0, now - self.timestamp_monotonic)


def _serialize(value: Any) -> Any:
    """Serialize without deepcopying immutable MappingProxyType fields."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _serialize(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


class WorldModel:
    """Thread-safe authoritative holder of the latest typed vehicle state."""

    DEFAULT_MIRROR_KEY = "robocar:world_state"

    def __init__(
        self,
        *,
        mirror: Optional[Mirror] = None,
        mirror_key: str = DEFAULT_MIRROR_KEY,
        event_capacity: int = 256,
    ) -> None:
        if event_capacity < 1:
            raise ValueError("event_capacity must be >= 1")
        if not str(mirror_key).strip():
            raise ValueError("mirror_key must be non-empty")
        self._lock = threading.RLock()
        self._mirror = mirror
        self._mirror_key = str(mirror_key)
        self._events: Deque[WorldEvent] = deque(maxlen=int(event_capacity))
        self._event_sequence = 0
        now_mono = time.monotonic()
        self._snapshot = WorldSnapshot(
            revision=0,
            timestamp_wall=time.time(),
            timestamp_monotonic=now_mono,
        )

    def snapshot(self) -> WorldSnapshot:
        with self._lock:
            return self._snapshot

    def snapshot_dict(self) -> Dict[str, Any]:
        return self.snapshot().to_dict()

    def events(self, limit: Optional[int] = None) -> Tuple[WorldEvent, ...]:
        with self._lock:
            data = tuple(self._events)
        return data if limit is None else data[-max(0, int(limit)):]

    def update(
        self,
        *,
        pose: Optional[PoseState] = None,
        gnss: Optional[GNSSState] = None,
        obstacles: Optional[ObstacleState] = None,
        route: Optional[RouteState] = None,
        sensor_health: Optional[SensorHealthState] = None,
        safety: Optional[SafetyState] = None,
        actuation: Optional[ActuationState] = None,
        autonomy: Optional[AutonomyState] = None,
        context: Optional[Mapping[str, Any]] = None,
        event_type: Optional[str] = None,
        event_payload: Optional[Mapping[str, Any]] = None,
        event_severity: str = "info",
    ) -> WorldSnapshot:
        """Atomically replace supplied sub-states and return the new snapshot."""

        with self._lock:
            previous = self._snapshot
            now_mono = time.monotonic()
            new = replace(
                previous,
                revision=previous.revision + 1,
                timestamp_wall=time.time(),
                timestamp_monotonic=now_mono,
                pose=previous.pose if pose is None else pose,
                gnss=previous.gnss if gnss is None else gnss,
                obstacles=previous.obstacles if obstacles is None else obstacles,
                route=previous.route if route is None else route,
                sensor_health=(
                    previous.sensor_health if sensor_health is None else sensor_health
                ),
                safety=previous.safety if safety is None else safety,
                actuation=previous.actuation if actuation is None else actuation,
                autonomy=previous.autonomy if autonomy is None else autonomy,
                context=previous.context if context is None else _safe_mapping(context),
            )
            self._validate_transition(previous, new)
            self._snapshot = new
            if event_type:
                self._append_event_locked(
                    event_type,
                    payload=event_payload,
                    severity=event_severity,
                )

        self._mirror_snapshot(new)
        return new

    def record_event(
        self,
        event_type: str,
        *,
        payload: Optional[Mapping[str, Any]] = None,
        severity: str = "info",
    ) -> WorldEvent:
        with self._lock:
            event = self._append_event_locked(event_type, payload=payload, severity=severity)
        return event

    def _append_event_locked(
        self,
        event_type: str,
        *,
        payload: Optional[Mapping[str, Any]],
        severity: str,
    ) -> WorldEvent:
        self._event_sequence += 1
        event = WorldEvent(
            sequence=self._event_sequence,
            event_type=str(event_type),
            timestamp_wall=time.time(),
            timestamp_monotonic=time.monotonic(),
            severity=str(severity),
            payload=_safe_mapping(payload),
        )
        self._events.append(event)
        return event

    @staticmethod
    def _validate_transition(previous: WorldSnapshot, current: WorldSnapshot) -> None:
        if current.revision != previous.revision + 1:
            raise WorldModelError("world revision must advance exactly once")
        if current.timestamp_monotonic < previous.timestamp_monotonic:
            raise WorldModelError("world snapshot monotonic time cannot move backwards")
        if current.safety.estop_latched and current.safety.allowed_to_move:
            raise WorldModelError(
                "estop_latched and allowed_to_move cannot both be true"
            )

    def _mirror_snapshot(self, snapshot: WorldSnapshot) -> None:
        if self._mirror is None:
            return
        try:
            self._mirror.set(self._mirror_key, snapshot.to_dict())
        except Exception:
            # Mirroring is secondary.  The authoritative in-process state must
            # remain usable if telemetry/shared persistence is degraded.
            return


__all__ = [
    "WorldModelError",
    "OperatingMode",
    "HealthLevel",
    "PoseState",
    "GNSSState",
    "ObstacleState",
    "WaypointState",
    "RouteState",
    "SensorHealthState",
    "SafetyState",
    "ActuationState",
    "AutonomyState",
    "WorldEvent",
    "WorldSnapshot",
    "WorldModel",
]
