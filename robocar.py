"""Production RoboCar composition root and SLAI integration boundary.

The module deliberately separates four control domains:

1. Physical/hard boundary
   ``SensorBus`` / ``WheelEncoder`` / ``MotionController``.
2. Deterministic vehicle domain
   local safety, typed ``WorldModel``, trajectory control, KPI accounting,
   watchdogs, local A* fallback, and bounded adaptation guardrails.
3. SLAI outer autonomy
   the current ``AutonomousControlLoop`` through
   ``modules.slai_autonomy`` (reason -> plan -> authorize -> execute -> evaluate).
4. Reliability/assessment
   SLAI HandlerAgent, ObservabilityAgent, and EvaluationAgent.

Safety invariants
-----------------
* Emergency-stop and watchdog-critical paths command the hardware boundary
  before invoking any SLAI recovery or reasoning component.
* A physical ExecutionAgent is created with ``robot=RoboCarRobotAdapter`` and
  reused by the outer autonomy adapter.  A second generic execution instance is
  never created for the car.
* Only Ackermann-compatible robot actions are registered.  Differential-drive
  Motor/Spin/Navigate actions are intentionally not exposed.
* Non-zero Ackermann throttle is bounded in time by default.  The current SLAI
  AckermannAction leaves throttle active when ``duration == 0``; this module
  therefore rejects accidental persistent throttle unless the caller explicitly
  opts in.
* Autonomous physical operation can be configured to require calibrated local
  collision/freshness watchdog values.  Missing values are never fabricated.
* Observability is non-authoritative: telemetry failure cannot convert a denied
  or stopped vehicle into motion.
* HandlerAgent recovery occurs only after a local safe-stop attempt.
* Vehicle KPI semantics are computed by ``VehicleKPITracker``.  The current
  generic SLAI EvaluationAgent is retained as advisory/general evaluation and
  is wrapped so unrelated domain logic cannot silently become the vehicle's
  physical completion criterion.

The module does not implement localization or GNSS-to-local-coordinate
transforms.  Callers must feed validated ``PoseState`` and ``GNSSState`` values
from the corresponding deterministic localization/geodesy layer when those
modules are available.
"""

from __future__ import annotations

import heapq
import math
import threading
import time
import uuid

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .main_sensor import *
from .motion_controller import *
from .wheel_encoder import WheelEncoder
from .modules.edt2d import *
from .modules.world_model import *
from .modules.trajectory_control import *
from .modules.kpi_tracker import *
from .modules.watchdog import *
from .modules.adaptation_guard import *
# slai_autonomy is intentionally imported explicitly rather than through
# modules.__init__.py.  The deterministic modules remain importable without the
# wider SLAI agent graph, while this composition root intentionally has SLAI.
from .modules.slai_autonomy import build_robocar_autonomy_loop
from .utils.config_loader import get_config_section, load_global_config
from .utils.rc_errors import *
from .utils.rc_helpers import *

# RoboCar is expected to live at SLAI/RoboCar.  These are absolute imports from
# the parent SLAI repository, not ``..src`` relatives.
from src.agents.agent_factory import AgentFactory # type: ignore
from src.agents.collaborative.shared_memory import SharedMemory # type: ignore
from src.agents.execution.actions.robot_actions import ( # type: ignore
    AckermannAction,
    SensorReadAction,
    StopAction,
)
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]


logger = get_logger("SLAI AI RC Car")
printer = PrettyPrinter()


# ---------------------------------------------------------------------------
# Compatibility/public constants
# ---------------------------------------------------------------------------

MEM_FILE = "robot_memory.pkl"
DEFAULT_ROBOCAR_CONFIG = Path(__file__).resolve().parent / "configs" / "rc_configs.yaml"

K_MAP_LATEST = "map:latest"
K_DETECTIONS_SIGNS = "detections:signs"
K_GOAL_CURRENT = "goal:current"
K_PLAN_CURRENT = "plan:current"
K_ROUTE_TRAVELED = "route:traveled"
K_SAFETY_STATE = "safety:state"
K_DIRECTIVES = "reasoning:directives"
K_POSE_ESTIMATE = "pose:estimate"
K_CONFIG = "robocar:config"
K_SENSOR_LATEST = "sensors:latest"
K_ENCODER_TICKS = "sensors:encoder:ticks_total"
K_ENCODER_SPEED = "sensors:encoder:speed_mps"
K_ULTRA_FRONT = "sensors:ultra:front_m"
K_ULTRA_REAR = "sensors:ultra:rear_m"
K_TOF_MM = "sensors:tof:mm"
K_BATTERY_VOLT = "power:vbat"
K_BATTERY_STATE = "power:state"

K_WORLD_STATE = "robocar:world_state"
K_KPI_LATEST = "robocar:kpi:latest"
K_WATCHDOG_LATEST = "robocar:watchdog:latest"
K_AUTONOMY_LAST = "robocar:autonomy:last_run"
K_OBSERVABILITY_LATEST = "robocar:observability:last_report"
K_HANDLER_LATEST = "robocar:handler:last_recovery"
K_EVALUATION_LATEST = "robocar:evaluation:last_report"
K_ADAPTATION_LAST = "robocar:adaptation:last_record"

_REGISTERED_ROBOT_ACTIONS = {
    "ackermann": AckermannAction,
    "stop": StopAction,
    "sensor_read": SensorReadAction,
}


# ---------------------------------------------------------------------------
# Small local helpers
# ---------------------------------------------------------------------------


def _mapping(value: Any) -> Dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _optional_positive(value: Any) -> Optional[float]:
    if value is None:
        return None
    converted = optional_finite_float(value, minimum=0.0)
    if converted is None or converted <= 0.0:
        raise ValueError(f"Expected a finite positive value, got {value!r}")
    return converted


def _strict_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    if isinstance(value, int) and not isinstance(value, bool) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"Expected a boolean-compatible value, got {value!r}")


def _status_success(result: Mapping[str, Any]) -> bool:
    status = str(result.get("status", "")).strip().lower()
    return (
        result.get("success") is True
        or result.get("passed") is True
        or result.get("completed") is True
        or status
        in {
            "ok",
            "success",
            "succeeded",
            "pass",
            "passed",
            "complete",
            "completed",
            "allow",
            "approved",
            "normal",
        }
    )


def _serialize(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _serialize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _serialize(to_dict())
        except Exception:
            pass
    try:
        return _serialize(asdict(value))
    except Exception:
        return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _normalize_action_name(item: Mapping[str, Any]) -> str:
    return str(item.get("name") or item.get("action_name") or "").strip().lower()


# ---------------------------------------------------------------------------
# Compatibility geometry types and local A* map schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pose2D:
    """Backward-compatible lightweight pose type.

    New vehicle state should preferentially use ``modules.world_model.PoseState``.
    """

    x: float
    y: float
    theta: float
    v: float = 0.0

    def __post_init__(self) -> None:
        if not all(is_finite_number(v) for v in (self.x, self.y, self.theta, self.v)):
            raise ValueError("Pose2D values must be finite")

    def to_pose_state(self, *, confidence: float = 1.0) -> PoseState:
        return PoseState(
            x_m=float(self.x),
            y_m=float(self.y),
            yaw_rad=float(self.theta),
            speed_mps=float(self.v),
            confidence=require_probability(confidence, "pose.confidence"),
        )


@dataclass(frozen=True, slots=True)
class Waypoint:
    """Backward-compatible metric waypoint."""

    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(is_finite_number(v) for v in (self.x, self.y)):
            raise ValueError("Waypoint values must be finite")

    def to_waypoint_state(self, *, target_speed_mps: Optional[float] = None) -> WaypointState:
        return WaypointState(
            x_m=float(self.x),
            y_m=float(self.y),
            target_speed_mps=target_speed_mps,
        )


@dataclass(slots=True)
class OccupancyGrid:
    """Minimal occupancy-grid schema compatible with ``modules.edt2d``.

    Values >= 50 are occupied by the existing repository convention; negative
    values are unknown.  The local fallback planner treats unknown space as
    occupied unless the caller explicitly overrides that behavior.
    """

    width: int
    height: int
    resolution: float
    grid: Sequence[int]
    origin_x: float = 0.0
    origin_y: float = 0.0

    def __post_init__(self) -> None:
        self.width = int(self.width)
        self.height = int(self.height)
        self.resolution = float(self.resolution)
        self.origin_x = float(self.origin_x)
        self.origin_y = float(self.origin_y)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("OccupancyGrid width/height must be positive")
        if not math.isfinite(self.resolution) or self.resolution <= 0.0:
            raise ValueError("OccupancyGrid resolution must be finite and positive")
        flattened = list(self.grid)
        if len(flattened) != self.width * self.height:
            raise ValueError(
                f"OccupancyGrid requires {self.width * self.height} cells, "
                f"got {len(flattened)}"
            )
        self.grid = flattened

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "origin_x": self.origin_x,
            "origin_y": self.origin_y,
            "grid": list(self.grid),
        }

    def in_bounds(self, cell: Tuple[int, int]) -> bool:
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height

    def index(self, cell: Tuple[int, int]) -> int:
        x, y = cell
        if not self.in_bounds(cell):
            raise IndexError(f"Cell out of bounds: {cell}")
        return y * self.width + x

    def world_to_cell(self, x: float, y: float) -> Tuple[int, int]:
        return (
            int(math.floor((float(x) - self.origin_x) / self.resolution)),
            int(math.floor((float(y) - self.origin_y) / self.resolution)),
        )

    def cell_to_world(self, cell: Tuple[int, int]) -> Waypoint:
        x, y = cell
        if not self.in_bounds(cell):
            raise IndexError(f"Cell out of bounds: {cell}")
        return Waypoint(
            self.origin_x + (x + 0.5) * self.resolution,
            self.origin_y + (y + 0.5) * self.resolution,
        )


# ---------------------------------------------------------------------------
# Deterministic local safety
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SafetyDecision:
    allowed: bool
    throttle: float
    steering: float
    reasons: Tuple[str, ...]
    warnings: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class SafetyManager:
    """Dependency-light safety gate immediately upstream of the actuator path.

    Only configured constraints are enforced.  No collision distance, sensor
    freshness window, braking factor, or derating coefficient is invented.
    Where both ultrasonic and ToF front ranges are available, the smaller valid
    distance is used as a conservative stop observation; no probabilistic sensor
    confidence is fabricated.
    """

    def __init__(self, config: Mapping[str, Any], shared_memory: SharedMemory) -> None:
        self.config = dict(config)
        self.shared_memory = shared_memory
        power = get_config_section("power", self.config)
        robocar = get_config_section("robocar", self.config)

        self.v_warn = require_finite_float(
            power.get("v_warn"), "power.v_warn", minimum=0.0
        )
        self.v_cutback = require_finite_float(
            power.get("v_cutback"), "power.v_cutback", minimum=0.0
        )
        self.v_critical = require_finite_float(
            power.get("v_critical"), "power.v_critical", minimum=0.0
        )
        if not self.v_critical <= self.v_cutback <= self.v_warn:
            raise ValueError("Expected power.v_critical <= v_cutback <= v_warn")

        self.front_stop_distance_m = optional_finite_float(
            robocar.get("front_stop_distance_m"), minimum=0.0
        )
        self.sensor_max_age_s = optional_finite_float(
            robocar.get("sensor_max_age_s"), minimum=0.0
        )

    def battery_state(self, voltage: Optional[float]) -> str:
        if voltage is None:
            return "unknown"
        if voltage <= self.v_critical:
            return "critical"
        if voltage <= self.v_cutback:
            return "cutback"
        if voltage <= self.v_warn:
            return "warning"
        return "normal"

    @staticmethod
    def nearest_front_distance(reading: Optional[SensorReading]) -> Optional[float]:
        if reading is None:
            return None
        candidates: list[float] = []
        ultra = optional_finite_float(reading.ultra_front_m, minimum=0.0)
        if ultra is not None:
            candidates.append(ultra)
        tof_mm = optional_int(reading.tof_mm, minimum=0)
        if tof_mm is not None:
            candidates.append(float(tof_mm) / 1000.0)
        return min(candidates) if candidates else None

    def authorize_command(
        self,
        throttle: Any,
        steering: Any,
        *,
        reading: Optional[SensorReading],
        speed_mps: Optional[float] = None,
    ) -> SafetyDecision:
        throttle_value = normalize_signed_command(throttle, "RoboCarThrottle")
        steering_value = normalize_signed_command(steering, "RoboCarSteering")
        reasons: list[str] = []
        warnings: list[str] = []

        state = self.shared_memory.get(K_SAFETY_STATE, default={})
        state = dict(state) if isinstance(state, Mapping) else {}
        directives = self.shared_memory.get(K_DIRECTIVES, default={})
        directives = dict(directives) if isinstance(directives, Mapping) else {}

        if state.get("estop") is True:
            reasons.append("emergency_stop_latched")

        full_stop_until = optional_finite_float(directives.get("full_stop_until"))
        if full_stop_until is not None and time.time() < full_stop_until:
            reasons.append("reasoning_full_stop_directive")

        if self.sensor_max_age_s is not None:
            if reading is None:
                reasons.append("sensor_frame_missing")
            elif max(0.0, time.time() - reading.t) > self.sensor_max_age_s:
                reasons.append("sensor_frame_stale")

        voltage = reading.vbat if reading is not None else None
        power_state = self.battery_state(voltage)
        if power_state == "critical" and abs(throttle_value) > 1e-6:
            reasons.append("battery_voltage_critical")
        elif power_state in {"cutback", "warning"}:
            warnings.append(f"battery_voltage_{power_state}")

        front = self.nearest_front_distance(reading)
        if (
            self.front_stop_distance_m is not None
            and front is not None
            and front <= self.front_stop_distance_m
            and throttle_value > 0.0
        ):
            reasons.append("front_obstacle_inside_configured_stop_distance")

        speed_cap = optional_finite_float(state.get("speed_cap"), minimum=0.0)
        if (
            speed_cap is not None
            and speed_mps is not None
            and speed_mps >= speed_cap
            and throttle_value > 0.0
        ):
            reasons.append("configured_speed_cap_reached")

        directive_limit = optional_finite_float(
            directives.get("limit_speed"), minimum=0.0
        )
        if (
            directive_limit is not None
            and speed_mps is not None
            and speed_mps >= directive_limit
            and throttle_value > 0.0
        ):
            reasons.append("reasoning_speed_limit_reached")

        allowed = not reasons
        return SafetyDecision(
            allowed=allowed,
            throttle=throttle_value if allowed else 0.0,
            steering=steering_value,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )


# ---------------------------------------------------------------------------
# Backward-compatible Pure Pursuit façade
# ---------------------------------------------------------------------------


class PurePursuit:
    """Compatibility wrapper around ``PurePursuitController``.

    New control code should use ``RoboCar.trajectory_controller`` directly.
    The wrapper avoids maintaining a duplicate pure-pursuit implementation.
    """

    def __init__(
        self,
        lookahead_m: float,
        wheelbase_m: float,
        max_steer_rad: float,
    ) -> None:
        self._controller = PurePursuitController(
            lookahead_m=lookahead_m,
            wheelbase_m=wheelbase_m,
            max_steer_rad=max_steer_rad,
            # Preserve legacy behavior: this compatibility wrapper is geometry
            # only and does not declare a goal reached before the final point.
            goal_tolerance_m=0.0,
        )

    @property
    def lookahead_m(self) -> float:
        return self._controller.lookahead_m

    @property
    def wheelbase_m(self) -> float:
        return self._controller.wheelbase_m

    @property
    def max_steer_rad(self) -> float:
        return self._controller.max_steer_rad

    def compute_steering(self, pose: Pose2D, path: Sequence[Waypoint]) -> float:
        if not path:
            return 0.0
        state = pose.to_pose_state(confidence=1.0)
        points = tuple(point.to_waypoint_state() for point in path)
        return self._controller.compute(state, points).steering


# ---------------------------------------------------------------------------
# SLAI Evaluation bridge
# ---------------------------------------------------------------------------


class RoboCarEvaluationBridge:
    """Scope adapter around the current generic SLAI EvaluationAgent.

    ``EvaluationAgent.execute_validation_cycle`` currently includes generic
    cross-domain evaluators, including financial-health logic, while RoboCar
    vehicle semantics live in ``VehicleKPITracker``.  This bridge does not erase
    or falsify the generic result; it preserves it under ``slai_evaluation`` and
    reports whether the evaluation *ran* successfully.

    No vehicle KPI is declared passed merely because a value exists.  Until
    explicit vehicle KPI thresholds are configured/implemented, the bridge uses
    ``completed`` rather than an invented ``passed`` decision.
    """

    name = "robocar_evaluation_bridge"

    def __init__(self, raw_agent: Any, car: "RoboCar") -> None:
        self.raw_agent = raw_agent
        self.car = car
        self._last_report: Optional[Dict[str, Any]] = None
        self._calls = 0

    def execute_validation_cycle(self, params: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(params, Mapping):
            raise TypeError("RoboCar evaluation params must be a mapping")

        self._calls += 1
        payload = dict(params)
        kpis = self.car.kpi_tracker.snapshot().to_dict()
        world = self.car.world_model.snapshot().to_dict()
        payload.setdefault("robocar_kpis", kpis)
        payload.setdefault("world_state", world)

        raw_method = getattr(self.raw_agent, "execute_validation_cycle", None)
        if not callable(raw_method):
            report = {
                "status": "failed",
                "completed": False,
                "reason": "evaluation_agent_missing_execute_validation_cycle",
                "robocar_kpis": kpis,
            }
            self._last_report = report
            self.car.shared_memory.set(K_EVALUATION_LATEST, report)
            return report

        try:
            raw = raw_method(payload)
        except Exception as exc:
            report = {
                "status": "failed",
                "completed": False,
                "reason": "slai_evaluation_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "robocar_kpis": kpis,
            }
            self._last_report = report
            self.car.shared_memory.set(K_EVALUATION_LATEST, report)
            return report

        if not isinstance(raw, Mapping):
            report = {
                "status": "failed",
                "completed": False,
                "reason": "slai_evaluation_non_mapping_result",
                "raw_type": type(raw).__name__,
                "robocar_kpis": kpis,
            }
            self._last_report = report
            self.car.shared_memory.set(K_EVALUATION_LATEST, report)
            return report

        raw_dict = dict(raw)
        cycle_failed = raw_dict.get("cycle_failed") is True
        execution = _mapping(payload.get("control_loop_execution"))
        execution_ok = True if not execution else _status_success(execution)
        completed = bool(execution_ok and not cycle_failed and not raw_dict.get("error"))

        report = {
            "status": "completed" if completed else "failed",
            "completed": completed,
            "scope": "robocar_vehicle_evaluation_bridge",
            "robocar_kpis": kpis,
            "world_revision": world.get("revision"),
            "slai_evaluation_status": raw_dict.get("status"),
            "slai_evaluation": _serialize(raw_dict),
            "vehicle_kpi_thresholds_applied": False,
            "vehicle_kpi_threshold_note": (
                "Vehicle KPI semantics are measured locally, but no repository "
                "vehicle KPI pass/fail thresholds are configured; no pass result "
                "is fabricated."
            ),
        }
        self._last_report = report
        self.car.shared_memory.set(K_EVALUATION_LATEST, report)
        return report

    def health(self) -> Dict[str, Any]:
        raw_health = None
        for method_name in (
            "get_overall_system_health",
            "health",
            "health_check",
            "get_health_report",
        ):
            method = getattr(self.raw_agent, method_name, None)
            if not callable(method):
                continue
            try:
                raw_health = method()
            except Exception as exc:
                raw_health = {"status": "degraded", "error": str(exc)}
            break
        return {
            "status": "ok" if self.raw_agent is not None else "failed",
            "calls": self._calls,
            "last_report": self._last_report,
            "raw_agent_health": _serialize(raw_health),
        }


# ---------------------------------------------------------------------------
# Safe HandlerAgent recovery target for non-agent vehicle failures
# ---------------------------------------------------------------------------


class RoboCarRecoveryTarget:
    """Structural recovery target for watchdog/sensor failures.

    HandlerAgent requires a target agent for its recovery pipeline.  This target
    exposes only fail-safe recovery operations; it never commands non-zero
    throttle and never clears an actuator fault or e-stop automatically.
    """

    name = "robocar_recovery"

    def __init__(self, car: "RoboCar") -> None:
        self.car = car
        self._lightweight = False

    def use_lightweight_mode(self, enabled: bool) -> None:
        self._lightweight = bool(enabled)

    def perform_task(self, task_data: Any) -> Dict[str, Any]:
        task = _mapping(task_data)
        operation = str(task.get("operation", "safe_degraded_recovery")).strip().lower()
        events = {
            str(item).strip().lower()
            for item in task.get("watchdog_events", [])
            if str(item).strip()
        }

        # All recovery paths are entered after the physical stop boundary.
        self.car._enter_degraded_mode(
            reason=f"handler_recovery:{operation}",
            event_type="recovery.target_invoked",
        )

        if "actuator_fault" in events or operation == "actuator_fault":
            return {
                "status": "failed",
                "recovered": False,
                "reason": "actuator_fault_requires_manual_inspection",
                "lightweight_mode": self._lightweight,
            }

        if events.intersection({"sensor_frame_stale", "pico_heartbeat_stale"}) or operation == "recover_sensor_transport":
            try:
                self.car.sensor_bus.stop()
                self.car.sensor_bus.start()
            except Exception as exc:
                return {
                    "status": "failed",
                    "recovered": False,
                    "reason": "sensor_transport_restart_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            return {
                "status": "degraded",
                "recovered": True,
                "reason": "sensor_transport_restarted_vehicle_remains_stopped",
                "sensor_bus": self.car.sensor_bus.health(),
            }

        # Deadline/GNSS/planner faults have no safe generic automatic repair in
        # the current RoboCar repository.  A stopped degraded state is itself a
        # valid fail-operational recovery outcome, not permission to resume.
        return {
            "status": "degraded",
            "recovered": True,
            "reason": "safe_degraded_state_established_no_automatic_rearm",
            "watchdog_events": sorted(events),
            "lightweight_mode": self._lightweight,
        }


# ---------------------------------------------------------------------------
# Physical robot adapter used by SLAI AckermannAction
# ---------------------------------------------------------------------------


class RoboCarRobotAdapter:
    """Structural adapter consumed by SLAI's existing Ackermann robot action."""

    def __init__(self, owner: "RoboCar") -> None:
        self._owner = owner
        self._lock = threading.RLock()
        self._steering = 0.0
        self._throttle = 0.0

    def set_steering(self, angle: float) -> bool:
        steering = normalize_signed_command(angle, "RoboCarAdapterSteering")
        with self._lock:
            # AckermannAction invokes steering before throttle.  Neutralize the
            # ESC here so the steering operation cannot inherit stale throttle.
            decision = self._owner.local_safety.authorize_command(
                0.0,
                steering,
                reading=self._owner.sensor_bus.latest(),
                speed_mps=self._owner.encoder.get_speed(),
            )
            self._owner._publish_local_safety(decision)
            if not decision.allowed:
                self._owner._safe_hardware_stop("adapter_steering_denied")
                return False

            result = self._owner.motion.send(0.0, decision.steering)
            self._steering = decision.steering
            self._throttle = 0.0
            self._owner._update_actuation_state(
                throttle=0.0,
                steering=decision.steering,
                source="slai.execution.ackermann.set_steering",
                status="applied",
            )
            return result.get("ok") is True

    def set_throttle(self, speed: float) -> bool:
        throttle = normalize_signed_command(speed, "RoboCarAdapterThrottle")
        with self._lock:
            decision = self._owner.local_safety.authorize_command(
                throttle,
                self._steering,
                reading=self._owner.sensor_bus.latest(),
                speed_mps=self._owner.encoder.get_speed(),
            )
            self._owner._publish_local_safety(decision)
            if not decision.allowed:
                self._owner._safe_hardware_stop("adapter_throttle_denied")
                return False

            result = self._owner.motion.send(decision.throttle, decision.steering)
            self._throttle = decision.throttle
            self._owner._update_actuation_state(
                throttle=decision.throttle,
                steering=decision.steering,
                source="slai.execution.ackermann.set_throttle",
                status="applied",
            )
            return result.get("ok") is True

    def stop(self) -> bool:
        with self._lock:
            result = self._owner.motion.stop()
            self._throttle = 0.0
            self._steering = 0.0
            self._owner._set_actuation_neutral(source="slai.execution.stop")
            return result.get("ok") is True

    def get_sensor_value(self, sensor_name: str) -> Any:
        return self._owner.get_sensor_value(sensor_name)

    def get_pose(self) -> Tuple[float, float, float]:
        snapshot = self._owner.world_model.snapshot()
        if snapshot.pose is not None:
            return (
                snapshot.pose.x_m,
                snapshot.pose.y_m,
                snapshot.pose.yaw_rad,
            )

        # Compatibility fallback for callers that still publish the historical
        # shared-memory pose key before the localization module migrates.
        raw = self._owner.shared_memory.get(K_POSE_ESTIMATE)
        if not isinstance(raw, Mapping):
            raise SensorError(
                "pose_estimate",
                raw,
                ("finite x/y/theta", "finite x/y/theta"),
            )
        # Keep the validated values explicitly typed for type checkers: values
        # read from a generic Mapping are otherwise inferred as possibly None.
        x: Any = raw.get("x")
        y: Any = raw.get("y")
        theta: Any = raw.get("theta")
        values = (x, y, theta)
        if not all(is_finite_number(value) for value in values):
            raise SensorError("pose_estimate", values, ("finite", "finite"))
        return float(x), float(y), float(theta)


# ---------------------------------------------------------------------------
# RoboCar composition root
# ---------------------------------------------------------------------------

class RoboCar:
    """Physical RoboCar composition root for the current SLAI runtime.

    The class owns hardware-facing objects and deterministic vehicle state.  The
    SLAI ``AutonomousControlLoop`` remains the sole outer autonomy owner; this
    class exposes explicit mission/control/service methods rather than creating a
    second hidden autonomous forever-loop.
    """

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        sensor_port: Optional[str] = None,
        allow_simulation: bool = False,
        shared_memory: Optional[SharedMemory] = None,
        agent_factory: Optional[AgentFactory] = None,
        eager_support_agents: bool = True,
    ) -> None:
        # The current repository config loader still points to rc_config.yaml
        # (singular), while the live file is rc_configs.yaml.  Resolve the known
        # package default here so RoboCar itself remains runnable without
        # mutating unrelated loader code in this replacement.
        resolved_config = (
            str(DEFAULT_ROBOCAR_CONFIG)
            if config_path is None
            else str(Path(config_path).expanduser())
        )
        self.config_path = resolved_config
        self.config = load_global_config(resolved_config)

        hardware = get_config_section("hardware", self.config)
        serial_cfg = (
            hardware.get("pico_serial", {})
            if isinstance(hardware.get("pico_serial"), Mapping)
            else {}
        )
        lighting_cfg = get_config_section("lighting", self.config)
        port = (
            sensor_port
            if sensor_port is not None
            else serial_cfg.get("port", "auto")
        )
        baud = optional_int(
            serial_cfg.get("baud"),
            minimum=1,
        ) or 115200

        self.allow_simulation = bool(allow_simulation)
        self.shared_memory = shared_memory if shared_memory is not None else SharedMemory()
        self.agent_factory = agent_factory if agent_factory is not None else AgentFactory()
        self._owns_memory = shared_memory is None
        self._owns_factory = agent_factory is None
        self._started = False
        self._lifecycle_lock = threading.RLock()
        self._control_lock = threading.RLock()
        self._agents: Dict[str, Any] = {}
        self._raw_agents: Dict[str, Any] = {}
        self._agent_errors: Dict[str, str] = {}
        self._last_error: Optional[str] = None
        self._runtime_started_monotonic: Optional[float] = None
        self._autonomy_started_monotonic: Optional[float] = None
        self._last_sensor_frame_monotonic: Optional[float] = None
        self._last_pico_heartbeat_monotonic: Optional[float] = None
        self._last_control_cycle_duration_s: Optional[float] = None
        self._last_gnss_fix_monotonic: Optional[float] = None
        self._gnss_seen = False
        self._last_handler_result: Optional[Dict[str, Any]] = None
        self._last_observability_result: Optional[Dict[str, Any]] = None
        self._last_autonomy_result: Optional[Dict[str, Any]] = None

        # -------------------------- hardware boundary ----------------------
        self.motion = MotionController(
            config=self.config,
            allow_simulation=self.allow_simulation,
        )
        self.speed_controller = PIDSpeedController(config=self.config)
        self.encoder = WheelEncoder(config=self.config)
        self.sensor_bus = SensorBus(
            port=str(port) if port is not None else "auto",
            baud=baud,
            allow_simulation=self.allow_simulation,
            lighting_config=lighting_cfg,
        )
        minimum_turn_angle_deg = optional_finite_float(
            lighting_cfg.get("turn_detection_min_angle_deg", 10.0),
            minimum=0.0,
            maximum=180.0,
        )
        
        if (
            minimum_turn_angle_deg is None
            or minimum_turn_angle_deg <= 0.0
        ):
            raise ValueError(
                "lighting.turn_detection_min_angle_deg "
                "must be within (0, 180]"
            )
        
        self._turn_detection_min_angle_rad = math.radians(minimum_turn_angle_deg)

        # -------------------------- deterministic state --------------------
        self.world_model = WorldModel(
            mirror=self.shared_memory,
            mirror_key=K_WORLD_STATE,
        )
        self.local_safety = SafetyManager(self.config, self.shared_memory)
        self.robot_adapter = RoboCarRobotAdapter(self)

        robocar_cfg = get_config_section("robocar", self.config)
        motion_cfg = get_config_section("motion", self.config)
        lookahead_m = require_finite_float(
            robocar_cfg.get("lookahead"), "robocar.lookahead", minimum=0.0
        )
        wheelbase_m = require_finite_float(
            robocar_cfg.get("wheelbase"), "robocar.wheelbase", minimum=0.0
        )
        max_steer_rad = require_finite_float(
            motion_cfg.get("servo_max_angle_rad"),
            "motion.servo_max_angle_rad",
            minimum=0.0,
        )
        goal_tolerance = optional_finite_float(
            robocar_cfg.get("goal_tolerance_m"), minimum=0.0
        )
        if goal_tolerance is None:
            # Use the deterministic module's established default rather than
            # introducing a second repository default in configuration.
            lateral = PurePursuitController(
                lookahead_m=lookahead_m,
                wheelbase_m=wheelbase_m,
                max_steer_rad=max_steer_rad,
            )
        else:
            lateral = PurePursuitController(
                lookahead_m=lookahead_m,
                wheelbase_m=wheelbase_m,
                max_steer_rad=max_steer_rad,
                goal_tolerance_m=goal_tolerance,
            )
        longitudinal = LongitudinalPIDController(pid=self.speed_controller)
        self.trajectory_controller = TrajectoryController(lateral, longitudinal)
        # Compatibility façade; no duplicate control implementation.
        self.pure_pursuit = PurePursuit(
            lookahead_m=lookahead_m,
            wheelbase_m=wheelbase_m,
            max_steer_rad=max_steer_rad,
        )

        self.kpi_tracker = self._build_kpi_tracker()
        self.watchdog = self._build_watchdog()
        self._watchdog_cfg = get_config_section("watchdog", self.config)
        self._gnss_required = _strict_bool(
            self._watchdog_cfg.get("gnss_required"), default=False
        )
        self._planner_required = _strict_bool(
            self._watchdog_cfg.get("planner_required"), default=False
        )
        self.adaptation_guard = self._build_adaptation_guard()
        self._recovery_target = RoboCarRecoveryTarget(self)

        self.sensor_bus.subscribe(self._on_sensor_reading)

        # Built on start after the physical ExecutionAgent exists.
        self.autonomy_loop: Any = None
        self._eager_support_agents = bool(eager_support_agents)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_kpi_tracker(self) -> VehicleKPITracker:
        cfg = get_config_section("kpi", self.config)
        return VehicleKPITracker(
            near_miss_distance_m=optional_finite_float(
                cfg.get("near_miss_distance_m"), minimum=0.0
            ),
            reaction_time_s=optional_finite_float(
                cfg.get("reaction_time_s"), minimum=0.0
            ),
            max_deceleration_mps2=optional_finite_float(
                cfg.get("max_deceleration_mps2"), minimum=0.0
            ),
        )

    def _build_watchdog(self) -> VehicleWatchdog:
        cfg = get_config_section("watchdog", self.config)
        thresholds = WatchdogThresholds(
            sensor_frame_timeout_s=_optional_positive(
                cfg.get("sensor_frame_timeout_s")
            ),
            pico_heartbeat_timeout_s=_optional_positive(
                cfg.get("pico_heartbeat_timeout_s")
            ),
            control_cycle_deadline_s=_optional_positive(
                cfg.get("control_cycle_deadline_s")
            ),
            gnss_timeout_s=_optional_positive(cfg.get("gnss_timeout_s")),
            planner_timeout_s=_optional_positive(cfg.get("planner_timeout_s")),
        )
        return VehicleWatchdog(thresholds)

    def _build_adaptation_guard(self) -> AdaptationGuard:
        cfg = get_config_section("adaptation", self.config)
        raw_rules = cfg.get("rules", {})
        if raw_rules is None:
            raw_rules = {}
        if not isinstance(raw_rules, Mapping):
            raise ValueError("adaptation.rules must be a mapping when configured")

        rules: Dict[str, ParameterRule] = {}
        for name, raw in raw_rules.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"adaptation.rules.{name} must be a mapping")
            rules[str(name)] = ParameterRule(
                minimum=require_finite_float(
                    raw.get("minimum"), f"adaptation.rules.{name}.minimum"
                ),
                maximum=require_finite_float(
                    raw.get("maximum"), f"adaptation.rules.{name}.maximum"
                ),
                max_delta_per_proposal=require_finite_float(
                    raw.get("max_delta_per_proposal"),
                    f"adaptation.rules.{name}.max_delta_per_proposal",
                    minimum=1e-12,
                ),
                max_change_per_second=require_finite_float(
                    raw.get("max_change_per_second"),
                    f"adaptation.rules.{name}.max_change_per_second",
                    minimum=1e-12,
                ),
                minimum_samples=require_int(
                    raw.get("minimum_samples"),
                    f"adaptation.rules.{name}.minimum_samples",
                    minimum=1,
                ),
                minimum_confidence=require_probability(
                    raw.get("minimum_confidence", 0.0),
                    f"adaptation.rules.{name}.minimum_confidence",
                ),
            )

        # Empty rules are deliberately valid and mean deny-all adaptation.
        return AdaptationGuard(rules=rules)

    # ------------------------------------------------------------------
    # Lifecycle and SLAI agent integration
    # ------------------------------------------------------------------

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._started:
                return

            # Keep the physical actuator boundary neutral throughout startup.
            self.motion.stop()
            try:
                self.sensor_bus.start()
                self._runtime_started_monotonic = time.monotonic()
                if (
                    self.sensor_bus.is_simulation or self.motion.simulation_mode
                ) and not self.allow_simulation:
                    raise RuntimeError(
                        "Simulation became active without explicit permission"
                    )

                self._initialize_required_agents()
                if self._eager_support_agents:
                    self._initialize_support_agents()
                configured_knowledge_k = optional_int(
                    get_config_section("robocar", self.config).get("knowledge_k"),
                    minimum=0,
                )
                self.autonomy_loop = build_robocar_autonomy_loop(
                    car=self,
                    world_model=self.world_model,
                    kpi_tracker=self.kpi_tracker,
                    adaptation_guard=self.adaptation_guard,
                    knowledge_k=(
                        3 if configured_knowledge_k is None else configured_knowledge_k
                    ),
                )
                # The repository stage adapter already implements stop-first
                # Handler recovery, but a stage may return an explicit failed
                # status without throwing; in that case its private failed-agent
                # identity can be absent.  Resolve the stage->agent mapping here
                # so HandlerAgent always receives a meaningful recovery target.
                self.autonomy_loop.handler = self._autonomy_failure_handler

                self.shared_memory.set(K_CONFIG, self._public_config_snapshot())
                existing_safety = self.shared_memory.get(K_SAFETY_STATE, default={})
                existing_safety = (
                    dict(existing_safety)
                    if isinstance(existing_safety, Mapping)
                    else {}
                )
                startup_estop = existing_safety.get("estop") is True
                # Never clear a pre-existing safety latch merely because the
                # process restarted.  Operator-confirmed clear remains explicit.
                self.world_model.update(
                    safety=SafetyState(
                        estop_latched=startup_estop,
                        allowed_to_move=False,
                        degraded=False,
                        reasons=("preexisting_estop_latch",) if startup_estop else (),
                    ),
                    actuation=ActuationState(
                        throttle=0.0,
                        steering=0.0,
                        source="robocar.start",
                        status="neutral",
                    ),
                    autonomy=AutonomyState(
                        mode=(
                            OperatingMode.EMERGENCY_STOP
                            if startup_estop
                            else OperatingMode.STOPPED
                        )
                    ),
                    event_type="robocar.started",
                    event_payload={
                        "simulation_allowed": self.allow_simulation,
                        "sensor_mode": self.sensor_bus.health().get("mode"),
                        "motion_mode": self.motion.get_status().get("mode"),
                        "preexisting_estop_latch": startup_estop,
                    },
                )
                self._started = True
                self._emit_observability(
                    event="robocar_started",
                    payload={"health": self._core_health_snapshot()},
                )
                logger.info("RoboCar started (simulation_allowed=%s)", self.allow_simulation)
            except Exception:
                # Startup failure must never leave a previously initialized PWM
                # boundary energized.
                try:
                    self.motion.stop()
                except Exception:
                    pass
                try:
                    self.sensor_bus.stop()
                except Exception:
                    pass
                raise

    def _initialize_required_agents(self) -> None:
        safety = self._raw_agent("safety")
        execution = self.agent_factory.create(
            "execution",
            shared_memory=self.shared_memory,
            robot=self.robot_adapter,
        )
        self._agents["execution"] = execution
        self._raw_agents["execution"] = execution

        # Register only robot actions compatible with this Ackermann chassis.
        for name, action_class in _REGISTERED_ROBOT_ACTIONS.items():
            if name not in getattr(execution, "action_class_registry", {}):
                execution.register_action(name, action_class)

        if safety is None or execution is None:
            raise RuntimeError("Required SLAI Safety/Execution agents unavailable")

    def _initialize_support_agents(self) -> None:
        for name in ("handler", "observability", "evaluation"):
            try:
                self.agent(name)
                self._agent_errors.pop(name, None)
            except Exception as exc:
                self._agent_errors[name] = f"{type(exc).__name__}: {exc}"
                logger.warning("RoboCar support agent %s unavailable: %s", name, exc)

    def _raw_agent(self, name: str) -> Any:
        normalized = str(name).strip().lower()
        if normalized in self._raw_agents:
            return self._raw_agents[normalized]
        agent = self.agent_factory.create(
            normalized,
            shared_memory=self.shared_memory,
        )
        self._raw_agents[normalized] = agent
        return agent

    def agent(self, name: str) -> Any:
        """Return one scoped SLAI agent, preserving physical execution identity."""

        normalized = str(name).strip().lower()
        if not normalized:
            raise ValueError("agent name cannot be empty")
        if normalized in self._agents:
            return self._agents[normalized]

        if normalized == "execution":
            existing = self._raw_agents.get("execution")
            if existing is None:
                raise RuntimeError(
                    "Physical ExecutionAgent is not initialized; call RoboCar.start()"
                )
            self._agents[normalized] = existing
            return existing

        if normalized == "evaluation":
            raw = self._raw_agent("evaluation")
            bridge = RoboCarEvaluationBridge(raw, self)
            self._agents[normalized] = bridge
            return bridge

        raw = self._raw_agent(normalized)
        self._agents[normalized] = raw
        return raw

    def close(self) -> None:
        with self._lifecycle_lock:
            # Fail-safe order: hardware first, then producers/agents/resources.
            try:
                self.motion.stop()
                self._set_actuation_neutral(source="robocar.close")
            except Exception as exc:
                self._last_error = (
                    f"stop_during_close: {type(exc).__name__}: {exc}"
                )
                logger.critical("RoboCar failed to confirm neutral during close: %s", exc)

            loop = self.autonomy_loop
            if loop is not None:
                close_loop = getattr(loop, "close", None)
                if callable(close_loop):
                    try:
                        close_loop()
                    except Exception as exc:
                        logger.warning("Autonomy loop close degraded: %s", exc)

            try:
                self.sensor_bus.stop()
            except Exception as exc:
                logger.warning("SensorBus stop degraded: %s", exc)

            # Agent lifecycle is owned by AgentFactory.  Do not directly
            # shutdown a factory-managed instance here: with an injected factory
            # that could invalidate another owner's runtime scope.  An owned
            # factory is shut down below after the hardware boundary is closed.
            self.motion.close()
            self._started = False
            self._runtime_started_monotonic = None
            self._autonomy_started_monotonic = None

            try:
                self.world_model.update(
                    autonomy=AutonomyState(mode=OperatingMode.STOPPED),
                    actuation=ActuationState(
                        throttle=0.0,
                        steering=0.0,
                        source="robocar.close",
                        status="neutral",
                    ),
                    event_type="robocar.stopped",
                )
            except Exception:
                pass

            if self._owns_factory:
                shutdown_factory = getattr(self.agent_factory, "shutdown", None)
                if callable(shutdown_factory):
                    try:
                        shutdown_factory()
                    except Exception as exc:
                        logger.warning("Owned AgentFactory shutdown degraded: %s", exc)

    stop = close

    # ------------------------------------------------------------------
    # Sensor / world-state publication
    # ------------------------------------------------------------------

    def _on_sensor_reading(self, reading: SensorReading) -> None:
        """Fast deterministic sensor callback; never invokes SLAI agents."""

        now_mono = time.monotonic()
        self._last_sensor_frame_monotonic = now_mono
        try:
            payload = reading.to_dict()
            self.shared_memory.set(K_SENSOR_LATEST, payload)
            if reading.ultra_front_m is not None:
                self.shared_memory.set(K_ULTRA_FRONT, reading.ultra_front_m)
            if reading.ultra_rear_m is not None:
                self.shared_memory.set(K_ULTRA_REAR, reading.ultra_rear_m)
            if reading.tof_mm is not None:
                self.shared_memory.set(K_TOF_MM, reading.tof_mm)
            if reading.vbat is not None:
                self.shared_memory.set(K_BATTERY_VOLT, reading.vbat)
                self.shared_memory.set(
                    K_BATTERY_STATE,
                    self.local_safety.battery_state(reading.vbat),
                )
            if reading.encoder_ticks_total is not None:
                self.shared_memory.set(K_ENCODER_TICKS, reading.encoder_ticks_total)
                speed = self.encoder.update_from_ticks(reading.encoder_ticks_total)
                self.shared_memory.set(K_ENCODER_SPEED, speed)

            sensor_health = self._sensor_health_state(reading, now_mono)
            obstacles = self._obstacle_state(reading, now_mono)

            gnss = self._gnss_from_sensor_reading(reading, now_mono)
            updated = self.world_model.update(
                gnss=gnss,
                obstacles=obstacles,
                sensor_health=sensor_health,
                event_type="sensor.frame",
                event_payload={
                    "frame_wall_time": reading.t,
                    "required_sensor_score": sensor_health.score,
                },
            )
            self._observe_kpis(updated)
        except Exception as exc:
            self._last_error = f"sensor_publish: {type(exc).__name__}: {exc}"
            try:
                self.world_model.record_event(
                    "sensor.publish_failed",
                    payload={"error": self._last_error},
                    severity="warning",
                )
            except Exception:
                pass
            logger.exception("Failed to publish RoboCar sensor frame")

    def _sensor_health_state(
        self,
        reading: SensorReading,
        now_mono: float,
    ) -> SensorHealthState:
        bus = self.sensor_bus.health()
        required = get_config_section("robocar", self.config).get(
            "required_sensor_fields", []
        )
        if required is None:
            required = []
        if isinstance(required, str):
            required = [required]
        if not isinstance(required, Sequence):
            raise ValueError("robocar.required_sensor_fields must be a sequence")

        required_names = tuple(
            str(name).strip() for name in required if str(name).strip()
        )
        available: list[str] = []
        unavailable: list[str] = []
        for name in required_names:
            if not hasattr(reading, name):
                unavailable.append(name)
                continue
            value = getattr(reading, name)
            if value is None:
                unavailable.append(name)
            else:
                available.append(name)

        score = (
            len(available) / len(required_names)
            if required_names
            else None
        )
        status = str(bus.get("status", "unknown")).lower()
        transport_errors = int(bus.get("transport_errors", 0) or 0)
        parse_errors = int(bus.get("parse_errors", 0) or 0)

        if status in {"failed", "fault", "critical"}:
            level = HealthLevel.FAILED
        elif unavailable or transport_errors > 0 or status in {"simulation", "degraded"}:
            level = HealthLevel.DEGRADED
        elif status == "operational":
            level = HealthLevel.HEALTHY
        else:
            level = HealthLevel.UNKNOWN

        return SensorHealthState(
            level=level,
            available=tuple(available),
            unavailable=tuple(unavailable),
            stale=(),
            score=score,
            dropped_frames=int(bus.get("dropped_frames", 0) or 0),
            parse_errors=parse_errors,
            transport_errors=transport_errors,
            last_sensor_frame_monotonic=now_mono,
            last_pico_heartbeat_monotonic=self._last_pico_heartbeat_monotonic,
            details={
                "bus_status": status,
                "bus_mode": bus.get("mode"),
                "required_sensor_fields": list(required_names),
                "score_definition": (
                    "unweighted_required_field_availability_ratio"
                    if required_names
                    else "not_configured"
                ),
            },
        )

    @staticmethod
    def _obstacle_state(reading: SensorReading, now_mono: float) -> ObstacleState:
        front_candidates: list[float] = []
        front_ultra = optional_finite_float(reading.ultra_front_m, minimum=0.0)
        if front_ultra is not None:
            front_candidates.append(front_ultra)
        tof_mm = optional_int(reading.tof_mm, minimum=0)
        tof_m = None if tof_mm is None else float(tof_mm) / 1000.0
        if tof_m is not None:
            front_candidates.append(tof_m)
        front = min(front_candidates) if front_candidates else None

        rear = optional_finite_float(reading.ultra_rear_m, minimum=0.0)
        all_ranges = [value for value in (front, rear) if value is not None]
        nearest = min(all_ranges) if all_ranges else None

        return ObstacleState(
            front_distance_m=front,
            rear_distance_m=rear,
            nearest_distance_m=nearest,
            # No confidence/disagreement threshold is configured in the current
            # repository; keep those fields explicitly unknown/default.
            front_confidence=None,
            rear_confidence=None,
            disagreement=False,
            source_health={
                "front_ultrasonic": (
                    "available" if front_ultra is not None else "missing"
                ),
                "front_tof": "available" if tof_m is not None else "missing",
                "rear_ultrasonic": "available" if rear is not None else "missing",
            },
            timestamp_monotonic=now_mono,
        )

    def _gnss_from_sensor_reading(
        self,
        reading: SensorReading,
        now_mono: float,
    ) -> Optional[GNSSState]:
        """Consume future Pico-forwarded GNSS fields when SensorReading exposes them.

        The current live SensorReading does not yet include these attributes, so
        this method returns ``None`` today rather than fabricating GNSS state.
        """

        lat = getattr(reading, "gnss_lat_deg", None)
        lon = getattr(reading, "gnss_lon_deg", None)
        valid_raw = getattr(reading, "gnss_valid", None)
        if lat is None and lon is None and valid_raw is None:
            return None

        parsed_lat = optional_finite_float(lat, minimum=-90.0, maximum=90.0)
        parsed_lon = optional_finite_float(lon, minimum=-180.0, maximum=180.0)
        valid = bool(valid_raw) and parsed_lat is not None and parsed_lon is not None
        state = GNSSState(
            latitude_deg=parsed_lat,
            longitude_deg=parsed_lon,
            altitude_m=optional_finite_float(getattr(reading, "gnss_alt_m", None)),
            speed_mps=optional_finite_float(
                getattr(reading, "gnss_speed_mps", None), minimum=0.0
            ),
            track_deg=optional_finite_float(
                getattr(reading, "gnss_track_deg", None), minimum=0.0, maximum=360.0
            ),
            hdop=optional_finite_float(
                getattr(reading, "gnss_hdop", None), minimum=0.0
            ),
            satellites_used=optional_int(
                getattr(reading, "gnss_satellites", None), minimum=0
            ),
            fix_quality=optional_int(
                getattr(reading, "gnss_fix_quality", None), minimum=0
            ),
            valid=valid,
            timestamp_monotonic=now_mono,
        )
        self._gnss_seen = True
        if state.valid:
            self._last_gnss_fix_monotonic = now_mono
        return state

    def update_gnss(self, fix: Any) -> GNSSState:
        """Publish a validated GNSSFix/GNSSState into the authoritative world model.

        This method deliberately does not transform WGS-84 coordinates into the
        local metric pose frame.
        """

        if isinstance(fix, GNSSState):
            state = fix
        else:
            lat = getattr(fix, "latitude_deg", None)
            lon = getattr(fix, "longitude_deg", None)
            valid = bool(getattr(fix, "valid", False))
            state = GNSSState(
                latitude_deg=lat,
                longitude_deg=lon,
                altitude_m=getattr(fix, "altitude_m", None),
                speed_mps=getattr(fix, "speed_mps", None),
                track_deg=getattr(fix, "track_deg", None),
                hdop=getattr(fix, "hdop", None),
                satellites_used=getattr(fix, "satellites_used", None),
                fix_quality=getattr(fix, "fix_quality", None),
                valid=valid,
                timestamp_monotonic=getattr(
                    fix, "receipt_monotonic", time.monotonic()
                ),
            )
        self._gnss_seen = True
        if state.valid and state.timestamp_monotonic is not None:
            self._last_gnss_fix_monotonic = state.timestamp_monotonic
        self.world_model.update(
            gnss=state,
            event_type="gnss.updated",
            event_payload={
                "valid": state.valid,
                "satellites_used": state.satellites_used,
                "hdop": state.hdop,
            },
        )
        return state

    def update_pose(self, pose: PoseState) -> PoseState:
        """Publish localization output without deriving it inside the orchestrator."""

        if not isinstance(pose, PoseState):
            raise TypeError("update_pose requires modules.world_model.PoseState")
        self.world_model.update(
            pose=pose,
            event_type="localization.pose_updated",
            event_payload={"confidence": pose.confidence},
        )
        self.shared_memory.set(
            K_POSE_ESTIMATE,
            {
                "x": pose.x_m,
                "y": pose.y_m,
                "theta": pose.yaw_rad,
                "v": pose.speed_mps,
                "confidence": pose.confidence,
            },
        )
        return pose

    def get_sensor_value(self, sensor_name: str) -> Any:
        name = str(sensor_name).strip()
        if name in {"speed", "speed_mps", K_ENCODER_SPEED}:
            return self.encoder.get_speed()
        reading = self.sensor_bus.latest()
        if reading is None:
            raise SensorError(name, None, ("available", "available"))
        aliases = {
            "front_distance": "ultra_front_m",
            "rear_distance": "ultra_rear_m",
            "battery_voltage": "vbat",
            "encoder_ticks": "encoder_ticks_total",
        }
        attribute = aliases.get(name, name)
        if not hasattr(reading, attribute):
            raise KeyError(f"Unknown RoboCar sensor name: {sensor_name!r}")
        value = getattr(reading, attribute)
        if value is None:
            raise SensorError(name, None, ("valid reading", "valid reading"))
        return value

    # ------------------------------------------------------------------
    # Vehicle-lighting intent
    # ------------------------------------------------------------------
    
    def set_drive_intent(self, keep_driving: bool) -> Dict[str, Any]:
        """Set normal driving-light intent without changing vehicle motion."""
    
        command = self.sensor_bus.set_drive_intent(
            keep_driving
        )
        return command.to_payload()
    
    def update_turn_intent(
        self,
        direction: Optional[str],
        distance_to_turn_m: Optional[float],
    ) -> Dict[str, Any]:
        """Update a planner/localizer-provided upcoming turn measurement.
    
        The appropriate indicator remains off until the supplied distance reaches
        the configured one-metre activation boundary.
        """
    
        command = self.sensor_bus.set_turn_intent(
            direction,
            distance_to_turn_m,
        )
        return command.to_payload()
    
    def park(self, *, reason: str = "stationary_parking_intent") -> Dict[str, Any]:
        """Stop the chassis and start the finite parking-light acknowledgement."""
    
        if not self._started:
            raise RuntimeError(
                "RoboCar.start() must be called before parking"
            )
    
        hardware = self._safe_hardware_stop(reason)
        lighting = self.sensor_bus.park_lighting()
        snapshot = self.world_model.snapshot()
    
        self.world_model.update(
            autonomy=AutonomyState(
                mode=OperatingMode.STOPPED,
                run_id=snapshot.autonomy.run_id,
                goal_id=snapshot.autonomy.goal_id,
                cycle=snapshot.autonomy.cycle,
                planner_status=snapshot.autonomy.planner_status,
                last_plan_monotonic=(
                    snapshot.autonomy.last_plan_monotonic
                ),
                last_control_cycle_monotonic=(
                    snapshot.autonomy.last_control_cycle_monotonic
                ),
                last_recovery_monotonic=(
                    snapshot.autonomy.last_recovery_monotonic
                ),
                metadata={
                    **dict(snapshot.autonomy.metadata or {}),
                    "park_reason": str(reason),
                },
            ),
            event_type="vehicle.parked",
            event_payload={
                "reason": str(reason),
                "lighting": lighting.to_payload(),
            },
        )
    
        return {
            "status": "parked",
            "reason": str(reason),
            "hardware": hardware,
            "lighting": lighting.to_payload(),
        }
    
    def _update_route_turn_lighting(self, command: TrajectoryCommand) -> None:
        """Derive the next material route bend and update turn-light intent."""
    
        snapshot = self.world_model.snapshot()
    
        direction, distance_m = self._upcoming_route_turn(
            snapshot,
            active_index=command.target_index,
        )
    
        self.sensor_bus.set_drive_intent(True)
        self.sensor_bus.set_turn_intent(
            direction,
            distance_m,
        )
    
    def _upcoming_route_turn(self, snapshot: WorldSnapshot, *, active_index: int) -> tuple[Optional[str], Optional[float]]:
        """Return the next significant route bend and along-path distance.
    
        Small waypoint-to-waypoint changes are accumulated so a densely sampled
        curve can still be recognized. Opposing changes reset the accumulated
        angle, preventing ordinary waypoint noise from producing a false turn.
        """
    
        pose = snapshot.pose
        route = snapshot.route
    
        if pose is None or len(route.waypoints) < 2:
            return None, None
    
        index = max(
            0,
            min(
                int(active_index),
                len(route.waypoints) - 1,
            ),
        )
    
        vertices = [
            (pose.x_m, pose.y_m),
            *[
                (point.x_m, point.y_m)
                for point in route.waypoints[index:]
            ],
        ]
    
        if len(vertices) < 3:
            return None, None
    
        cumulative_distance = 0.0
        accumulated_angle = 0.0
        turn_start_distance: Optional[float] = None
        previous_sign = 0
    
        for vertex_index in range(1, len(vertices) - 1):
            previous = vertices[vertex_index - 1]
            vertex = vertices[vertex_index]
            following = vertices[vertex_index + 1]
    
            incoming_dx = vertex[0] - previous[0]
            incoming_dy = vertex[1] - previous[1]
            outgoing_dx = following[0] - vertex[0]
            outgoing_dy = following[1] - vertex[1]
    
            incoming_length = math.hypot(
                incoming_dx,
                incoming_dy,
            )
            outgoing_length = math.hypot(
                outgoing_dx,
                outgoing_dy,
            )
    
            cumulative_distance += incoming_length
    
            if (
                incoming_length <= 1e-9
                or outgoing_length <= 1e-9
            ):
                continue
    
            incoming_heading = math.atan2(
                incoming_dy,
                incoming_dx,
            )
            outgoing_heading = math.atan2(
                outgoing_dy,
                outgoing_dx,
            )
    
            delta = math.atan2(
                math.sin(outgoing_heading - incoming_heading),
                math.cos(outgoing_heading - incoming_heading),
            )
    
            if abs(delta) <= 1e-6:
                continue
    
            sign = 1 if delta > 0.0 else -1
    
            if previous_sign and sign != previous_sign:
                accumulated_angle = 0.0
                turn_start_distance = None
    
            if turn_start_distance is None:
                turn_start_distance = cumulative_distance
    
            previous_sign = sign
            accumulated_angle += delta
    
            if (
                abs(accumulated_angle)
                >= self._turn_detection_min_angle_rad
            ):
                return (
                    (
                        "left"
                        if accumulated_angle > 0.0
                        else "right"
                    ),
                    max(0.0, turn_start_distance),
                )
    
        return None, None

    # ------------------------------------------------------------------
    # World / KPI helpers
    # ------------------------------------------------------------------

    def _observe_kpis(
        self,
        snapshot: Optional[WorldSnapshot] = None,
        *,
        target_heading_rad: Optional[float] = None,
        control_loop_duration_s: Optional[float] = None,
    ) -> KPISnapshot:
        snap = snapshot if snapshot is not None else self.world_model.snapshot()
        route = snap.route.waypoints if snap.route.waypoints else None
        deadline = self.watchdog.thresholds.control_cycle_deadline_s
        kpi = self.kpi_tracker.observe(
            pose=snap.pose,
            path=route,
            target_heading_rad=target_heading_rad,
            front_obstacle_distance_m=snap.obstacles.front_distance_m,
            configured_stop_distance_m=self.local_safety.front_stop_distance_m,
            sensor_available=len(snap.sensor_health.available),
            sensor_total=(
                len(snap.sensor_health.available)
                + len(snap.sensor_health.unavailable)
            )
            if (snap.sensor_health.available or snap.sensor_health.unavailable)
            else None,
            sensor_health_score=snap.sensor_health.score,
            gnss_available=(snap.gnss.valid if self._gnss_seen else None),
            mode=snap.autonomy.mode.value,
            control_loop_duration_s=control_loop_duration_s,
            control_loop_deadline_s=deadline,
            dropped_pico_frames_total=snap.sensor_health.dropped_frames,
        )
        self.shared_memory.set(K_KPI_LATEST, kpi.to_dict())
        return kpi

    def _update_actuation_state(
        self,
        *,
        throttle: float,
        steering: float,
        source: str,
        status: str,
        requested_speed_mps: Optional[float] = None,
    ) -> None:
        self.world_model.update(
            actuation=ActuationState(
                throttle=throttle,
                steering=steering,
                requested_speed_mps=requested_speed_mps,
                source=source,
                status=status,
            ),
            event_type="actuation.updated",
            event_payload={
                "throttle": throttle,
                "steering": steering,
                "source": source,
                "status": status,
            },
        )

    def _set_actuation_neutral(self, *, source: str) -> None:
        self._update_actuation_state(
            throttle=0.0,
            steering=0.0,
            source=source,
            status="neutral",
        )

    # ------------------------------------------------------------------
    # Safety and direct execution
    # ------------------------------------------------------------------

    def _publish_local_safety(self, decision: SafetyDecision) -> None:
        current = self.shared_memory.get(K_SAFETY_STATE, default={})
        state = dict(current) if isinstance(current, Mapping) else {}
        state.update(
            {
                "local_allowed": decision.allowed,
                "local_reasons": list(decision.reasons),
                "local_warnings": list(decision.warnings),
                "updated_at": time.time(),
            }
        )
        self.shared_memory.set(K_SAFETY_STATE, state)

        snapshot = self.world_model.snapshot()
        self.world_model.update(
            safety=SafetyState(
                estop_latched=snapshot.safety.estop_latched,
                allowed_to_move=decision.allowed,
                degraded=snapshot.safety.degraded,
                speed_cap_mps=snapshot.safety.speed_cap_mps,
                reasons=decision.reasons,
                warnings=decision.warnings,
            ),
            event_type="local_safety.authorization",
            event_payload=decision.to_dict(),
            event_severity="info" if decision.allowed else "warning",
        )

    def _safe_hardware_stop(self, reason: str) -> Dict[str, Any]:
        """Stop hardware immediately and reflect neutral actuation state."""

        result = self.motion.stop()
        self._set_actuation_neutral(source=f"safe_stop:{reason}")
        return result

    def emergency_stop(self, reason: str = "operator_or_safety_request") -> Dict[str, Any]:
        """Hardware-first emergency stop with a latched software state."""

        hardware_error: Optional[str] = None
        hardware: Dict[str, Any] = {}
        try:
            hardware = self.motion.stop()
        except Exception as exc:
            hardware_error = f"{type(exc).__name__}: {exc}"

        # Latch even if hardware confirmation failed: subsequent local safety
        # gates must continue denying motion.
        current = self.shared_memory.get(K_SAFETY_STATE, default={})
        state = dict(current) if isinstance(current, Mapping) else {}
        state.update(
            {
                "estop": True,
                "reason": str(reason),
                "hardware_stop_error": hardware_error,
                "updated_at": time.time(),
            }
        )
        self.shared_memory.set(K_SAFETY_STATE, state)
        self.kpi_tracker.record_intervention()

        snapshot = self.world_model.snapshot()
        self.world_model.update(
            safety=SafetyState(
                estop_latched=True,
                allowed_to_move=False,
                degraded=hardware_error is not None or snapshot.safety.degraded,
                speed_cap_mps=snapshot.safety.speed_cap_mps,
                reasons=tuple(snapshot.safety.reasons) + (str(reason),),
                warnings=snapshot.safety.warnings,
            ),
            actuation=ActuationState(
                throttle=0.0,
                steering=0.0,
                source="emergency_stop",
                status="neutral" if hardware_error is None else "stop_unconfirmed",
            ),
            autonomy=AutonomyState(
                mode=OperatingMode.EMERGENCY_STOP,
                run_id=snapshot.autonomy.run_id,
                goal_id=snapshot.autonomy.goal_id,
                cycle=snapshot.autonomy.cycle,
                planner_status=snapshot.autonomy.planner_status,
                last_plan_monotonic=snapshot.autonomy.last_plan_monotonic,
                last_control_cycle_monotonic=(
                    snapshot.autonomy.last_control_cycle_monotonic
                ),
                last_recovery_monotonic=snapshot.autonomy.last_recovery_monotonic,
                metadata=snapshot.autonomy.metadata,
            ),
            event_type="safety.emergency_stop",
            event_payload={"reason": reason, "hardware_error": hardware_error},
            event_severity="critical",
        )
        self._emit_observability(
            event="emergency_stop",
            payload={"reason": reason, "hardware_error": hardware_error},
            severity="critical",
        )
        return {
            "status": "stopped" if hardware_error is None else "stop_unconfirmed",
            "reason": reason,
            "hardware": hardware,
            "hardware_error": hardware_error,
        }

    def clear_emergency_stop(self, *, operator_confirmed: bool = False) -> None:
        """Clear the software latch only after explicit operator confirmation.

        This never re-arms throttle.  The car remains stopped after clearing.
        """

        if operator_confirmed is not True:
            raise PermissionError(
                "clear_emergency_stop requires operator_confirmed=True"
            )
        report = self.check_watchdog(enforce=False)
        if report.requires_stop:
            raise RuntimeError(
                "Cannot clear emergency stop while watchdog reports a critical fault"
            )
        motion = self.motion.get_status()
        if str(motion.get("status", "")).lower() in {
            "failed",
            "fault",
            "faulty",
            "critical",
        }:
            raise RuntimeError("Cannot clear emergency stop while actuator is faulty")

        self.motion.stop()
        current = self.shared_memory.get(K_SAFETY_STATE, default={})
        state = dict(current) if isinstance(current, Mapping) else {}
        state.update(
            {
                "estop": False,
                "reason": "operator_cleared",
                "updated_at": time.time(),
            }
        )
        self.shared_memory.set(K_SAFETY_STATE, state)
        snap = self.world_model.snapshot()
        self.world_model.update(
            safety=SafetyState(
                estop_latched=False,
                allowed_to_move=False,
                degraded=snap.safety.degraded,
                speed_cap_mps=snap.safety.speed_cap_mps,
                reasons=(),
                warnings=snap.safety.warnings,
            ),
            actuation=ActuationState(
                throttle=0.0,
                steering=0.0,
                source="operator_clear_estop",
                status="neutral",
            ),
            autonomy=AutonomyState(mode=OperatingMode.STOPPED),
            event_type="safety.emergency_stop_cleared",
        )

    def execute_ackermann_action(
        self,
        *,
        throttle: float,
        steering: float,
        duration: float = 0.0,
        source: str = "robocar",
        require_slai_safety: bool = True,
        allow_persistent: bool = False,
    ) -> Dict[str, Any]:
        """Safety-gate and execute one current SLAI AckermannAction.

        Non-zero throttle with ``duration == 0`` is rejected unless
        ``allow_persistent=True`` because the current SLAI AckermannAction does
        not auto-neutralize that command.
        """

        if not self._started:
            raise RuntimeError("RoboCar.start() must be called before execution")
        duration_value = optional_finite_float(duration, minimum=0.0)
        if duration_value is None:
            raise ValueError("duration must be a finite non-negative number")
        throttle_value = normalize_signed_command(throttle, "AckermannThrottle")
        steering_value = normalize_signed_command(steering, "AckermannSteering")
        if (
            abs(throttle_value) > 1e-6
            and duration_value <= 0.0
            and not allow_persistent
        ):
            raise ValueError(
                "Non-zero Ackermann throttle requires duration > 0 unless "
                "allow_persistent=True is explicitly requested"
            )

        watchdog = self.check_watchdog(enforce=True)
        # A temporary stop between trajectory-control steps still represents an
        # intent to continue driving. Keep the head and tail lights enabled.
        self.sensor_bus.set_drive_intent(True)
        if watchdog.requires_stop:
            return {
                "status": "blocked",
                "reason": "watchdog",
                "watchdog": watchdog.to_dict(),
            }

        local = self.local_safety.authorize_command(
            throttle_value,
            steering_value,
            reading=self.sensor_bus.latest(),
            speed_mps=self.encoder.get_speed(),
        )
        self._publish_local_safety(local)
        if not local.allowed:
            self._safe_hardware_stop("local_safety_block")
            return {
                "status": "blocked",
                "reason": "local_safety",
                "local_safety": local.to_dict(),
            }

        action_params = {
            "name": "ackermann",
            "action_name": "ackermann",
            "throttle": local.throttle,
            "steering": local.steering,
            "duration": duration_value,
            "source": str(source),
        }
        safety_result: Dict[str, Any] = {
            "approved": True,
            "decision": "local_only",
        }
        if require_slai_safety:
            latest = self.sensor_bus.latest()
            safety_result = dict(
                self.agent("safety").validate_action(
                    action_params,
                    {
                        "system": "RoboCar",
                        "sensor": latest.to_dict() if latest is not None else None,
                        "world": self.world_model.snapshot().to_dict(),
                        "local_safety": local.to_dict(),
                    },
                )
            )
            if safety_result.get("approved") is not True:
                self._safe_hardware_stop("slai_safety_block")
                return {
                    "status": "blocked",
                    "reason": "slai_safety",
                    "local_safety": local.to_dict(),
                    "safety": safety_result,
                }

        execution = self.agent("execution")
        task_name = f"robocar_ackermann_{uuid.uuid4().hex[:10]}"
        task = {
            "name": task_name,
            "goal_type": "robot_control",
            "requirements": [],
            "priority": 1,
            "timeout": max(5.0, duration_value + 5.0),
            "action_sequence": [
                {
                    "name": "ackermann",
                    "steering": local.steering,
                    "throttle": local.throttle,
                    "duration": duration_value,
                }
            ],
            "metadata": {
                "source": str(source),
                "safety_validation": safety_result.get("validation_id"),
                "persistent_throttle_explicit": bool(allow_persistent),
            },
        }

        started = time.monotonic()
        try:
            result = dict(execution.perform_task(task))
        except Exception as exc:
            self._last_control_cycle_duration_s = time.monotonic() - started
            recovery = self.handle_failure(
                exc,
                source="execute_ackermann_action",
                target_agent=execution,
                task_data=task,
                safe_stop=True,
            )
            return {
                "status": "failed",
                "reason": "execution_exception",
                "error": f"{type(exc).__name__}: {exc}",
                "handler": recovery,
                "local_safety": local.to_dict(),
                "safety": safety_result,
            }

        self._last_control_cycle_duration_s = time.monotonic() - started
        if duration_value > 0.0:
            # AckermannAction itself neutralizes throttle after duration.  The
            # adapter updates the authoritative actuation snapshot accordingly.
            pass

        self._observe_kpis(
            control_loop_duration_s=self._last_control_cycle_duration_s
        )
        self._emit_observability(
            event="ackermann_action",
            payload={
                "source": source,
                "duration_s": self._last_control_cycle_duration_s,
                "execution_status": result.get("status"),
            },
        )
        return {
            "status": result.get("status", "unknown"),
            "execution": result,
            "local_safety": local.to_dict(),
            "safety": safety_result,
        }

    # ------------------------------------------------------------------
    # Local planning / trajectory control
    # ------------------------------------------------------------------

    def set_route(
        self,
        waypoints: Sequence[WaypointState | Waypoint | Mapping[str, Any]],
        *,
        route_id: Optional[str] = None,
        planner_name: str = "external",
        goal_tolerance_m: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> RouteState:
        points: list[WaypointState] = []
        for item in waypoints:
            if isinstance(item, WaypointState):
                points.append(item)
            elif isinstance(item, Waypoint):
                points.append(item.to_waypoint_state())
            elif isinstance(item, Mapping):
                x = item.get("x_m", item.get("x"))
                y = item.get("y_m", item.get("y"))
                points.append(
                    WaypointState(
                        x_m=require_finite_float(x, "route.waypoint.x_m"),
                        y_m=require_finite_float(y, "route.waypoint.y_m"),
                        target_speed_mps=optional_finite_float(
                            item.get("target_speed_mps"), minimum=0.0
                        ),
                    )
                )
            else:
                raise TypeError(
                    "Route waypoints must be WaypointState, Waypoint, or mapping"
                )
        if not points:
            raise ValueError("set_route requires at least one waypoint")

        route = RouteState(
            route_id=route_id or f"route:{uuid.uuid4().hex[:12]}",
            waypoints=tuple(points),
            active_index=0,
            completed=False,
            planner_name=str(planner_name),
            planned_at_monotonic=time.monotonic(),
            goal_tolerance_m=goal_tolerance_m,
            metadata=dict(metadata or {}),
        )
        self.world_model.update(
            route=route,
            event_type="route.updated",
            event_payload={
                "route_id": route.route_id,
                "waypoints": len(route.waypoints),
                "planner": route.planner_name,
            },
        )
        self.shared_memory.set(
            K_PLAN_CURRENT,
            [
                {
                    "x": point.x_m,
                    "y": point.y_m,
                    "target_speed_mps": point.target_speed_mps,
                }
                for point in points
            ],
        )
        return route

    def plan_local_path(
        self,
        occupancy: OccupancyGrid,
        start: Tuple[float, float],
        goal: Tuple[float, float],
        *,
        inflation_radius_m: Optional[float] = None,
    ) -> list[Waypoint]:
        """Run the vehicle-local A* fallback on a supplied occupancy map."""

        radius = inflation_radius_m
        if radius is None:
            radius = get_config_section("robocar", self.config).get(
                "inflation_radius_m", 0.0
            )
        radius_value = require_finite_float(
            radius, "robocar.inflation_radius_m", minimum=0.0
        )
        path = astar_path(
            occupancy,
            start,
            goal,
            inflation_radius_m=radius_value,
        )
        self.shared_memory.set(K_MAP_LATEST, occupancy.to_dict())
        self.set_route(
            [point.to_waypoint_state() for point in path],
            planner_name="robocar.local_astar",
            metadata={"inflation_radius_m": radius_value},
        )
        return path

    def plan_with_slai(self, planning_task: Any) -> Any:
        """Delegate a current SLAI Planning Task without inventing a remapping."""

        planner = self.agent("planning")
        return planner.generate_plan(planning_task)

    def compute_trajectory_command(
        self,
        *,
        desired_speed_mps: Optional[float] = None,
    ) -> TrajectoryCommand:
        """Compute one deterministic control command without writing hardware."""

        snapshot = self.world_model.snapshot()
        if snapshot.pose is None:
            raise RuntimeError("Trajectory control requires a current PoseState")
        route = snapshot.route
        if not route.waypoints:
            raise RuntimeError("Trajectory control requires a non-empty RouteState")

        requested = desired_speed_mps
        if requested is None:
            requested = route.waypoints[route.active_index].target_speed_mps
        if requested is None:
            raise ValueError(
                "desired_speed_mps is required because the active waypoint has "
                "no target_speed_mps"
            )
        requested_value = require_finite_float(
            requested, "desired_speed_mps", minimum=0.0
        )

        started = time.monotonic()
        command = self.trajectory_controller.compute(
            pose=snapshot.pose,
            path=route.waypoints,
            desired_speed_mps=requested_value,
            measured_speed_mps=self.encoder.get_speed(),
            active_index=route.active_index,
        )
        self._last_control_cycle_duration_s = time.monotonic() - started

        updated_route = RouteState(
            route_id=route.route_id,
            waypoints=route.waypoints,
            active_index=command.target_index,
            completed=command.goal_reached,
            planner_name=route.planner_name,
            planned_at_monotonic=route.planned_at_monotonic,
            goal_tolerance_m=route.goal_tolerance_m,
            metadata=route.metadata,
        )
        current_autonomy = snapshot.autonomy
        self.world_model.update(
            route=updated_route,
            autonomy=AutonomyState(
                mode=current_autonomy.mode,
                run_id=current_autonomy.run_id,
                goal_id=current_autonomy.goal_id,
                cycle=current_autonomy.cycle,
                planner_status=current_autonomy.planner_status,
                last_plan_monotonic=current_autonomy.last_plan_monotonic,
                last_control_cycle_monotonic=time.monotonic(),
                last_recovery_monotonic=current_autonomy.last_recovery_monotonic,
                metadata=current_autonomy.metadata,
            ),
            event_type="trajectory.command_computed",
            event_payload={
                "target_index": command.target_index,
                "goal_reached": command.goal_reached,
                "cross_track_error_m": command.cross_track_error_m,
                "heading_error_rad": command.heading_error_rad,
            },
        )
        target_heading = snapshot.pose.yaw_rad + command.heading_error_rad
        self._observe_kpis(
            target_heading_rad=target_heading,
            control_loop_duration_s=self._last_control_cycle_duration_s,
        )
        return command

    def execute_trajectory_step(
        self,
        *,
        desired_speed_mps: Optional[float] = None,
        duration_s: float,
        source: str = "trajectory_controller",
    ) -> Dict[str, Any]:
        """Compute and execute one bounded trajectory-control step."""
    
        duration = require_finite_float(duration_s, "trajectory.duration_s", minimum=1e-9)
        command = self.compute_trajectory_command(desired_speed_mps=desired_speed_mps)
    
        if command.goal_reached:
            parking = self.park(reason="trajectory_goal_reached")
    
            return {
                "status": "completed",
                "goal_reached": True,
                "command": _serialize(command),
                "parking": parking,
            }
    
        self._update_route_turn_lighting(command)
    
        result = self.execute_ackermann_action(
            throttle=command.throttle,
            steering=command.steering,
            duration=duration,
            source=source,
            require_slai_safety=True,
            allow_persistent=False,
        )
    
        return {
            "status": result.get("status", "unknown"),
            "goal_reached": False,
            "command": _serialize(command),
            "execution": result,
        }

    # ------------------------------------------------------------------
    # Watchdog and deterministic service cadence
    # ------------------------------------------------------------------

    def check_watchdog(self, *, enforce: bool = False) -> WatchdogReport:
        snapshot = self.world_model.snapshot()
        motion = self.motion.get_status()
        actuator_status = str(motion.get("status", ""))
        actuator_fault = actuator_status.lower() in {
            "failed",
            "fault",
            "faulty",
            "critical",
        }
        runtime_reference = self._runtime_started_monotonic
        sensor_reference = (
            self._last_sensor_frame_monotonic
            if self._last_sensor_frame_monotonic is not None
            else runtime_reference
        )
        heartbeat_reference = (
            self._last_pico_heartbeat_monotonic
            if self._last_pico_heartbeat_monotonic is not None
            else runtime_reference
        )
        gnss_reference = (
            self._last_gnss_fix_monotonic
            if self._last_gnss_fix_monotonic is not None
            else runtime_reference
        )
        planner_reference = (
            snapshot.autonomy.last_plan_monotonic
            if snapshot.autonomy.last_plan_monotonic is not None
            else self._autonomy_started_monotonic
        )
        report = self.watchdog.check(
            WatchdogInputs(
                now_monotonic=time.monotonic(),
                last_sensor_frame_monotonic=sensor_reference,
                last_pico_heartbeat_monotonic=heartbeat_reference,
                last_control_cycle_duration_s=self._last_control_cycle_duration_s,
                actuator_status=actuator_status,
                actuator_fault=actuator_fault,
                last_gnss_fix_monotonic=gnss_reference,
                gnss_required=self._gnss_required,
                last_plan_monotonic=planner_reference,
                planner_required=(
                    self._planner_required
                    and snapshot.autonomy.mode == OperatingMode.AUTONOMOUS
                ),
            )
        )
        self.shared_memory.set(K_WATCHDOG_LATEST, report.to_dict())

        if enforce:
            try:
                VehicleWatchdog.enforce(
                    report,
                    stop_callback=lambda: self._safe_hardware_stop(
                        "watchdog_critical"
                    ),
                    event_callback=self._record_watchdog_event,
                    recovery_callback=self._recover_watchdog,
                )
            except Exception as exc:
                # VehicleWatchdog correctly attempts STOP before callbacks.  If
                # the hardware stop itself fails, record/latch the critical
                # events here and still invoke HandlerAgent with an explicit
                # stop-unconfirmed failure before propagating the exception.
                for event in report.events:
                    self._record_watchdog_event(event)
                self.handle_failure(
                    exc,
                    source="watchdog_stop_unconfirmed",
                    target_agent=self._recovery_target,
                    task_data={
                        "operation": "actuator_fault",
                        "watchdog_events": [event.code for event in report.events],
                        "report": report.to_dict(),
                    },
                    safe_stop=False,
                )
                raise
        return report

    def service(self) -> WatchdogReport:
        """Run one deterministic supervisory service iteration.
    
        Call this from the outer process at a stable cadence, such as the existing
        ``rc_main`` loop. No additional watchdog or lighting thread is introduced.
        """
    
        self.sensor_bus.service_lighting()
        return self.check_watchdog(enforce=True)

    def _record_watchdog_event(self, event: WatchdogEvent) -> None:
        severity = event.severity.value
        self.world_model.record_event(
            f"watchdog.{event.code}",
            payload={
                "message": event.message,
                "age_or_value": event.age_or_value,
                "threshold": event.threshold,
            },
            severity=severity,
        )
        if event.severity == WatchdogSeverity.CRITICAL:
            current = self.shared_memory.get(K_SAFETY_STATE, default={})
            state = dict(current) if isinstance(current, Mapping) else {}
            state.update(
                {
                    "estop": True,
                    "reason": f"watchdog:{event.code}",
                    "updated_at": time.time(),
                }
            )
            self.shared_memory.set(K_SAFETY_STATE, state)
            snap = self.world_model.snapshot()
            self.world_model.update(
                safety=SafetyState(
                    estop_latched=True,
                    allowed_to_move=False,
                    degraded=True,
                    speed_cap_mps=snap.safety.speed_cap_mps,
                    reasons=tuple(snap.safety.reasons)
                    + (f"watchdog:{event.code}",),
                    warnings=snap.safety.warnings,
                ),
                autonomy=AutonomyState(
                    mode=OperatingMode.EMERGENCY_STOP,
                    run_id=snap.autonomy.run_id,
                    goal_id=snap.autonomy.goal_id,
                    cycle=snap.autonomy.cycle,
                    planner_status=snap.autonomy.planner_status,
                    last_plan_monotonic=snap.autonomy.last_plan_monotonic,
                    last_control_cycle_monotonic=(
                        snap.autonomy.last_control_cycle_monotonic
                    ),
                    last_recovery_monotonic=snap.autonomy.last_recovery_monotonic,
                    metadata=snap.autonomy.metadata,
                ),
            )
        self._emit_observability(
            event=f"watchdog_{event.code}",
            payload={
                "severity": severity,
                "message": event.message,
                "age_or_value": event.age_or_value,
                "threshold": event.threshold,
            },
            severity=severity,
        )

    def _recover_watchdog(self, report: WatchdogReport) -> None:
        codes = [event.code for event in report.events]
        error = RuntimeError(
            "RoboCar watchdog critical event(s): " + ", ".join(codes)
        )
        self.handle_failure(
            error,
            source="watchdog",
            target_agent=self._recovery_target,
            task_data={
                "operation": "recover_watchdog",
                "watchdog_events": codes,
                "report": report.to_dict(),
            },
            # VehicleWatchdog.enforce already applied the hardware stop.
            safe_stop=False,
        )


    def _autonomy_failure_handler(
        self,
        payload: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Stop first, then route a failed SLAI stage through HandlerAgent.

        Unlike the generic stage adapter, this mapping also handles an explicit
        failed *result* (not only an exception), so HandlerAgent is not called
        with ``target_agent=None`` for an Execution/Evaluation denial/failure.
        """

        failed_stage = str(payload.get("failed_stage", "unknown")).strip().lower()
        stage_agents = {
            "reason": "reasoning",
            "plan": "planning",
            "authorize": "safety",
            "execute": "execution",
            "evaluate": "evaluation",
        }
        target_name = stage_agents.get(failed_stage)
        target: Any = self._recovery_target
        if target_name is not None:
            try:
                if target_name == "evaluation":
                    # Handler should see the real factory-managed SLAI agent,
                    # while the evaluation bridge remains the autonomy contract.
                    target = self._raw_agent("evaluation")
                else:
                    target = self.agent(target_name)
            except Exception:
                target = self._recovery_target

        raw_error = payload.get("error")
        error = (
            raw_error
            if isinstance(raw_error, BaseException)
            else RuntimeError(str(raw_error or f"autonomy stage {failed_stage} failed"))
        )
        return self.handle_failure(
            error,
            source=f"autonomy.{failed_stage}",
            target_agent=target,
            task_data={
                "goal": _serialize(payload.get("goal")),
                "failed_stage": failed_stage,
                "stage_output": _serialize(payload.get(failed_stage)),
                "run_id": payload.get("run_id"),
                "cycle": payload.get("cycle"),
            },
            safe_stop=True,
        )

    # ------------------------------------------------------------------
    # HandlerAgent / ObservabilityAgent / EvaluationAgent integration
    # ------------------------------------------------------------------

    def handle_failure(
        self,
        error: BaseException,
        *,
        source: str,
        target_agent: Any = None,
        task_data: Any = None,
        safe_stop: bool = True,
    ) -> Dict[str, Any]:
        """Enter a safe/degraded state, then delegate recovery to HandlerAgent."""

        stop_error: Optional[str] = None
        if safe_stop:
            try:
                self._safe_hardware_stop(f"handler:{source}")
            except Exception as exc:
                stop_error = f"{type(exc).__name__}: {exc}"

        self.kpi_tracker.record_recovery()
        self._enter_degraded_mode(
            reason=f"failure:{source}",
            event_type="handler.recovery_requested",
        )
        target = target_agent if target_agent is not None else self._recovery_target

        try:
            handler = self.agent("handler")
            result = handler.perform_task(
                {
                    "error": error,
                    "target_agent": target,
                    "task_data": task_data,
                    "context": {
                        "source": "robocar",
                        "route": source,
                        "agent": getattr(target, "name", type(target).__name__),
                        "safe_stop_applied": stop_error is None,
                        "safe_stop_error": stop_error,
                    },
                }
            )
            normalized = dict(result) if isinstance(result, Mapping) else {
                "status": "failed",
                "reason": "handler_non_mapping_result",
                "result_type": type(result).__name__,
            }
        except Exception as exc:
            normalized = {
                "status": "failed",
                "reason": "handler_unavailable_or_failed",
                "error": f"{type(exc).__name__}: {exc}",
                "safe_stop_error": stop_error,
            }
            self._agent_errors["handler"] = normalized["error"]

        self._last_handler_result = normalized
        self.shared_memory.set(K_HANDLER_LATEST, _serialize(normalized))
        self._emit_observability(
            event="handler_recovery",
            payload={
                "source": source,
                "handler_status": normalized.get("status"),
                "safe_stop_error": stop_error,
            },
            severity=(
                "warning"
                if str(normalized.get("status", "")).lower()
                in {"ok", "recovered", "degraded"}
                else "critical"
            ),
        )
        return normalized

    def _enter_degraded_mode(self, *, reason: str, event_type: str) -> None:
        snap = self.world_model.snapshot()
        self.world_model.update(
            safety=SafetyState(
                estop_latched=snap.safety.estop_latched,
                allowed_to_move=False,
                degraded=True,
                speed_cap_mps=snap.safety.speed_cap_mps,
                reasons=tuple(snap.safety.reasons) + (str(reason),),
                warnings=snap.safety.warnings,
            ),
            autonomy=AutonomyState(
                mode=(
                    OperatingMode.EMERGENCY_STOP
                    if snap.safety.estop_latched
                    else OperatingMode.DEGRADED
                ),
                run_id=snap.autonomy.run_id,
                goal_id=snap.autonomy.goal_id,
                cycle=snap.autonomy.cycle,
                planner_status=snap.autonomy.planner_status,
                last_plan_monotonic=snap.autonomy.last_plan_monotonic,
                last_control_cycle_monotonic=snap.autonomy.last_control_cycle_monotonic,
                last_recovery_monotonic=time.monotonic(),
                metadata=snap.autonomy.metadata,
            ),
            event_type=event_type,
            event_payload={"reason": reason},
            event_severity="warning",
        )

    def _emit_observability(
        self,
        *,
        event: str,
        payload: Optional[Mapping[str, Any]] = None,
        severity: str = "info",
        duration_ms: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Best-effort bounded report to the current ObservabilityAgent."""

        try:
            observability = self.agent("observability")
            sensor = self.sensor_bus.health()
            latencies = []
            if duration_ms is not None and math.isfinite(float(duration_ms)):
                latencies.append(
                    {
                        "subject": f"robocar.{event}",
                        "duration_ms": float(duration_ms),
                        "status": "ok" if severity == "info" else severity,
                    }
                )
            report = observability.perform_task(
                {
                    "task_name": "robocar_runtime",
                    "agent_name": "RoboCar",
                    "operation_name": str(event),
                    "source": "robocar",
                    "latencies": latencies,
                    "throughput": [
                        {
                            "subject": "robocar.sensor_bus",
                            "count": int(sensor.get("frames_received", 0) or 0),
                            "failure_count": int(sensor.get("parse_errors", 0) or 0)
                            + int(sensor.get("transport_errors", 0) or 0),
                        }
                    ],
                    "events": [
                        {
                            "event": str(event),
                            "severity": str(severity),
                            "payload": _serialize(dict(payload or {})),
                        }
                    ],
                }
            )
            normalized = dict(report) if isinstance(report, Mapping) else {
                "status": "unknown",
                "result": _serialize(report),
            }
            self._last_observability_result = normalized
            self.shared_memory.set(K_OBSERVABILITY_LATEST, normalized)
            self._agent_errors.pop("observability", None)
            return normalized
        except Exception as exc:
            # Telemetry must never become authority over motion.
            self._agent_errors["observability"] = (
                f"{type(exc).__name__}: {exc}"
            )
            return None

    def evaluate_now(
        self,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run one explicit RoboCar-scoped EvaluationAgent cycle."""

        evaluation = self.agent("evaluation")
        payload = dict(params or {})
        payload.setdefault("robocar_kpis", self.kpi_tracker.snapshot().to_dict())
        payload.setdefault("world_state", self.world_model.snapshot().to_dict())
        report = evaluation.execute_validation_cycle(payload)
        normalized = dict(report) if isinstance(report, Mapping) else {
            "status": "failed",
            "completed": False,
            "reason": "evaluation_non_mapping_result",
        }
        self.shared_memory.set(K_EVALUATION_LATEST, normalized)
        return normalized

    # ------------------------------------------------------------------
    # Bounded adaptation bridge (outside motion-critical loop)
    # ------------------------------------------------------------------

    def propose_adaptation(
        self,
        *,
        parameter: str,
        current_value: float,
        proposed_value: float,
        evidence_samples: int,
        confidence: float,
        source: str,
        reason: str,
        evidence: Optional[Mapping[str, Any]] = None,
    ) -> AdaptationAuditRecord:
        record = self.adaptation_guard.propose(
            parameter=parameter,
            current_value=current_value,
            proposed_value=proposed_value,
            evidence_samples=evidence_samples,
            confidence=confidence,
            source=source,
            reason=reason,
            evidence=evidence,
        )
        self.shared_memory.set(K_ADAPTATION_LAST, record.to_dict())
        return record

    def review_adaptation(
        self,
        proposal_id: str,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AdaptationAuditRecord:
        record = self.adaptation_guard.safety_approve(
            proposal_id,
            self.agent("safety").validate_action,
            context={
                "world": self.world_model.snapshot().to_dict(),
                **dict(context or {}),
            },
        )
        self.shared_memory.set(K_ADAPTATION_LAST, record.to_dict())
        return record

    def apply_adaptation(
        self,
        proposal_id: str,
        *,
        getter: Callable[[str], float],
        setter: Callable[[str, float], None],
    ) -> AdaptationAuditRecord:
        record = self.adaptation_guard.apply(
            proposal_id,
            getter=getter,
            setter=setter,
            require_safety_approval=True,
        )
        self.shared_memory.set(K_ADAPTATION_LAST, record.to_dict())
        return record

    def rollback_adaptation(
        self,
        proposal_id: str,
        *,
        setter: Callable[[str, float], None],
    ) -> AdaptationAuditRecord:
        record = self.adaptation_guard.rollback(
            proposal_id,
            setter=setter,
        )
        self.shared_memory.set(K_ADAPTATION_LAST, record.to_dict())
        return record

    # ------------------------------------------------------------------
    # SLAI outer autonomy
    # ------------------------------------------------------------------

    def run_autonomous(
        self,
        goal: Mapping[str, Any],
        *,
        context: Optional[Mapping[str, Any]] = None,
        stop_on_finish: bool = True,
        require_calibrated_safety: bool = True,
    ) -> Dict[str, Any]:
        """Run one bounded mission through SLAI's current AutonomousControlLoop.

        The outer loop is mission/policy scale, not a 20-50 Hz steering loop.
        Physical execution tasks therefore must contain explicit bounded robot
        actions.  Continuous path following should call ``execute_trajectory_step``
        from a deterministic fast loop while SLAI owns mission-level decisions.
        """

        if not self._started:
            raise RuntimeError("RoboCar.start() must be called before autonomy")
        if not isinstance(goal, Mapping):
            raise TypeError(
                "Physical RoboCar autonomy requires a mapping-valued goal with "
                "explicit execution_task/evaluation_params"
            )
        if self.autonomy_loop is None:
            raise RuntimeError("AutonomousControlLoop is not initialized")

        self._validate_autonomy_goal(goal)
        if require_calibrated_safety:
            self._validate_autonomous_safety_readiness()
        self._require_autonomy_agents()

        preflight = self.check_watchdog(enforce=True)
        if preflight.requires_stop:
            return {
                "state": "blocked",
                "succeeded": False,
                "reason": "watchdog_preflight_failed",
                "watchdog": preflight.to_dict(),
            }

        self.shared_memory.set(K_GOAL_CURRENT, _serialize(goal))
        snap = self.world_model.snapshot()
        self.world_model.update(
            autonomy=AutonomyState(
                mode=OperatingMode.AUTONOMOUS,
                run_id=None,
                goal_id=str(goal.get("id") or "") or None,
                cycle=0,
                planner_status=snap.autonomy.planner_status,
                last_plan_monotonic=snap.autonomy.last_plan_monotonic,
                last_control_cycle_monotonic=snap.autonomy.last_control_cycle_monotonic,
                last_recovery_monotonic=snap.autonomy.last_recovery_monotonic,
                metadata={"objective": goal.get("objective")},
            ),
            event_type="autonomy.requested",
            event_payload={"goal": _serialize(goal)},
        )

        started = time.monotonic()
        self._autonomy_started_monotonic = started
        normalized: Dict[str, Any] = {
            "state": "failed",
            "succeeded": False,
            "reason": "autonomy_result_unavailable",
        }
        try:
            result_obj = self.autonomy_loop.run(goal, context=context)
            result = result_obj.to_dict() if callable(
                getattr(result_obj, "to_dict", None)
            ) else _serialize(result_obj)
            if not isinstance(result, Mapping):
                result = {"state": "failed", "result": _serialize(result)}
            normalized = dict(result)
        except Exception as exc:
            normalized = {
                "state": "failed",
                "succeeded": False,
                "reason": f"{type(exc).__name__}: {exc}",
            }
            self.handle_failure(
                exc,
                source="autonomous_control_loop",
                target_agent=self._recovery_target,
                task_data=goal,
                safe_stop=True,
            )
        finally:
            duration_ms = (time.monotonic() - started) * 1000.0
            final_stop_error: Optional[str] = None
            if stop_on_finish:
                try:
                    self._safe_hardware_stop("autonomy_run_finished")
                except Exception as exc:
                    final_stop_error = f"{type(exc).__name__}: {exc}"
                    self._last_error = f"autonomy_final_stop: {final_stop_error}"
                    if "normalized" in locals():
                        normalized = dict(normalized)
                        normalized.update(
                            {
                                "state": "failed",
                                "succeeded": False,
                                "reason": "autonomy_final_stop_unconfirmed",
                                "final_stop_error": final_stop_error,
                            }
                        )
            self._emit_observability(
                event="autonomy_run_finished",
                payload={
                    "result": locals().get("normalized", {}),
                    "kpis": self.kpi_tracker.snapshot().to_dict(),
                    "final_stop_error": final_stop_error,
                },
                severity="critical" if final_stop_error else "info",
                duration_ms=duration_ms,
            )

        self._autonomy_started_monotonic = None
        self._last_autonomy_result = dict(normalized)
        self.shared_memory.set(K_AUTONOMY_LAST, _serialize(normalized))
        terminal_state = str(normalized.get("state", "failed")).lower()
        mode = (
            OperatingMode.STOPPED
            if terminal_state in {"succeeded", "stopped", "blocked", "review_required"}
            else OperatingMode.DEGRADED
        )
        current = self.world_model.snapshot()
        self.world_model.update(
            autonomy=AutonomyState(
                mode=mode,
                run_id=normalized.get("run_id"),
                goal_id=normalized.get("goal_id"),
                cycle=(
                    len(normalized.get("cycles", []))
                    if isinstance(normalized.get("cycles"), Sequence)
                    else current.autonomy.cycle
                ),
                planner_status=current.autonomy.planner_status,
                last_plan_monotonic=current.autonomy.last_plan_monotonic,
                last_control_cycle_monotonic=current.autonomy.last_control_cycle_monotonic,
                last_recovery_monotonic=current.autonomy.last_recovery_monotonic,
                metadata={"terminal_reason": normalized.get("reason")},
            ),
            event_type="autonomy.finished",
            event_payload={
                "state": terminal_state,
                "reason": normalized.get("reason"),
            },
            event_severity=(
                "info" if terminal_state == "succeeded" else "warning"
            ),
        )
        return normalized

    def _validate_autonomy_goal(self, goal: Mapping[str, Any]) -> None:
        objective = goal.get("objective") or goal.get("goal") or goal.get("name")
        if not str(objective or "").strip():
            raise ValueError("Autonomous goal requires objective, goal, or name")

        execution_task = goal.get("execution_task")
        if not isinstance(execution_task, Mapping):
            raise ValueError(
                "Physical autonomous goal requires explicit mapping execution_task"
            )
        sequence = execution_task.get("action_sequence")
        if (
            not isinstance(sequence, Sequence)
            or isinstance(sequence, (str, bytes))
            or not sequence
        ):
            raise ValueError(
                "Physical autonomous execution_task requires a non-empty action_sequence"
            )

        for index, raw in enumerate(sequence):
            if not isinstance(raw, Mapping):
                raise ValueError(f"action_sequence[{index}] must be a mapping")
            name = _normalize_action_name(raw)
            if name not in _REGISTERED_ROBOT_ACTIONS:
                raise ValueError(
                    f"action_sequence[{index}] uses unsupported RoboCar action {name!r}; "
                    f"supported={sorted(_REGISTERED_ROBOT_ACTIONS)}"
                )
            if name == "ackermann":
                throttle = normalize_signed_command(
                    raw.get("throttle", 0.0),
                    f"action_sequence[{index}].throttle",
                )
                normalize_signed_command(
                    raw.get("steering", 0.0),
                    f"action_sequence[{index}].steering",
                )
                duration = optional_finite_float(raw.get("duration", 0.0), minimum=0.0)
                if duration is None:
                    raise ValueError(
                        f"action_sequence[{index}].duration must be finite and non-negative"
                    )
                if abs(throttle) > 1e-6 and duration <= 0.0:
                    raise ValueError(
                        f"action_sequence[{index}] non-zero throttle requires duration > 0"
                    )

        if not isinstance(goal.get("evaluation_params"), Mapping):
            raise ValueError(
                "Autonomous goal requires evaluation_params because the current "
                "SLAI outer loop requires an explicit evaluation stage contract"
            )

    def _validate_autonomous_safety_readiness(self) -> None:
        if self.allow_simulation:
            # Simulation callers may still demand calibrated values by keeping
            # this flag enabled; no automatic exemption is assumed here.
            pass
        missing: list[str] = []
        if self.local_safety.front_stop_distance_m is None:
            missing.append("robocar.front_stop_distance_m")
        if self.local_safety.sensor_max_age_s is None:
            missing.append("robocar.sensor_max_age_s")
        if self.watchdog.thresholds.sensor_frame_timeout_s is None:
            missing.append("watchdog.sensor_frame_timeout_s")
        if missing:
            raise RuntimeError(
                "Autonomous physical motion is fail-closed until calibrated "
                "safety values are configured: " + ", ".join(missing)
            )

        reading = self.sensor_bus.latest()
        if reading is None:
            raise RuntimeError(
                "Autonomous physical motion requires at least one real/current "
                "SensorBus frame before mission execution"
            )
        age = max(0.0, time.time() - reading.t)
        if (
            self.local_safety.sensor_max_age_s is not None
            and age > self.local_safety.sensor_max_age_s
        ):
            raise RuntimeError(
                "Autonomous physical motion requires a fresh SensorBus frame; "
                f"age={age:.6f}s limit={self.local_safety.sensor_max_age_s:.6f}s"
            )

        snapshot = self.world_model.snapshot()
        if snapshot.safety.estop_latched:
            raise RuntimeError(
                "Autonomous physical motion is blocked by the latched emergency stop"
            )
        if self._gnss_required and not snapshot.gnss.valid:
            raise RuntimeError(
                "watchdog.gnss_required=true but no valid GNSS fix is present"
            )

    def _require_autonomy_agents(self) -> None:
        required = (
            "safety",
            "execution",
            "reasoning",
            "planning",
            "handler",
            "observability",
            "evaluation",
        )
        failures: Dict[str, str] = {}
        for name in required:
            try:
                self.agent(name)
                self._agent_errors.pop(name, None)
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"
                self._agent_errors[name] = failures[name]
        if failures:
            raise RuntimeError(
                "Required RoboCar autonomy agents unavailable: "
                + "; ".join(f"{k}={v}" for k, v in failures.items())
            )

    # ------------------------------------------------------------------
    # Health / diagnostics
    # ------------------------------------------------------------------

    def _agent_health(self, name: str) -> Any:
        agent = self._agents.get(name) or self._raw_agents.get(name)
        if agent is None:
            return None
        for method_name in (
            "health",
            "health_check",
            "get_health_report",
            "get_overall_system_health",
        ):
            method = getattr(agent, method_name, None)
            if not callable(method):
                continue
            try:
                return _serialize(method())
            except Exception as exc:
                return {"status": "degraded", "error": f"{type(exc).__name__}: {exc}"}
        return {"status": "available", "type": type(agent).__name__}

    def _core_health_snapshot(self) -> Dict[str, Any]:
        return {
            "started": self._started,
            "simulation_allowed": self.allow_simulation,
            "sensor_bus": self.sensor_bus.health(),
            "motion": self.motion.get_status(),
            "encoder": self.encoder.health(),
            "world_revision": self.world_model.snapshot().revision,
            "kpis": self.kpi_tracker.snapshot().to_dict(),
            "watchdog": self.watchdog.health(),
            "last_error": self._last_error,
        }

    def health(self) -> Dict[str, Any]:
        watchdog_report = self.check_watchdog(enforce=False)
        loop_health = None
        if self.autonomy_loop is not None:
            method = getattr(self.autonomy_loop, "health", None)
            if callable(method):
                try:
                    loop_health = _serialize(method())
                except Exception as exc:
                    loop_health = {"status": "degraded", "error": str(exc)}

        return {
            **self._core_health_snapshot(),
            "world": self.world_model.snapshot().to_dict(),
            "watchdog_report": watchdog_report.to_dict(),
            "autonomy_loop": loop_health,
            "agents_initialized": sorted(self._agents),
            "raw_agents_initialized": sorted(self._raw_agents),
            "agent_errors": dict(self._agent_errors),
            "agents": {
                name: self._agent_health(name)
                for name in (
                    "safety",
                    "execution",
                    "handler",
                    "observability",
                    "evaluation",
                    "reasoning",
                    "planning",
                    "knowledge",
                )
                if name in self._agents or name in self._raw_agents
            },
            "last_handler_result": _serialize(self._last_handler_result),
            "last_observability_result": _serialize(
                self._last_observability_result
            ),
            "last_autonomy_result": _serialize(self._last_autonomy_result),
        }

    def _public_config_snapshot(self) -> Dict[str, Any]:
        # Keep only RoboCar-owned sections and omit loader internals.
        return {
            key: value
            for key, value in self.config.items()
            if key
            in {
                "main",
                "encoder",
                "motion",
                "speed",
                "hardware",
                "power",
                "robocar",
                "lighting",
                "watchdog",
                "kpi",
                "adaptation",
            }
        }


# ---------------------------------------------------------------------------
# Local A* fallback
# ---------------------------------------------------------------------------


def astar_path(
    occupancy: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    *,
    inflation_radius_m: float = 0.0,
    occupied_threshold: int = 50,
    treat_unknown_as_obstacle: bool = True,
) -> list[Waypoint]:
    """Compute an 8-connected collision-free path on an inflated occupancy grid."""

    start = occupancy.world_to_cell(*start_world)
    goal = occupancy.world_to_cell(*goal_world)
    if not occupancy.in_bounds(start) or not occupancy.in_bounds(goal):
        raise PlanningError(
            "A*", start_world, goal_world, "start_or_goal_out_of_bounds"
        )

    distance_map, _ = distance_map_from_occupancy(
        occupancy.to_dict(),
        occupied_threshold=occupied_threshold,
        treat_unknown_as_obstacle=treat_unknown_as_obstacle,
    )
    inflated = inflate_obstacles(distance_map, inflation_radius_m)

    def blocked(cell: Tuple[int, int]) -> bool:
        x, y = cell
        try:
            return bool(inflated[y][x])
        except Exception:
            return bool(inflated[y][x])

    if blocked(start) or blocked(goal):
        raise PlanningError(
            "A*", start_world, goal_world, "start_or_goal_in_obstacle"
        )

    neighbor_steps = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    frontier: list[Tuple[float, Tuple[int, int]]] = [(0.0, start)]
    came_from: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    g_score: Dict[Tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break
        current_g = g_score[current]
        for dx, dy, step_cost in neighbor_steps:
            nxt = (current[0] + dx, current[1] + dy)
            if not occupancy.in_bounds(nxt) or blocked(nxt):
                continue
            # Prevent diagonal corner cutting through occupied orthogonal cells.
            if dx and dy:
                if blocked((current[0] + dx, current[1])) or blocked(
                    (current[0], current[1] + dy)
                ):
                    continue
            candidate = current_g + step_cost
            if candidate >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = candidate
            came_from[nxt] = current
            heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
            heapq.heappush(frontier, (candidate + heuristic, nxt))

    if goal not in came_from:
        raise PlanningError(
            "A*", start_world, goal_world, "no_collision_free_path"
        )

    cells: list[Tuple[int, int]] = []
    cursor: Optional[Tuple[int, int]] = goal
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    cells.reverse()
    return [occupancy.cell_to_world(cell) for cell in cells]


__all__ = [
    "MEM_FILE",
    "DEFAULT_ROBOCAR_CONFIG",
    "K_MAP_LATEST",
    "K_DETECTIONS_SIGNS",
    "K_GOAL_CURRENT",
    "K_PLAN_CURRENT",
    "K_ROUTE_TRAVELED",
    "K_SAFETY_STATE",
    "K_DIRECTIVES",
    "K_POSE_ESTIMATE",
    "K_CONFIG",
    "K_SENSOR_LATEST",
    "K_ENCODER_TICKS",
    "K_ENCODER_SPEED",
    "K_ULTRA_FRONT",
    "K_ULTRA_REAR",
    "K_TOF_MM",
    "K_BATTERY_VOLT",
    "K_BATTERY_STATE",
    "K_WORLD_STATE",
    "K_KPI_LATEST",
    "K_WATCHDOG_LATEST",
    "K_AUTONOMY_LAST",
    "K_OBSERVABILITY_LATEST",
    "K_HANDLER_LATEST",
    "K_EVALUATION_LATEST",
    "K_ADAPTATION_LAST",
    "Pose2D",
    "Waypoint",
    "OccupancyGrid",
    "SafetyDecision",
    "SafetyManager",
    "PurePursuit",
    "RoboCarEvaluationBridge",
    "RoboCarRecoveryTarget",
    "RoboCarRobotAdapter",
    "RoboCar",
    "astar_path",
]

if __name__ == "__main__":
    print("\n=== Running RoboCar Integrated Self-Test ===\n")
    printer.status("TEST", "Initializing full RoboCar integration self-test", "info")

    class _SelfTestPWM:
        """Safe in-memory actuator boundary."""

        def __init__(self):
            self.pulses_us = {}

        def write_us(self, channel: int, pulse_us: int) -> None:
            self.pulses_us[int(channel)] = int(pulse_us)

    class _SelfTestMotionController(MotionController):
        """Use the real controller logic over an in-memory PWM backend."""

        def __init__(self, *, config=None, allow_simulation=False, **kwargs):
            super().__init__(
                config=config,
                allow_simulation=True,
                pwm_backend=_SelfTestPWM(),
            )

    _RealMotionController = MotionController
    car = None

    try:
        # RoboCar resolves MotionController at construction time. Temporarily
        # substitute only the hardware backend; all controller logic remains real.
        MotionController = _SelfTestMotionController

        car = RoboCar(
            sensor_port="__robocar_selftest_missing_port__",
            allow_simulation=True,
            eager_support_agents=True,
        )
        car.start()

        print("\n* * * Phase 1 - Runtime / Sensors / Agents * * *\n")

        deadline = time.monotonic() + 1.0
        while car.sensor_bus.latest() is None and time.monotonic() < deadline:
            time.sleep(0.02)

        assert car.sensor_bus.latest() is not None
        assert car.sensor_bus.is_simulation
        assert {"safety", "execution"}.issubset(car._agents | car._raw_agents)

        printer.pretty(
            "RUNTIME",
            {
                "sensor": car.sensor_bus.health(),
                "motion": car.motion.get_status(),
                "agents": sorted(set(car._agents) | set(car._raw_agents)),
            },
            "success",
        )

        print("\n* * * Phase 2 - World Model / Planning / Control * * *\n")

        car.update_pose(
            PoseState(
                x_m=0.5,
                y_m=0.5,
                yaw_rad=0.0,
                speed_mps=0.0,
                confidence=1.0,
            )
        )

        grid = OccupancyGrid(
            width=5,
            height=5,
            resolution=1.0,
            grid=[0] * 25,
        )

        path = car.plan_local_path(
            grid,
            start=(0.5, 0.5),
            goal=(4.5, 4.5),
            inflation_radius_m=0.0,
        )

        assert len(path) >= 2

        car.set_route(
            [
                {"x_m": 0.5, "y_m": 0.5, "target_speed_mps": 0.20},
                {"x_m": 2.5, "y_m": 1.5, "target_speed_mps": 0.20},
                {"x_m": 4.5, "y_m": 4.5, "target_speed_mps": 0.0},
            ],
            route_id="selftest-route",
            planner_name="robocar.selftest",
        )

        command = car.compute_trajectory_command()

        assert math.isfinite(command.throttle)
        assert math.isfinite(command.steering)
        assert -1.0 <= command.throttle <= 1.0
        assert -1.0 <= command.steering <= 1.0

        printer.pretty(
            "PLANNING / CONTROL",
            {
                "astar_waypoints": len(path),
                "trajectory_command": _serialize(command),
                "world_revision": car.world_model.snapshot().revision,
            },
            "success",
        )

        print("\n* * * Phase 3 - Safety / Execution / Watchdog * * *\n")

        # Exercise SafetyAgent -> ExecutionAgent -> RoboCarRobotAdapter while
        # remaining physically neutral.
        execution = car.execute_ackermann_action(
            throttle=0.0,
            steering=0.0,
            duration=0.0,
            source="robocar.selftest",
        )

        watchdog = car.service()

        assert isinstance(execution, Mapping)
        assert car.motion.get_status()["throttle"] == 0.0
        assert isinstance(watchdog, WatchdogReport)

        printer.pretty(
            "SAFETY / EXECUTION",
            {
                "execution": execution,
                "watchdog": watchdog.to_dict(),
            },
            "success",
        )

        print("\n* * * Phase 4 - Adaptation / Observability / Evaluation * * *\n")

        # Current default configuration has no adaptation allowlist. A proposal
        # must therefore be rejected rather than silently becoming tunable.
        proposal = car.propose_adaptation(
            parameter="speed.kp",
            current_value=float(car.speed_controller.kp),
            proposed_value=float(car.speed_controller.kp) + 0.01,
            evidence_samples=1,
            confidence=1.0,
            source="robocar.selftest",
            reason="verify deny-by-default adaptation",
        )

        assert proposal.status.value == "rejected"

        observability = car._emit_observability(
            event="robocar_selftest",
            payload={
                "world_revision": car.world_model.snapshot().revision,
                "kpis": car.kpi_tracker.snapshot().to_dict(),
            },
        )

        assert isinstance(observability, Mapping)

        evaluation = car.evaluate_now(
            {
                "agent_performance_metrics": {},
            }
        )

        assert isinstance(evaluation, Mapping)

        printer.pretty(
            "INTELLIGENCE SERVICES",
            {
                "adaptation": proposal.to_dict(),
                "observability_status": observability.get("status"),
                "evaluation_status": evaluation.get("status"),
            },
            "success",
        )

        print("\n* * * Phase 5 - E-Stop / Handler Recovery * * *\n")

        estop = car.emergency_stop("robocar_selftest")
        assert car.world_model.snapshot().safety.estop_latched

        car.clear_emergency_stop(operator_confirmed=True)
        assert not car.world_model.snapshot().safety.estop_latched

        recovery = car.handle_failure(
            RuntimeError("controlled RoboCar self-test recovery probe"),
            source="robocar.selftest",
            target_agent=car._recovery_target,
            task_data={
                "operation": "safe_degraded_recovery",
            },
            safe_stop=True,
        )

        assert isinstance(recovery, Mapping)
        assert car.motion.get_status()["throttle"] == 0.0

        printer.pretty(
            "RECOVERY",
            {
                "emergency_stop": estop,
                "handler": recovery,
            },
            "success",
        )

        print("\n* * * Phase 6 - Final Health * * *\n")

        health = car.health()

        assert health["started"] is True
        assert health["world_revision"] > 0
        assert health["sensor_bus"]["frames_received"] > 0

        printer.pretty("FINAL HEALTH", health, "success")

    finally:
        if car is not None:
            car.close()

        # Restore the module-global class even if the self-test fails.
        MotionController = _RealMotionController

    print("\n=== RoboCar Integrated Self-Test Passed ===\n")