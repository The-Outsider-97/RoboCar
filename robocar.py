"""SLAI integration layer for the physical RoboCar package.

This module is intentionally a bridge, not a second general-purpose SLAI
orchestrator.  It owns vehicle-specific geometry, deterministic local safety,
sensor publication, a local A* fallback, and the hardware adapter consumed by
SLAI's existing ExecutionAgent robot actions.

The low-level command path is fail-closed:

    sensor/Pico -> SensorBus -> SharedMemory
                               -> local SafetyManager
                               -> SLAI SafetyAgent (high-level authorization)
                               -> SLAI ExecutionAgent/AckermannAction
                               -> RoboCarRobotAdapter
                               -> MotionController -> PCA9685 -> servo/ESC

Emergency stop bypasses agents and addresses the hardware boundary directly.
"""

from __future__ import annotations

import heapq
import math
import threading
import time
import uuid

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .main_sensor import SensorBus, SensorReading
from .motion_controller import MotionController, PIDSpeedController
from .wheel_encoder import WheelEncoder
from .modules.edt2d import distance_map_from_occupancy, inflate_obstacles
from .utils.config_loader import get_config_section, load_global_config
from .utils.rc_errors import *
from .utils.rc_helpers import *

# RoboCar is expected to be cloned at SLAI/RoboCar.  These are therefore
# absolute imports from the parent SLAI repository, not ``..src`` relatives.
from src.agents.agent_factory import AgentFactory
from src.agents.collaborative.shared_memory import SharedMemory
from src.agents.execution.actions.robot_actions import (
    AckermannAction,
    SensorReadAction,
    StopAction,
)
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI AI RC Car")
printer = PrettyPrinter()

MEM_FILE = "robot_memory.pkl"

# Shared-memory keys: one canonical vocabulary for this integration layer.
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


@dataclass(frozen=True, slots=True)
class Pose2D:
    x: float
    y: float
    theta: float
    v: float = 0.0

    def __post_init__(self) -> None:
        if not all(is_finite_number(v) for v in (self.x, self.y, self.theta, self.v)):
            raise ValueError("Pose2D values must be finite")


@dataclass(frozen=True, slots=True)
class Waypoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not all(is_finite_number(v) for v in (self.x, self.y)):
            raise ValueError("Waypoint values must be finite")


@dataclass(slots=True)
class OccupancyGrid:
    """Minimal occupancy-grid schema compatible with ``modules.edt2d``.

    Cells use the existing repository convention: values >= ``50`` are occupied
    and negative values are unknown.  Unknown space is treated as occupied by
    the local fallback planner unless explicitly changed by the caller.
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
                f"OccupancyGrid requires {self.width * self.height} cells, got {len(flattened)}"
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
    """Deterministic, dependency-light vehicle safety gate.

    It enforces only constraints that are actually represented by the RoboCar
    configuration or shared runtime state.  No unconfigured collision distance,
    throttle derating factor, or kinematic limit is fabricated here.
    """

    def __init__(self, config: Mapping[str, Any], shared_memory: SharedMemory) -> None:
        self.config = dict(config)
        self.shared_memory = shared_memory
        power = get_config_section("power", self.config)
        robocar = get_config_section("robocar", self.config)

        self.v_warn = require_finite_float(power.get("v_warn"), "power.v_warn", minimum=0.0)
        self.v_cutback = require_finite_float(power.get("v_cutback"), "power.v_cutback", minimum=0.0)
        self.v_critical = require_finite_float(power.get("v_critical"), "power.v_critical", minimum=0.0)
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

        if self.sensor_max_age_s is not None and reading is not None:
            if max(0.0, time.time() - reading.t) > self.sensor_max_age_s:
                reasons.append("sensor_frame_stale")
        elif self.sensor_max_age_s is not None and reading is None:
            reasons.append("sensor_frame_missing")

        voltage = reading.vbat if reading is not None else None
        power_state = self.battery_state(voltage)
        if power_state == "critical" and abs(throttle_value) > 1e-6:
            reasons.append("battery_voltage_critical")
        elif power_state in {"cutback", "warning"}:
            warnings.append(f"battery_voltage_{power_state}")

        if (
            self.front_stop_distance_m is not None
            and reading is not None
            and reading.ultra_front_m is not None
            and reading.ultra_front_m <= self.front_stop_distance_m
            and throttle_value > 0.0
        ):
            reasons.append("front_obstacle_inside_configured_stop_distance")

        speed_cap = optional_finite_float(state.get("speed_cap"), minimum=0.0)
        if speed_cap is not None and speed_mps is not None and speed_mps >= speed_cap and throttle_value > 0.0:
            reasons.append("configured_speed_cap_reached")

        directive_limit = optional_finite_float(directives.get("limit_speed"), minimum=0.0)
        if directive_limit is not None and speed_mps is not None and speed_mps >= directive_limit and throttle_value > 0.0:
            reasons.append("reasoning_speed_limit_reached")

        allowed = not reasons
        return SafetyDecision(
            allowed=allowed,
            throttle=throttle_value if allowed else 0.0,
            steering=steering_value,
            reasons=tuple(reasons),
            warnings=tuple(warnings),
        )


class PurePursuit:
    """Vehicle-local pure-pursuit steering geometry.

    This class returns normalized steering only.  It deliberately does not map a
    target speed in m/s to ESC throttle because the repository does not contain a
    calibrated speed-to-throttle model.
    """

    def __init__(self, lookahead_m: float, wheelbase_m: float, max_steer_rad: float) -> None:
        self.lookahead_m = require_finite_float(lookahead_m, "pure_pursuit.lookahead_m", minimum=1e-6)
        self.wheelbase_m = require_finite_float(wheelbase_m, "pure_pursuit.wheelbase_m", minimum=1e-6)
        self.max_steer_rad = require_finite_float(max_steer_rad, "pure_pursuit.max_steer_rad", minimum=1e-6)

    def compute_steering(self, pose: Pose2D, path: Sequence[Waypoint]) -> float:
        if not path:
            return 0.0
        target = path[-1]
        for waypoint in path:
            if math.hypot(waypoint.x - pose.x, waypoint.y - pose.y) >= self.lookahead_m:
                target = waypoint
                break
        heading = math.atan2(target.y - pose.y, target.x - pose.x)
        alpha = math.atan2(math.sin(heading - pose.theta), math.cos(heading - pose.theta))
        distance = max(1e-6, math.hypot(target.x - pose.x, target.y - pose.y))
        curvature = 2.0 * math.sin(alpha) / distance
        steer_rad = math.atan(self.wheelbase_m * curvature)
        return clamp(steer_rad / self.max_steer_rad, -1.0, 1.0)


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
            # AckermannAction calls steering before throttle.  Neutralizing the
            # ESC here prevents a steering update from inheriting stale throttle.
            decision = self._owner.local_safety.authorize_command(
                0.0,
                steering,
                reading=self._owner.sensor_bus.latest(),
                speed_mps=self._owner.encoder.get_speed(),
            )
            if not decision.allowed:
                self._owner.motion.stop()
                return False
            result = self._owner.motion.send(0.0, decision.steering)
            self._steering = decision.steering
            self._throttle = 0.0
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
                self._owner.motion.stop()
                return False
            result = self._owner.motion.send(decision.throttle, decision.steering)
            self._throttle = decision.throttle
            return result.get("ok") is True

    def stop(self) -> bool:
        with self._lock:
            result = self._owner.motion.stop()
            self._throttle = 0.0
            self._steering = 0.0
            return result.get("ok") is True

    def get_sensor_value(self, sensor_name: str) -> Any:
        return self._owner.get_sensor_value(sensor_name)

    def get_pose(self) -> Tuple[float, float, float]:
        raw = self._owner.shared_memory.get(K_POSE_ESTIMATE)
        if not isinstance(raw, Mapping):
            raise SensorError("pose_estimate", raw, ("finite x/y/theta", "finite x/y/theta"))
        values = (raw.get("x"), raw.get("y"), raw.get("theta"))
        if not all(is_finite_number(value) for value in values):
            raise SensorError("pose_estimate", values, ("finite", "finite"))
        return float(values[0]), float(values[1]), float(values[2])


class RoboCar:
    """Physical RoboCar integration boundary for SLAI v2.2."""

    def __init__(
        self,
        *,
        config_path: Optional[str] = None,
        sensor_port: Optional[str] = None,
        allow_simulation: bool = False,
        shared_memory: Optional[SharedMemory] = None,
        agent_factory: Optional[AgentFactory] = None,
    ) -> None:
        self.config = load_global_config(config_path)
        hardware = get_config_section("hardware", self.config)
        serial_cfg = hardware.get("pico_serial", {}) if isinstance(hardware.get("pico_serial"), Mapping) else {}
        port = sensor_port if sensor_port is not None else serial_cfg.get("port", "auto")
        baud = optional_int(serial_cfg.get("baud"), minimum=1) or 115200

        self.allow_simulation = bool(allow_simulation)
        self.shared_memory = shared_memory if shared_memory is not None else SharedMemory()
        self.agent_factory = agent_factory if agent_factory is not None else AgentFactory()
        self._owns_memory = shared_memory is None
        self._owns_factory = agent_factory is None
        self._started = False
        self._agents: Dict[str, Any] = {}
        self._last_error: Optional[str] = None

        self.motion = MotionController(config=self.config, allow_simulation=self.allow_simulation)
        self.speed_controller = PIDSpeedController(config=self.config)
        self.encoder = WheelEncoder(config=self.config)
        self.sensor_bus = SensorBus(
            port=str(port) if port is not None else "auto",
            baud=baud,
            allow_simulation=self.allow_simulation,
        )
        self.local_safety = SafetyManager(self.config, self.shared_memory)
        self.robot_adapter = RoboCarRobotAdapter(self)
        self.sensor_bus.subscribe(self._on_sensor_reading)

        robocar_cfg = get_config_section("robocar", self.config)
        motion_cfg = get_config_section("motion", self.config)
        self.pure_pursuit = PurePursuit(
            lookahead_m=robocar_cfg.get("lookahead"),
            wheelbase_m=robocar_cfg.get("wheelbase"),
            max_steer_rad=motion_cfg.get("servo_max_angle_rad"),
        )

    # ------------------------------------------------------------------
    # Lifecycle and SLAI integration
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        # Keep the physical actuator boundary neutral throughout startup.
        self.motion.stop()
        self.sensor_bus.start()
        if (self.sensor_bus.is_simulation or self.motion.simulation_mode) and not self.allow_simulation:
            raise RuntimeError("Simulation became active without explicit permission")
        self._initialize_required_agents()
        self.shared_memory.set(K_CONFIG, self._public_config_snapshot())
        self._started = True
        logger.info("RoboCar started (simulation=%s)", self.allow_simulation)

    def _initialize_required_agents(self) -> None:
        safety = self.agent("safety")
        execution = self.agent_factory.create(
            "execution",
            shared_memory=self.shared_memory,
            robot=self.robot_adapter,
        )
        self._agents["execution"] = execution

        # execution_agent.enabled_actions is currently empty/default-only in
        # SLAI's global configuration.  Register only actions that match this
        # Ackermann chassis.  Differential-drive Motor/Spin/Navigate are not
        # exposed because that would misrepresent the vehicle kinematics.
        for name, action_class in {
            "ackermann": AckermannAction,
            "stop": StopAction,
            "sensor_read": SensorReadAction,
        }.items():
            if name not in getattr(execution, "action_class_registry", {}):
                execution.register_action(name, action_class)

        if safety is None or execution is None:
            raise RuntimeError("Required SLAI agents could not be initialized")

    def agent(self, name: str) -> Any:
        normalized = str(name).strip().lower()
        if normalized in self._agents:
            return self._agents[normalized]
        agent = self.agent_factory.create(normalized, shared_memory=self.shared_memory)
        self._agents[normalized] = agent
        return agent

    def close(self) -> None:
        # Emergency-safe order: stop actuator first, then stop data producers and
        # agent-owned resources.  The global SharedMemory singleton is not closed
        # here because it may be shared with the parent SLAI process.
        try:
            self.motion.stop()
        except Exception as exc:
            self._last_error = f"stop_during_close: {type(exc).__name__}: {exc}"
            logger.critical("RoboCar failed to confirm neutral during close: %s", exc)
        self.sensor_bus.stop()
        execution = self._agents.get("execution")
        shutdown = getattr(execution, "shutdown", None)
        if callable(shutdown):
            try:
                shutdown()
            except Exception as exc:
                logger.warning("ExecutionAgent shutdown degraded: %s", exc)
        self.motion.close()
        self._started = False

    stop = close

    # ------------------------------------------------------------------
    # Sensor/state publication
    # ------------------------------------------------------------------
    def _on_sensor_reading(self, reading: SensorReading) -> None:
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
                self.shared_memory.set(K_BATTERY_STATE, self.local_safety.battery_state(reading.vbat))
            if reading.encoder_ticks_total is not None:
                self.shared_memory.set(K_ENCODER_TICKS, reading.encoder_ticks_total)
                speed = self.encoder.update_from_ticks(reading.encoder_ticks_total)
                self.shared_memory.set(K_ENCODER_SPEED, speed)
        except Exception as exc:
            self._last_error = f"sensor_publish: {type(exc).__name__}: {exc}"
            logger.exception("Failed to publish RoboCar sensor frame")

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
    # Safety and execution
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

    def emergency_stop(self, reason: str = "operator_or_safety_request") -> Dict[str, Any]:
        current = self.shared_memory.get(K_SAFETY_STATE, default={})
        state = dict(current) if isinstance(current, Mapping) else {}
        state.update({"estop": True, "reason": str(reason), "updated_at": time.time()})
        self.shared_memory.set(K_SAFETY_STATE, state)
        # Do not wait for an agent, planner, or network path to stop the car.
        result = self.motion.stop()
        return {"status": "stopped", "reason": reason, "hardware": result}

    def clear_emergency_stop(self) -> None:
        current = self.shared_memory.get(K_SAFETY_STATE, default={})
        state = dict(current) if isinstance(current, Mapping) else {}
        state.update({"estop": False, "updated_at": time.time()})
        self.shared_memory.set(K_SAFETY_STATE, state)

    def execute_ackermann_action(
        self,
        *,
        throttle: float,
        steering: float,
        duration: float = 0.0,
        source: str = "robocar",
        require_slai_safety: bool = True,
    ) -> Dict[str, Any]:
        """Safety-gate and execute one existing SLAI AckermannAction."""

        if not self._started:
            raise RuntimeError("RoboCar.start() must be called before execution")
        duration_value = optional_finite_float(duration, minimum=0.0)
        if duration_value is None:
            raise ValueError("duration must be a finite non-negative number")

        local = self.local_safety.authorize_command(
            throttle,
            steering,
            reading=self.sensor_bus.latest(),
            speed_mps=self.encoder.get_speed(),
        )
        self._publish_local_safety(local)
        if not local.allowed:
            self.motion.stop()
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
        safety_result: Dict[str, Any] = {"approved": True, "decision": "local_only"}
        if require_slai_safety:
            safety_result = dict(
                self.agent("safety").validate_action(
                    action_params,
                    {
                        "system": "RoboCar",
                        "sensor": (
                            self.sensor_bus.latest().to_dict()
                            if self.sensor_bus.latest() is not None
                            else None
                        ),
                        "local_safety": local.to_dict(),
                    },
                )
            )
            if safety_result.get("approved") is not True:
                self.motion.stop()
                return {
                    "status": "blocked",
                    "reason": "slai_safety",
                    "local_safety": local.to_dict(),
                    "safety": safety_result,
                }

        execution = self._agents.get("execution")
        if execution is None:
            raise RuntimeError("ExecutionAgent is not initialized")
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
            "metadata": {"source": str(source), "safety_validation": safety_result.get("validation_id")},
        }
        result = dict(execution.perform_task(task))
        return {
            "status": result.get("status", "unknown"),
            "execution": result,
            "local_safety": local.to_dict(),
            "safety": safety_result,
        }

    # ------------------------------------------------------------------
    # Planning/local control utilities
    # ------------------------------------------------------------------
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
            radius = get_config_section("robocar", self.config).get("inflation_radius_m", 0.0)
        radius_value = require_finite_float(radius, "robocar.inflation_radius_m", minimum=0.0)
        path = astar_path(occupancy, start, goal, inflation_radius_m=radius_value)
        self.shared_memory.set(K_MAP_LATEST, occupancy.to_dict())
        self.shared_memory.set(K_PLAN_CURRENT, [asdict(point) for point in path])
        return path

    def plan_with_slai(self, planning_task: Any) -> Any:
        """Delegate a caller-supplied current SLAI planning Task without remapping it."""

        planner = self.agent("planning")
        return planner.generate_plan(planning_task)

    def health(self) -> Dict[str, Any]:
        execution = self._agents.get("execution")
        execution_health = None
        if execution is not None and callable(getattr(execution, "get_health_report", None)):
            try:
                execution_health = execution.get_health_report()
            except Exception as exc:
                execution_health = {"status": "degraded", "error": str(exc)}
        return {
            "started": self._started,
            "simulation_allowed": self.allow_simulation,
            "sensor_bus": self.sensor_bus.health(),
            "motion": self.motion.get_status(),
            "encoder": self.encoder.health(),
            "execution_agent": execution_health,
            "agents_initialized": sorted(self._agents),
            "last_error": self._last_error,
        }

    def _public_config_snapshot(self) -> Dict[str, Any]:
        # Keep only RoboCar-owned sections and omit the loader's internal path key.
        return {
            key: value
            for key, value in self.config.items()
            if key in {"main", "encoder", "motion", "speed", "hardware", "power", "robocar"}
        }


# ----------------------------------------------------------------------
# Local A* fallback
# ----------------------------------------------------------------------
def astar_path(
    occupancy: OccupancyGrid,
    start_world: Tuple[float, float],
    goal_world: Tuple[float, float],
    *,
    inflation_radius_m: float = 0.0,
    occupied_threshold: int = 50,
    treat_unknown_as_obstacle: bool = True,
) -> list[Waypoint]:
    start = occupancy.world_to_cell(*start_world)
    goal = occupancy.world_to_cell(*goal_world)
    if not occupancy.in_bounds(start) or not occupancy.in_bounds(goal):
        raise PlanningError("A*", start_world, goal_world, "start_or_goal_out_of_bounds")

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
            return bool(inflated[y, x])

    if blocked(start) or blocked(goal):
        raise PlanningError("A*", start_world, goal_world, "start_or_goal_in_obstacle")

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
            # Do not allow a diagonal to cut through two blocked orthogonal cells.
            if dx and dy:
                if blocked((current[0] + dx, current[1])) or blocked((current[0], current[1] + dy)):
                    continue
            candidate = current_g + step_cost
            if candidate >= g_score.get(nxt, float("inf")):
                continue
            g_score[nxt] = candidate
            came_from[nxt] = current
            heuristic = math.hypot(goal[0] - nxt[0], goal[1] - nxt[1])
            heapq.heappush(frontier, (candidate + heuristic, nxt))

    if goal not in came_from:
        raise PlanningError("A*", start_world, goal_world, "no_collision_free_path")

    cells: list[Tuple[int, int]] = []
    cursor: Optional[Tuple[int, int]] = goal
    while cursor is not None:
        cells.append(cursor)
        cursor = came_from[cursor]
    cells.reverse()
    return [occupancy.cell_to_world(cell) for cell in cells]


__all__ = [
    "MEM_FILE",
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
    "Pose2D",
    "Waypoint",
    "OccupancyGrid",
    "SafetyDecision",
    "SafetyManager",
    "PurePursuit",
    "RoboCarRobotAdapter",
    "RoboCar",
    "astar_path",
]
# -------------------------------------------------------------
# robocar implements:
# - geometry/trajectory types
# - occupancy grid schema under dataclass class OccupancyGrid:
# - A* planner (fallback when external planner not available)
# - Safety manager: speed caps, e-stop, zones under SafetyManager()
# - Pure Pursuit follower (local controller) under PurePursuit()
# - RoboCar main orchestrator with the following architecture: Perception → Planning → Execution, with Reasoning + Knowledge enriching context.
#   + Learning and Adaptive with constrain to bounded parameter adaptation with SafetyAgent guardrails.
# - RoboCar main orchestrator uses Observability Agent for high value in robotics bring-up: trace IDs, loop latency histograms, dropped frame counters, watchdog events.
# - RoboCar main orchestrator uses Handler Agent as recovery orchestrator (retry sensor read, switch degraded mode, safe stop).
# - RoboCar main orchestrator uses Evaluation Agent for online KPI scoring: near-miss count, stop distance margin, route tracking error, sensor health score, intervention rate.
# -------------------------------------------------------------
