"""Deterministic Ackermann trajectory-control primitives for RoboCar.

This module owns *control laws only*:

    path + pose -> lateral controller -> normalized steering
    desired speed + measured speed -> PIDSpeedController -> normalized throttle

It does not:
* select missions or routes;
* authorize movement;
* write PWM;
* call SLAI agents;
* infer obstacle safety;
* transform GNSS coordinates.

Those responsibilities belong to planning/localization, Safety/Execution, and
the hardware boundary respectively.
"""

from __future__ import annotations

import math
import threading
import time

from dataclasses import dataclass
from typing import Any, Optional, Protocol, Sequence, Tuple

from ..motion_controller import PIDSpeedController
from ..utils.rc_helpers import *
from .world_model import *


class SpeedController(Protocol):
    def update(self, v_des: float, v_meas: float) -> float:
        ...

    def reset(self) -> None:
        ...


@dataclass(frozen=True, slots=True)
class LateralControlResult:
    steering: float
    steering_angle_rad: float
    target_index: int
    target_x_m: float
    target_y_m: float
    lookahead_distance_m: float
    heading_error_rad: float
    cross_track_error_m: float
    curvature_per_m: float
    goal_reached: bool


@dataclass(frozen=True, slots=True)
class LongitudinalControlResult:
    throttle: float
    desired_speed_mps: float
    measured_speed_mps: float
    speed_error_mps: float


@dataclass(frozen=True, slots=True)
class TrajectoryCommand:
    throttle: float
    steering: float
    desired_speed_mps: float
    measured_speed_mps: float
    target_index: int
    cross_track_error_m: float
    heading_error_rad: float
    goal_reached: bool
    timestamp_monotonic: float


def wrap_angle_rad(angle: float) -> float:
    value = require_finite_float(angle, "angle")
    return math.atan2(math.sin(value), math.cos(value))


def point_segment_distance(
    px: float,
    py: float,
    ax: float,
    ay: float,
    bx: float,
    by: float,
) -> float:
    """Return Euclidean distance from point P to segment AB."""

    px, py, ax, ay, bx, by = map(
        float, (px, py, ax, ay, bx, by)
    )
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby
    if denom <= 1e-18:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * abx + (py - ay) * aby) / denom
    t = max(0.0, min(1.0, t))
    qx, qy = ax + t * abx, ay + t * aby
    return math.hypot(px - qx, py - qy)


def route_cross_track_error(
    pose: PoseState,
    path: Sequence[WaypointState],
) -> float:
    """Distance from pose to the nearest path segment."""

    if not path:
        return math.inf
    if len(path) == 1:
        return math.hypot(pose.x_m - path[0].x_m, pose.y_m - path[0].y_m)
    return min(
        point_segment_distance(
            pose.x_m,
            pose.y_m,
            path[i].x_m,
            path[i].y_m,
            path[i + 1].x_m,
            path[i + 1].y_m,
        )
        for i in range(len(path) - 1)
    )


class PurePursuitController:
    """Pure-pursuit lateral controller for an Ackermann vehicle."""

    def __init__(
        self,
        *,
        lookahead_m: float,
        wheelbase_m: float,
        max_steer_rad: float,
        goal_tolerance_m: float = 0.10,
    ) -> None:
        self.lookahead_m = require_finite_float(
            lookahead_m, "pure_pursuit.lookahead_m", minimum=1e-6
        )
        self.wheelbase_m = require_finite_float(
            wheelbase_m, "pure_pursuit.wheelbase_m", minimum=1e-6
        )
        self.max_steer_rad = require_finite_float(
            max_steer_rad, "pure_pursuit.max_steer_rad", minimum=1e-6
        )
        self.goal_tolerance_m = require_finite_float(
            goal_tolerance_m, "pure_pursuit.goal_tolerance_m", minimum=0.0
        )

    def set_lookahead(self, value: float) -> None:
        """Apply an already-authorized adaptation to the look-ahead."""

        self.lookahead_m = require_finite_float(
            value, "pure_pursuit.lookahead_m", minimum=1e-6
        )

    def compute(
        self,
        pose: PoseState,
        path: Sequence[WaypointState],
        *,
        start_index: int = 0,
    ) -> LateralControlResult:
        if not path:
            raise ValueError("PurePursuitController requires a non-empty path")
        if start_index < 0 or start_index >= len(path):
            raise IndexError("start_index must refer to an existing waypoint")

        goal = path[-1]
        goal_distance = math.hypot(goal.x_m - pose.x_m, goal.y_m - pose.y_m)
        goal_reached = goal_distance <= self.goal_tolerance_m

        target_index = self._target_index(pose, path, start_index=start_index)
        target = path[target_index]
        dx = target.x_m - pose.x_m
        dy = target.y_m - pose.y_m
        distance = max(math.hypot(dx, dy), 1e-9)
        target_heading = math.atan2(dy, dx)
        heading_error = wrap_angle_rad(target_heading - pose.yaw_rad)

        curvature = 2.0 * math.sin(heading_error) / distance
        steering_angle = math.atan(self.wheelbase_m * curvature)
        steering_angle = clamp(
            steering_angle, -self.max_steer_rad, self.max_steer_rad
        )
        normalized_steering = clamp(
            steering_angle / self.max_steer_rad, -1.0, 1.0
        )

        return LateralControlResult(
            steering=0.0 if goal_reached else normalized_steering,
            steering_angle_rad=0.0 if goal_reached else steering_angle,
            target_index=target_index,
            target_x_m=target.x_m,
            target_y_m=target.y_m,
            lookahead_distance_m=distance,
            heading_error_rad=heading_error,
            cross_track_error_m=route_cross_track_error(pose, path),
            curvature_per_m=curvature,
            goal_reached=goal_reached,
        )

    def _target_index(
        self,
        pose: PoseState,
        path: Sequence[WaypointState],
        *,
        start_index: int,
    ) -> int:
        # Begin from the nearest waypoint at/after the active index.  This
        # prevents selection behind the current route progress.
        nearest_index = min(
            range(start_index, len(path)),
            key=lambda i: math.hypot(
                path[i].x_m - pose.x_m,
                path[i].y_m - pose.y_m,
            ),
        )
        for index in range(nearest_index, len(path)):
            point = path[index]
            if math.hypot(point.x_m - pose.x_m, point.y_m - pose.y_m) >= self.lookahead_m:
                return index
        return len(path) - 1


class LongitudinalPIDController:
    """Thin deterministic adapter around RoboCar's PIDSpeedController."""

    def __init__(
        self,
        pid: Optional[SpeedController] = None,
        *,
        config: Optional[dict[str, Any]] = None,
    ) -> None:
        self.pid: SpeedController = (
            pid if pid is not None else PIDSpeedController(config=config)
        )

    def reset(self) -> None:
        self.pid.reset()

    def compute(
        self,
        desired_speed_mps: float,
        measured_speed_mps: float,
    ) -> LongitudinalControlResult:
        desired = require_finite_float(
            desired_speed_mps, "desired_speed_mps", minimum=0.0
        )
        measured = require_finite_float(
            measured_speed_mps, "measured_speed_mps"
        )
        throttle = float(self.pid.update(desired, measured))
        throttle = clamp(throttle, -1.0, 1.0)
        return LongitudinalControlResult(
            throttle=throttle,
            desired_speed_mps=desired,
            measured_speed_mps=measured,
            speed_error_mps=desired - measured,
        )


class TrajectoryController:
    """Combine lateral Pure Pursuit and longitudinal PID into one command."""

    def __init__(
        self,
        lateral: PurePursuitController,
        longitudinal: LongitudinalPIDController,
    ) -> None:
        self.lateral = lateral
        self.longitudinal = longitudinal
        self._lock = threading.RLock()
        self._last_command: Optional[TrajectoryCommand] = None

    @property
    def last_command(self) -> Optional[TrajectoryCommand]:
        with self._lock:
            return self._last_command

    def reset(self) -> None:
        with self._lock:
            self.longitudinal.reset()
            self._last_command = None

    def compute(
        self,
        *,
        pose: PoseState,
        path: Sequence[WaypointState],
        desired_speed_mps: float,
        measured_speed_mps: Optional[float] = None,
        active_index: int = 0,
    ) -> TrajectoryCommand:
        measured = pose.speed_mps if measured_speed_mps is None else measured_speed_mps

        with self._lock:
            lateral = self.lateral.compute(
                pose, path, start_index=active_index
            )
            if lateral.goal_reached:
                # Reset the longitudinal integrator so a completed route cannot
                # carry accumulated positive integral state into a later mission.
                self.longitudinal.reset()
                longitudinal = LongitudinalControlResult(
                    throttle=0.0,
                    desired_speed_mps=0.0,
                    measured_speed_mps=float(measured),
                    speed_error_mps=-float(measured),
                )
            else:
                longitudinal = self.longitudinal.compute(
                    desired_speed_mps, measured
                )

            command = TrajectoryCommand(
                throttle=longitudinal.throttle,
                steering=lateral.steering,
                desired_speed_mps=longitudinal.desired_speed_mps,
                measured_speed_mps=longitudinal.measured_speed_mps,
                target_index=lateral.target_index,
                cross_track_error_m=lateral.cross_track_error_m,
                heading_error_rad=lateral.heading_error_rad,
                goal_reached=lateral.goal_reached,
                timestamp_monotonic=time.monotonic(),
            )
            self._last_command = command
            return command


__all__ = [
    "LateralControlResult",
    "LongitudinalControlResult",
    "TrajectoryCommand",
    "wrap_angle_rad",
    "point_segment_distance",
    "route_cross_track_error",
    "PurePursuitController",
    "LongitudinalPIDController",
    "TrajectoryController",
]
