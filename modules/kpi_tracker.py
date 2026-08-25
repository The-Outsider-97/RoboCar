"""Online vehicle KPI accounting for RoboCar.

This module computes vehicle-specific semantics locally and deterministically.
It does not ask SLAI EvaluationAgent to invent meanings for RC-car metrics.
The resulting snapshot can be forwarded to EvaluationAgent/ObservabilityAgent.
"""

from __future__ import annotations

import math
import threading
import time

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from .trajectory_control import *
from .world_model import *


def _finite_optional(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


@dataclass(frozen=True, slots=True)
class KPISnapshot:
    timestamp_wall: float
    elapsed_s: float
    samples: int

    near_miss_count: int
    minimum_obstacle_margin_m: Optional[float]
    minimum_stop_distance_margin_m: Optional[float]

    route_cross_track_error_m: Optional[float]
    route_cross_track_error_mean_m: Optional[float]
    route_cross_track_error_max_m: Optional[float]
    heading_error_rad: Optional[float]
    heading_error_abs_mean_rad: Optional[float]
    heading_error_abs_max_rad: Optional[float]

    sensor_health_score: Optional[float]
    sensor_availability_ratio: Optional[float]
    gnss_availability_ratio: Optional[float]

    intervention_count: int
    autonomous_cycles: int
    manual_cycles: int
    autonomy_ratio: Optional[float]
    manual_ratio: Optional[float]

    control_loop_deadline_misses: int
    dropped_pico_frames: int
    recovery_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VehicleKPITracker:
    """Thread-safe online accumulator for RoboCar operational KPIs.

    Parameters are deliberately explicit.  The tracker will not invent a
    near-miss threshold or braking model.

    ``near_miss_distance_m``
        Distance at/below which a valid forward obstacle sample counts as a
        near-miss *transition*.  Continuous samples inside the same near-miss
        episode count once until the distance rises above the threshold again.
        Set to ``None`` to disable near-miss counting.

    ``reaction_time_s`` and ``max_deceleration_mps2``
        If both are supplied, stop-distance margin is estimated as:

            margin = obstacle_distance
                     - (speed * reaction_time
                        + speed^2 / (2 * max_deceleration))

        The deceleration value must come from physical calibration, not theory.
    """

    def __init__(
        self,
        *,
        near_miss_distance_m: Optional[float] = None,
        reaction_time_s: Optional[float] = None,
        max_deceleration_mps2: Optional[float] = None,
    ) -> None:
        for name, value in (
            ("near_miss_distance_m", near_miss_distance_m),
            ("reaction_time_s", reaction_time_s),
            ("max_deceleration_mps2", max_deceleration_mps2),
        ):
            if value is not None and (not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if max_deceleration_mps2 == 0:
            raise ValueError("max_deceleration_mps2 must be > 0 when configured")

        self.near_miss_distance_m = (
            None if near_miss_distance_m is None else float(near_miss_distance_m)
        )
        self.reaction_time_s = (
            None if reaction_time_s is None else float(reaction_time_s)
        )
        self.max_deceleration_mps2 = (
            None if max_deceleration_mps2 is None else float(max_deceleration_mps2)
        )

        self._lock = threading.RLock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._started_monotonic = time.monotonic()
            self._samples = 0
            self._near_miss_count = 0
            self._near_miss_active = False
            self._minimum_obstacle_margin_m: Optional[float] = None
            self._minimum_stop_distance_margin_m: Optional[float] = None

            self._cross_track_latest: Optional[float] = None
            self._cross_track_sum = 0.0
            self._cross_track_count = 0
            self._cross_track_max: Optional[float] = None

            self._heading_latest: Optional[float] = None
            self._heading_abs_sum = 0.0
            self._heading_count = 0
            self._heading_abs_max: Optional[float] = None

            self._sensor_health_latest: Optional[float] = None
            self._sensor_available_sum = 0
            self._sensor_total_sum = 0
            self._gnss_available = 0
            self._gnss_samples = 0

            self._intervention_count = 0
            self._autonomous_cycles = 0
            self._manual_cycles = 0
            self._control_loop_deadline_misses = 0
            self._dropped_pico_frames = 0
            self._recovery_count = 0

    def observe(
        self,
        *,
        pose: Optional[PoseState] = None,
        path: Optional[Sequence[WaypointState]] = None,
        target_heading_rad: Optional[float] = None,
        front_obstacle_distance_m: Optional[float] = None,
        configured_stop_distance_m: Optional[float] = None,
        sensor_available: Optional[int] = None,
        sensor_total: Optional[int] = None,
        sensor_health_score: Optional[float] = None,
        gnss_available: Optional[bool] = None,
        mode: Optional[str] = None,
        control_loop_duration_s: Optional[float] = None,
        control_loop_deadline_s: Optional[float] = None,
        dropped_pico_frames_total: Optional[int] = None,
    ) -> KPISnapshot:
        """Consume one coherent vehicle observation and return a snapshot."""

        with self._lock:
            self._samples += 1

            obstacle = _finite_optional(front_obstacle_distance_m)
            if obstacle is not None and obstacle >= 0.0:
                if configured_stop_distance_m is not None:
                    stop_distance = float(configured_stop_distance_m)
                    margin = obstacle - stop_distance
                    self._minimum_obstacle_margin_m = self._min_optional(
                        self._minimum_obstacle_margin_m, margin
                    )

                if self.near_miss_distance_m is not None:
                    active = obstacle <= self.near_miss_distance_m
                    if active and not self._near_miss_active:
                        self._near_miss_count += 1
                    self._near_miss_active = active

                if (
                    pose is not None
                    and self.reaction_time_s is not None
                    and self.max_deceleration_mps2 is not None
                ):
                    speed = max(0.0, float(pose.speed_mps))
                    required_stop = (
                        speed * self.reaction_time_s
                        + (speed * speed) / (2.0 * self.max_deceleration_mps2)
                    )
                    margin = obstacle - required_stop
                    self._minimum_stop_distance_margin_m = self._min_optional(
                        self._minimum_stop_distance_margin_m, margin
                    )
            elif obstacle is None:
                self._near_miss_active = False

            if pose is not None and path:
                cte = route_cross_track_error(pose, path)
                if math.isfinite(cte):
                    self._cross_track_latest = cte
                    self._cross_track_sum += cte
                    self._cross_track_count += 1
                    self._cross_track_max = self._max_optional(
                        self._cross_track_max, cte
                    )

            if pose is not None and target_heading_rad is not None:
                error = wrap_angle_rad(float(target_heading_rad) - pose.yaw_rad)
                absolute = abs(error)
                self._heading_latest = error
                self._heading_abs_sum += absolute
                self._heading_count += 1
                self._heading_abs_max = self._max_optional(
                    self._heading_abs_max, absolute
                )

            if sensor_health_score is not None:
                score = float(sensor_health_score)
                if math.isfinite(score) and 0.0 <= score <= 1.0:
                    self._sensor_health_latest = score

            if sensor_available is not None and sensor_total is not None:
                available, total = int(sensor_available), int(sensor_total)
                if total >= 0 and 0 <= available <= total:
                    self._sensor_available_sum += available
                    self._sensor_total_sum += total

            if gnss_available is not None:
                self._gnss_samples += 1
                self._gnss_available += int(bool(gnss_available))

            normalized_mode = str(mode or "").strip().lower()
            if normalized_mode == "autonomous":
                self._autonomous_cycles += 1
            elif normalized_mode == "manual":
                self._manual_cycles += 1

            if (
                control_loop_duration_s is not None
                and control_loop_deadline_s is not None
                and float(control_loop_duration_s) > float(control_loop_deadline_s)
            ):
                self._control_loop_deadline_misses += 1

            if dropped_pico_frames_total is not None:
                value = max(0, int(dropped_pico_frames_total))
                self._dropped_pico_frames = max(self._dropped_pico_frames, value)

            return self.snapshot()

    def record_intervention(self, *, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            self._intervention_count += int(count)

    def record_recovery(self, *, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            self._recovery_count += int(count)

    def record_deadline_miss(self, *, count: int = 1) -> None:
        if count < 1:
            raise ValueError("count must be >= 1")
        with self._lock:
            self._control_loop_deadline_misses += int(count)

    def snapshot(self) -> KPISnapshot:
        with self._lock:
            total_mode = self._autonomous_cycles + self._manual_cycles
            return KPISnapshot(
                timestamp_wall=time.time(),
                elapsed_s=max(0.0, time.monotonic() - self._started_monotonic),
                samples=self._samples,
                near_miss_count=self._near_miss_count,
                minimum_obstacle_margin_m=self._minimum_obstacle_margin_m,
                minimum_stop_distance_margin_m=self._minimum_stop_distance_margin_m,
                route_cross_track_error_m=self._cross_track_latest,
                route_cross_track_error_mean_m=self._mean(
                    self._cross_track_sum, self._cross_track_count
                ),
                route_cross_track_error_max_m=self._cross_track_max,
                heading_error_rad=self._heading_latest,
                heading_error_abs_mean_rad=self._mean(
                    self._heading_abs_sum, self._heading_count
                ),
                heading_error_abs_max_rad=self._heading_abs_max,
                sensor_health_score=self._sensor_health_latest,
                sensor_availability_ratio=(
                    self._sensor_available_sum / self._sensor_total_sum
                    if self._sensor_total_sum > 0
                    else None
                ),
                gnss_availability_ratio=(
                    self._gnss_available / self._gnss_samples
                    if self._gnss_samples > 0
                    else None
                ),
                intervention_count=self._intervention_count,
                autonomous_cycles=self._autonomous_cycles,
                manual_cycles=self._manual_cycles,
                autonomy_ratio=(
                    self._autonomous_cycles / total_mode if total_mode else None
                ),
                manual_ratio=(
                    self._manual_cycles / total_mode if total_mode else None
                ),
                control_loop_deadline_misses=self._control_loop_deadline_misses,
                dropped_pico_frames=self._dropped_pico_frames,
                recovery_count=self._recovery_count,
            )

    @staticmethod
    def _min_optional(current: Optional[float], value: float) -> float:
        return value if current is None else min(current, value)

    @staticmethod
    def _max_optional(current: Optional[float], value: float) -> float:
        return value if current is None else max(current, value)

    @staticmethod
    def _mean(total: float, count: int) -> Optional[float]:
        return total / count if count else None


__all__ = ["KPISnapshot", "VehicleKPITracker"]
