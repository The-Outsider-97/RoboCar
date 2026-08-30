"""SLAI multi-agent orchestration adapter for RoboCar.

This is the *only* module under ``RoboCar/modules`` that imports SLAI agents.
All deterministic vehicle mechanics remain outside the agent layer.

The module adapts RoboCar into SLAI's current ``AutonomousControlLoop`` contract:

    reason -> plan -> authorize -> execute -> evaluate

Knowledge enriches reasoning context.  Observability is emitted at bounded
mission/cycle boundaries.  HandlerAgent is invoked only after a local hardware
safe-stop has already been attempted.  Learning/Adaptive are intentionally not
placed in the motion-critical loop; bounded parameter proposals pass through
``AdaptationGuard`` separately.
"""

from __future__ import annotations

import math
import time
import uuid

from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Sequence

from src.agents.autonomous_control_loop import (
    AutonomousControlLoop,
    AutonomousLoopConfig,
    ControlLoopContractError,
)

from .adaptation_guard import *
from .kpi_tracker import *
from .world_model import *


def _safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if is_dataclass(value) and not isinstance(value, type):
        return _safe(asdict(value))
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _safe(to_dict())
        except Exception:
            pass
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _status_success(result: Mapping[str, Any]) -> bool:
    status = str(result.get("status", "")).strip().lower()
    return (
        result.get("success") is True
        or result.get("passed") is True
        or result.get("completed") is True
        or status in {
            "ok", "success", "succeeded", "pass", "passed",
            "complete", "completed", "allow", "approved", "normal",
        }
    )


class RoboCarAutonomyStages:
    """Current SLAI stage adapter around one physical RoboCar instance.

    ``car`` is structural: the adapter only requires the fields/callables it
    actually uses, which keeps this module independent of ``robocar.py`` and
    avoids circular imports.
    """

    def __init__(
        self,
        *,
        car: Any,
        world_model: WorldModel,
        kpi_tracker: VehicleKPITracker,
        adaptation_guard: Optional[AdaptationGuard] = None,
        knowledge_k: int = 3,
    ) -> None:
        self.car = car
        self.world_model = world_model
        self.kpi_tracker = kpi_tracker
        self.adaptation_guard = adaptation_guard
        self.knowledge_k = max(0, int(knowledge_k))
        self._agents: Dict[str, Any] = {}
        self._failed_agent_name: Optional[str] = None
        self._run_started_mono: Optional[float] = None

    # ------------------------------------------------------------------
    # Agent access
    # ------------------------------------------------------------------
    def _agent(self, name: str) -> Any:
        if name in self._agents:
            return self._agents[name]

        # Reuse RoboCar's physically attached ExecutionAgent.  Creating another
        # factory execution agent would lose ``robot=self.robot_adapter``.
        if name == "execution":
            existing = getattr(self.car, "_agents", {}).get("execution")
            if existing is None:
                existing = getattr(self.car, "execution_agent", None)
            if existing is None:
                raise RuntimeError(
                    "RoboCar physical ExecutionAgent is not initialized"
                )
            self._agents[name] = existing
            return existing

        accessor = getattr(self.car, "agent", None)
        if callable(accessor):
            agent = accessor(name)
        else:
            factory = getattr(self.car, "agent_factory", None)
            shared = getattr(self.car, "shared_memory", None)
            if factory is None or not callable(getattr(factory, "create", None)):
                raise RuntimeError("RoboCar has no AgentFactory-compatible accessor")
            agent = factory.create(name, shared_memory=shared)

        self._agents[name] = agent
        return agent

    def _call(
        self,
        agent_name: str,
        method_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        try:
            agent = self._agent(agent_name)
            method = getattr(agent, method_name, None)
            if not callable(method):
                raise ControlLoopContractError(
                    f"{type(agent).__name__} has no callable {method_name}()"
                )
            return method(*args, **kwargs)
        except Exception:
            self._failed_agent_name = agent_name
            raise

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------
    def observation(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        snapshot = self.world_model.snapshot()
        return snapshot.to_dict()

    # ------------------------------------------------------------------
    # Fixed SLAI stages
    # ------------------------------------------------------------------
    def reason(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        objective = str(
            goal.get("objective") or goal.get("goal") or goal.get("name") or ""
        ).strip()
        if not objective:
            raise ControlLoopContractError(
                "RoboCar autonomous goal requires objective, goal, or name"
            )

        knowledge = self._retrieve_knowledge(objective)
        reasoning_result = self._call(
            "reasoning",
            "perform_task",
            {
                "task_type": "reason",
                "problem": objective,
                "context": {
                    **self._mapping(payload.get("context")),
                    "world": self._mapping(payload.get("observation")),
                    "knowledge": knowledge,
                    "previous_feedback": self._mapping(
                        payload.get("previous_feedback")
                    ),
                },
            },
        )

        # The current AutonomousControlLoop factory adapter requires an explicit
        # calibrated numeric confidence at authorization.  ReasoningAgent output
        # shapes vary by reasoning mode, so the goal/context must provide one if
        # the agent result does not.
        confidence = self._extract_confidence(reasoning_result)
        if confidence is None:
            confidence = self._extract_confidence(goal)
        if confidence is None:
            return {
                "status": "failed",
                "reason": "reasoning_confidence_required",
                "reasoning": _safe(reasoning_result),
                "knowledge": knowledge,
            }

        return {
            "status": "success",
            "confidence": confidence,
            "reasoning": _safe(reasoning_result),
            "knowledge": knowledge,
        }

    def plan(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        supplied = goal.get("planning_task")
        if supplied is None:
            supplied = self._build_planning_task(goal)

        plan = self._call("planning", "generate_plan", supplied)
        if not isinstance(plan, Sequence) or isinstance(plan, (str, bytes)) or not plan:
            return {
                "status": "failed",
                "plan": [],
                "reason": "planning_agent_returned_no_plan",
            }

        now = time.monotonic()
        snapshot = self.world_model.snapshot()
        self.world_model.update(
            autonomy=AutonomyState(
                mode=snapshot.autonomy.mode,
                run_id=str(payload.get("run_id") or "") or None,
                goal_id=str(payload.get("goal_id") or "") or None,
                cycle=int(payload.get("cycle") or 0),
                planner_status="success",
                last_plan_monotonic=now,
                last_control_cycle_monotonic=(
                    snapshot.autonomy.last_control_cycle_monotonic
                ),
                last_recovery_monotonic=snapshot.autonomy.last_recovery_monotonic,
                metadata=snapshot.autonomy.metadata,
            ),
            event_type="planning.completed",
            event_payload={"items": len(plan)},
        )
        return {
            "status": "success",
            "plan": [_safe(item) for item in plan],
            "planning_task": _safe(supplied),
        }

    def authorize(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        reason_output = self._mapping(payload.get("reason"))
        confidence = self._extract_confidence(reason_output)
        if confidence is None:
            raise ControlLoopContractError(
                "reason.confidence must be an explicit calibrated number"
            )

        loop = getattr(self.car, "autonomy_loop", None)
        minimum = (
            getattr(getattr(loop, "config", None), "min_action_confidence", 0.7)
            if loop is not None
            else 0.7
        )
        if confidence < float(minimum):
            return {
                "status": "review_required",
                "approved": False,
                "decision": "review_required",
                "reason": "action_confidence_below_threshold",
                "confidence": confidence,
                "required_confidence": float(minimum),
            }

        local = self._local_authorization(goal)
        if local is not None and local.get("allowed") is not True:
            return {
                "status": "blocked",
                "approved": False,
                "decision": "blocked",
                "reason": "local_vehicle_safety_blocked",
                "local_safety": local,
            }

        action = goal.get("safety_action")
        if not isinstance(action, Mapping):
            execution_task = goal.get("execution_task")
            action = {
                "name": "robocar_autonomous_execution",
                "goal_id": payload.get("goal_id"),
                "execution_task": _safe(execution_task),
                "plan": _safe(payload.get("plan", {})),
            }

        safety_result = self._call(
            "safety",
            "validate_action",
            dict(action),
            {
                "type": "robocar_autonomous_control_loop",
                "run_id": payload.get("run_id"),
                "cycle": payload.get("cycle"),
                "world": self._mapping(payload.get("observation")),
                "local_safety": local,
                **self._mapping(payload.get("context")),
            },
        )
        if not isinstance(safety_result, Mapping):
            raise ControlLoopContractError(
                "SafetyAgent.validate_action() must return a mapping"
            )

        approved = safety_result.get("approved") is True
        decision = (
            "approved"
            if approved
            else str(
                safety_result.get(
                    "decision",
                    safety_result.get("overall_recommendation", "blocked"),
                )
            ).strip().lower()
        )
        if decision not in {"approved", "allow", "pass", "passed"} and not approved:
            decision = (
                "review_required"
                if decision in {"review", "review_required", "human_review"}
                else "blocked"
            )

        snapshot = self.world_model.snapshot()
        self.world_model.update(
            safety=SafetyState(
                estop_latched=snapshot.safety.estop_latched,
                allowed_to_move=approved,
                degraded=snapshot.safety.degraded,
                speed_cap_mps=snapshot.safety.speed_cap_mps,
                reasons=(
                    snapshot.safety.reasons
                    if approved
                    else tuple(snapshot.safety.reasons) + ("slai_safety_denied",)
                ),
                warnings=snapshot.safety.warnings,
            ),
            event_type="safety.authorization",
            event_payload={"approved": approved, "decision": decision},
            event_severity="info" if approved else "warning",
        )

        return {
            "status": decision,
            "approved": approved,
            "decision": decision,
            "confidence": confidence,
            "local_safety": local,
            "safety": _safe(safety_result),
        }

    def execute(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        goal = self._goal(payload)
        execution_task = goal.get("execution_task")
        if not isinstance(execution_task, Mapping):
            return {
                "status": "blocked",
                "success": False,
                "reason": "explicit_execution_task_required",
            }

        task = dict(execution_task)
        task.setdefault(
            "id",
            f"{payload.get('goal_id')}:cycle:{payload.get('cycle')}",
        )
        task.setdefault(
            "name",
            str(goal.get("name") or goal.get("objective") or "robocar_autonomy"),
        )
        task.setdefault("goal_type", "robot_control")
        metadata = task.get("metadata")
        task["metadata"] = {
            **(dict(metadata) if isinstance(metadata, Mapping) else {}),
            "autonomous_run_id": payload.get("run_id"),
            "autonomous_cycle": payload.get("cycle"),
            "source": "robocar.slai_autonomy",
        }

        result = self._call("execution", "perform_task", task)
        if not isinstance(result, Mapping):
            raise ControlLoopContractError(
                "ExecutionAgent.perform_task() must return a mapping"
            )
        return dict(result)

    def evaluate(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        execution = self._mapping(payload.get("execute"))
        kpis = self.kpi_tracker.snapshot().to_dict()
        goal = self._goal(payload)

        evaluation_params = goal.get("evaluation_params")
        if not isinstance(evaluation_params, Mapping):
            incomplete = {
                "status": "incomplete",
                "completed": False,
                "passed": False,
                "reason": "evaluation_params_required",
                "robocar_kpis": kpis,
            }
            self._emit_observability(payload, incomplete)
            return incomplete

        params = dict(evaluation_params)
        params.setdefault("control_loop_execution", _safe(execution))
        params.setdefault(
            "agent_performance_metrics",
            dict(self._mapping(payload.get("stage_metrics"))),
        )
        params.setdefault("robocar_kpis", kpis)
        params.setdefault("world_state", self.world_model.snapshot().to_dict())

        result = self._call(
            "evaluation", "execute_validation_cycle", params
        )
        if not isinstance(result, Mapping):
            raise ControlLoopContractError(
                "EvaluationAgent.execute_validation_cycle() must return a mapping"
            )

        normalized = dict(result)
        execution_ok = _status_success(execution)
        normalized.setdefault(
            "completed",
            execution_ok
            and str(normalized.get("status", "")).lower()
            not in {"failed", "critical", "error"},
        )

        self._emit_observability(payload, normalized)
        return normalized

    # ------------------------------------------------------------------
    # Failure / recovery
    # ------------------------------------------------------------------
    def handle_failure(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        failed_stage = str(payload.get("failed_stage", "unknown")).strip().lower()

        # Required safety ordering: neutral hardware first.
        stop = getattr(getattr(self.car, "motion", None), "stop", None)
        stop_error: Optional[str] = None
        if callable(stop):
            try:
                stop()
            except Exception as exc:
                stop_error = f"{type(exc).__name__}: {exc}"

        self.kpi_tracker.record_recovery()
        snapshot = self.world_model.snapshot()
        self.world_model.update(
            autonomy=AutonomyState(
                mode=OperatingMode.DEGRADED,
                run_id=str(payload.get("run_id") or "") or None,
                goal_id=str(payload.get("goal_id") or "") or None,
                cycle=int(payload.get("cycle") or 0),
                planner_status=snapshot.autonomy.planner_status,
                last_plan_monotonic=snapshot.autonomy.last_plan_monotonic,
                last_control_cycle_monotonic=(
                    snapshot.autonomy.last_control_cycle_monotonic
                ),
                last_recovery_monotonic=time.monotonic(),
                metadata=snapshot.autonomy.metadata,
            ),
            event_type="autonomy.recovery_requested",
            event_payload={
                "failed_stage": failed_stage,
                "safe_stop_error": stop_error,
            },
            event_severity="critical" if stop_error else "warning",
        )

        target_name = self._failed_agent_name
        self._failed_agent_name = None
        target_agent = None
        if target_name:
            try:
                target_agent = self._agent(target_name)
            except Exception:
                target_agent = None

        result = self._call(
            "handler",
            "perform_task",
            {
                "error": payload.get("error"),
                "target_agent": target_agent,
                "task_data": self._goal(payload),
                "context": {
                    "source": "robocar.slai_autonomy",
                    "run_id": payload.get("run_id"),
                    "cycle": payload.get("cycle"),
                    "stage": failed_stage,
                    "agent": target_name,
                    "safe_stop_applied": stop_error is None,
                    "safe_stop_error": stop_error,
                },
            },
        )
        if not isinstance(result, Mapping):
            raise ControlLoopContractError(
                "HandlerAgent.perform_task() must return a mapping"
            )
        return dict(result)

    # ------------------------------------------------------------------
    # Adaptation bridge (outside motion-critical loop)
    # ------------------------------------------------------------------
    def safety_review_adaptation(
        self,
        proposal: AdaptationAuditRecord,
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AdaptationAuditRecord:
        if self.adaptation_guard is None:
            raise RuntimeError("No AdaptationGuard is configured")
        return self.adaptation_guard.safety_approve(
            proposal.proposal.proposal_id,
            self._agent("safety").validate_action,
            context=context,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _retrieve_knowledge(self, objective: str) -> list[dict[str, Any]]:
        if self.knowledge_k <= 0:
            return []
        try:
            results = self._call(
                "knowledge", "retrieve", objective, self.knowledge_k
            )
        except Exception:
            # Knowledge enrichment is non-safety-critical.  Reasoning may still
            # proceed without retrieval; the failure is surfaced in observability
            # through stage context rather than converted into a drive command.
            return []

        normalized: list[dict[str, Any]] = []
        if isinstance(results, Sequence):
            for item in results:
                if (
                    isinstance(item, Sequence)
                    and not isinstance(item, (str, bytes))
                    and len(item) == 2
                ):
                    score, doc = item
                    normalized.append(
                        {
                            "score": float(score),
                            "document": _safe(doc),
                        }
                    )
        return normalized

    def _local_authorization(
        self,
        goal: Mapping[str, Any],
    ) -> Optional[Dict[str, Any]]:
        local = getattr(self.car, "local_safety", None)
        authorize = getattr(local, "authorize_command", None)
        if not callable(authorize):
            return None

        task = goal.get("execution_task")
        if not isinstance(task, Mapping):
            return None
        sequence = task.get("action_sequence")
        if (
            not isinstance(sequence, Sequence)
            or isinstance(sequence, (str, bytes))
            or not sequence
            or not isinstance(sequence[0], Mapping)
        ):
            return None

        action = dict(sequence[0])
        if str(action.get("name", "")).strip().lower() != "ackermann":
            return None

        reading = getattr(getattr(self.car, "sensor_bus", None), "latest", lambda: None)()
        speed = getattr(getattr(self.car, "encoder", None), "get_speed", lambda: None)()
        decision = authorize(
            action.get("throttle", 0.0),
            action.get("steering", 0.0),
            reading=reading,
            speed_mps=speed,
        )
        to_dict = getattr(decision, "to_dict", None)
        return dict(to_dict()) if callable(to_dict) else _safe(decision)

    @staticmethod
    def _build_planning_task(goal: Mapping[str, Any]) -> Any:
        from src.agents.planning.planning_types import Task, TaskType

        objective = str(
            goal.get("objective")
            or goal.get("goal")
            or goal.get("name")
            or "RoboCar autonomous goal"
        )
        raw_steps = goal.get("plan_steps")
        if (
            isinstance(raw_steps, Sequence)
            and not isinstance(raw_steps, (str, bytes))
            and raw_steps
        ):
            steps = []
            for index, raw in enumerate(raw_steps, 1):
                step = dict(raw) if isinstance(raw, Mapping) else {"name": str(raw)}
                steps.append(
                    Task(
                        name=str(step.get("name") or f"{objective}:step:{index}"),
                        task_type=TaskType.PRIMITIVE,
                        description=str(
                            step.get("description") or step.get("name") or objective
                        ),
                        context=step,
                        duration=float(step.get("duration", 1.0)),
                        preconditions=[lambda _state: True],
                    )
                )
            return Task(
                name=str(goal.get("name") or objective),
                task_type=TaskType.ABSTRACT,
                goal_state=dict(RoboCarAutonomyStages._mapping(goal.get("goal_state"))),
                context=dict(RoboCarAutonomyStages._mapping(goal.get("context"))),
                methods=[steps],
            )

        return Task(
            name=str(goal.get("name") or objective),
            task_type=TaskType.PRIMITIVE,
            goal_state=dict(RoboCarAutonomyStages._mapping(goal.get("goal_state"))),
            context=dict(RoboCarAutonomyStages._mapping(goal.get("context"))),
            description=objective,
            duration=float(goal.get("duration", 1.0)),
            preconditions=[lambda _state: True],
        )

    def _emit_observability(
        self,
        payload: Mapping[str, Any],
        evaluation: Mapping[str, Any],
    ) -> None:
        try:
            observability = self._agent("observability")
            perform = getattr(observability, "perform_task", None)
            if not callable(perform):
                return
            stage_metrics = self._mapping(payload.get("stage_metrics"))
            latencies = []
            for stage, metrics in stage_metrics.items():
                if isinstance(metrics, Mapping):
                    latency = metrics.get("latency_ms")
                    if isinstance(latency, (int, float)) and math.isfinite(float(latency)):
                        latencies.append(
                            {
                                "subject": f"robocar.autonomy.{stage}",
                                "duration_ms": float(latency),
                                "status": "ok",
                            }
                        )

            sensor_health = {}
            health = getattr(getattr(self.car, "sensor_bus", None), "health", None)
            if callable(health):
                sensor_health = dict(health())

            perform(
                {
                    "task_name": "robocar_autonomous_mission",
                    "agent_name": "RoboCar",
                    "operation_name": "autonomy_cycle",
                    "source": "robocar.slai_autonomy",
                    "latencies": latencies,
                    "throughput": [
                        {
                            "subject": "robocar.sensor_bus",
                            "count": int(sensor_health.get("frames_received", 0)),
                            "failure_count": int(sensor_health.get("parse_errors", 0))
                            + int(sensor_health.get("transport_errors", 0)),
                        }
                    ],
                    "events": [
                        {
                            "event": "evaluation",
                            "run_id": payload.get("run_id"),
                            "cycle": payload.get("cycle"),
                            "result": _safe(evaluation),
                        }
                    ],
                }
            )
        except Exception:
            # Observability is intentionally non-authoritative.  Its failure must
            # never convert an otherwise safe stop or denied command into motion.
            return

    @staticmethod
    def _goal(payload: Mapping[str, Any]) -> Dict[str, Any]:
        goal = payload.get("goal", {})
        if not isinstance(goal, Mapping):
            raise ControlLoopContractError("goal must remain a mapping")
        return dict(goal)

    @staticmethod
    def _mapping(value: Any) -> Mapping[str, Any]:
        return value if isinstance(value, Mapping) else {}

    @staticmethod
    def _extract_confidence(value: Any) -> Optional[float]:
        if isinstance(value, Mapping):
            for key in ("confidence", "action_confidence", "overall_confidence"):
                raw = value.get(key)
                if (
                    isinstance(raw, (int, float))
                    and not isinstance(raw, bool)
                    and math.isfinite(float(raw))
                    and 0.0 <= float(raw) <= 1.0
                ):
                    return float(raw)
            for nested in ("reasoning", "result", "metadata"):
                found = RoboCarAutonomyStages._extract_confidence(value.get(nested))
                if found is not None:
                    return found
        return None

    def stage_mapping(self) -> Dict[str, Callable[[Mapping[str, Any]], Mapping[str, Any]]]:
        return {
            "reason": self.reason,
            "plan": self.plan,
            "authorize": self.authorize,
            "execute": self.execute,
            "evaluate": self.evaluate,
        }


def build_robocar_autonomy_loop(
    *,
    car: Any,
    world_model: WorldModel,
    kpi_tracker: VehicleKPITracker,
    adaptation_guard: Optional[AdaptationGuard] = None,
    config: Optional[Mapping[str, Any] | AutonomousLoopConfig] = None,
    knowledge_k: int = 3,
) -> AutonomousControlLoop:
    """Construct the current SLAI outer autonomy owner for one RoboCar."""

    stages = RoboCarAutonomyStages(
        car=car,
        world_model=world_model,
        kpi_tracker=kpi_tracker,
        adaptation_guard=adaptation_guard,
        knowledge_k=knowledge_k,
    )
    loop = AutonomousControlLoop(
        stages.stage_mapping(),
        shared_memory=getattr(car, "shared_memory", None),
        handler=stages.handle_failure,
        observation_provider=stages.observation,
        config=config,
    )
    # Expose the same concrete loop instance to stage authorization so it can
    # consume the resolved min_action_confidence without duplicating config.
    try:
        setattr(car, "autonomy_loop", loop)
    except Exception:
        pass
    return loop


__all__ = ["RoboCarAutonomyStages", "build_robocar_autonomy_loop"]
