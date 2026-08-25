"""Bounded learned-parameter adaptation guard for RoboCar.

Adaptive/Learning components may propose parameter changes, but this module is
the only authority that decides whether a proposal is eligible to be staged.
It is intentionally independent from SLAI so it can be tested deterministically.

The guard enforces:
* explicit allowlist;
* permanently denied names/prefixes;
* absolute value range;
* maximum absolute delta per proposal;
* maximum change rate per second;
* minimum evidence/sample count;
* optional confidence floor;
* external safety approval;
* snapshot-before-apply;
* auditable proposal lifecycle;
* explicit rollback.

It does not discover "safe" bounds.  Bounds must be configured from engineering
validation and physical testing.
"""

from __future__ import annotations

import copy
import math
import threading
import time
import uuid

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Protocol, Tuple


class AdaptationStatus(str, Enum):
    PROPOSED = "proposed"
    REJECTED = "rejected"
    APPROVED = "approved"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class ParameterRule:
    minimum: float
    maximum: float
    max_delta_per_proposal: float
    max_change_per_second: float
    minimum_samples: int
    minimum_confidence: float = 0.0

    def __post_init__(self) -> None:
        values = {
            "minimum": self.minimum,
            "maximum": self.maximum,
            "max_delta_per_proposal": self.max_delta_per_proposal,
            "max_change_per_second": self.max_change_per_second,
            "minimum_confidence": self.minimum_confidence,
        }
        for name, value in values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.minimum >= self.maximum:
            raise ValueError("minimum must be < maximum")
        if self.max_delta_per_proposal <= 0:
            raise ValueError("max_delta_per_proposal must be > 0")
        if self.max_change_per_second <= 0:
            raise ValueError("max_change_per_second must be > 0")
        if self.minimum_samples < 1:
            raise ValueError("minimum_samples must be >= 1")
        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class AdaptationProposal:
    proposal_id: str
    parameter: str
    current_value: float
    proposed_value: float
    evidence_samples: int
    confidence: float
    source: str
    reason: str
    created_wall: float
    created_monotonic: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AdaptationAuditRecord:
    proposal: AdaptationProposal
    status: AdaptationStatus
    decision_reason: str
    safety_result: Optional[Mapping[str, Any]] = None
    snapshot_before: Optional[Mapping[str, Any]] = None
    applied_wall: Optional[float] = None
    rolled_back_wall: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class AdaptationGuard:
    """Enforce bounded adaptation and maintain rollback/audit state."""

    DEFAULT_PERMANENT_DENY = frozenset(
        {
            "emergency_stop",
            "estop",
            "battery.v_critical",
            "battery.hard_cutoff",
            "power.v_critical",
            "motion.esc_min_us",
            "motion.esc_max_us",
            "motion.esc_neutral_us",
            "motion.servo_min_us",
            "motion.servo_max_us",
            "motion.servo_center_us",
            "motion.servo_max_angle_rad",
            "watchdog.sensor_frame_timeout_s",
            "watchdog.pico_heartbeat_timeout_s",
            "watchdog.control_cycle_deadline_s",
            "robocar.front_stop_distance_m",
        }
    )

    def __init__(
        self,
        rules: Mapping[str, ParameterRule],
        *,
        permanent_deny: Optional[set[str] | frozenset[str]] = None,
        audit_capacity: int = 512,
    ) -> None:
        if audit_capacity < 1:
            raise ValueError("audit_capacity must be >= 1")
        self.rules = dict(rules)
        self.permanent_deny = frozenset(
            self.DEFAULT_PERMANENT_DENY
            | frozenset(permanent_deny or ())
        )
        overlap = self.permanent_deny.intersection(self.rules)
        if overlap:
            raise ValueError(
                "permanently denied parameters cannot be allowlisted: "
                + ", ".join(sorted(overlap))
            )
        self._audit_capacity = int(audit_capacity)
        self._lock = threading.RLock()
        self._records: Dict[str, AdaptationAuditRecord] = {}
        self._order: list[str] = []
        self._last_applied: Dict[str, tuple[float, float]] = {}

    def propose(
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
        proposal = AdaptationProposal(
            proposal_id=f"adapt:{uuid.uuid4().hex}",
            parameter=str(parameter).strip(),
            current_value=self._finite(current_value, "current_value"),
            proposed_value=self._finite(proposed_value, "proposed_value"),
            evidence_samples=int(evidence_samples),
            confidence=self._probability(confidence),
            source=str(source).strip() or "unknown",
            reason=str(reason).strip() or "unspecified",
            created_wall=time.time(),
            created_monotonic=time.monotonic(),
            evidence=copy.deepcopy(dict(evidence or {})),
        )

        decision = self._validate(proposal)
        record = AdaptationAuditRecord(
            proposal=proposal,
            status=(
                AdaptationStatus.APPROVED
                if decision is None
                else AdaptationStatus.REJECTED
            ),
            decision_reason=decision or "bounded_checks_passed",
        )
        self._store(record)
        return record

    def safety_approve(
        self,
        proposal_id: str,
        safety_validator: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        *,
        context: Optional[Mapping[str, Any]] = None,
    ) -> AdaptationAuditRecord:
        """Require SafetyAgent-style ``validate_action(action, context)`` approval."""

        with self._lock:
            record = self._require_record(proposal_id)
            if record.status != AdaptationStatus.APPROVED:
                return record
            proposal = record.proposal

        result = safety_validator(
            {
                "name": "robocar_parameter_adaptation",
                "parameter": proposal.parameter,
                "current_value": proposal.current_value,
                "proposed_value": proposal.proposed_value,
                "delta": proposal.proposed_value - proposal.current_value,
                "source": proposal.source,
            },
            {
                "type": "robocar_adaptation_guard",
                "proposal_id": proposal.proposal_id,
                "evidence_samples": proposal.evidence_samples,
                "confidence": proposal.confidence,
                **dict(context or {}),
            },
        )
        if not isinstance(result, Mapping):
            raise TypeError("safety validator must return a mapping")

        approved = result.get("approved") is True
        with self._lock:
            record = self._require_record(proposal_id)
            record.safety_result = copy.deepcopy(dict(result))
            if not approved:
                record.status = AdaptationStatus.REJECTED
                record.decision_reason = "safety_agent_rejected"
            else:
                record.decision_reason = "bounded_checks_and_safety_approved"
            return copy.deepcopy(record)

    def apply(
        self,
        proposal_id: str,
        *,
        getter: Callable[[str], float],
        setter: Callable[[str, float], None],
        require_safety_approval: bool = True,
    ) -> AdaptationAuditRecord:
        """Apply an approved proposal with snapshot-first rollback state."""

        with self._lock:
            record = self._require_record(proposal_id)
            if record.status != AdaptationStatus.APPROVED:
                return copy.deepcopy(record)
            if require_safety_approval and not (
                isinstance(record.safety_result, Mapping)
                and record.safety_result.get("approved") is True
            ):
                record.status = AdaptationStatus.REJECTED
                record.decision_reason = "safety_approval_required"
                return copy.deepcopy(record)
            proposal = record.proposal

        actual_current = self._finite(getter(proposal.parameter), "actual_current")
        # Detect stale proposals: adaptation must not overwrite an intervening
        # configuration change it did not observe.
        if not math.isclose(
            actual_current,
            proposal.current_value,
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            with self._lock:
                record = self._require_record(proposal_id)
                record.status = AdaptationStatus.REJECTED
                record.decision_reason = "current_value_changed_since_proposal"
                return copy.deepcopy(record)

        snapshot = {
            "parameter": proposal.parameter,
            "value": actual_current,
            "captured_wall": time.time(),
        }
        try:
            setter(proposal.parameter, proposal.proposed_value)
            observed = self._finite(getter(proposal.parameter), "post_apply_value")
            if not math.isclose(
                observed,
                proposal.proposed_value,
                rel_tol=1e-9,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"post-apply verification mismatch: {observed} != "
                    f"{proposal.proposed_value}"
                )
        except Exception as exc:
            # Attempt immediate rollback before reporting the application failure.
            try:
                setter(proposal.parameter, actual_current)
            except Exception:
                pass
            with self._lock:
                record = self._require_record(proposal_id)
                record.status = AdaptationStatus.FAILED
                record.decision_reason = "apply_failed"
                record.snapshot_before = snapshot
                record.error = f"{type(exc).__name__}: {exc}"
                return copy.deepcopy(record)

        with self._lock:
            record = self._require_record(proposal_id)
            record.status = AdaptationStatus.APPLIED
            record.decision_reason = "applied"
            record.snapshot_before = snapshot
            record.applied_wall = time.time()
            self._last_applied[proposal.parameter] = (
                proposal.proposed_value,
                time.monotonic(),
            )
            return copy.deepcopy(record)

    def rollback(
        self,
        proposal_id: str,
        *,
        setter: Callable[[str, float], None],
    ) -> AdaptationAuditRecord:
        with self._lock:
            record = self._require_record(proposal_id)
            if record.status != AdaptationStatus.APPLIED:
                return copy.deepcopy(record)
            snapshot = dict(record.snapshot_before or {})
            parameter = record.proposal.parameter

        if "value" not in snapshot:
            raise RuntimeError("rollback snapshot is unavailable")

        try:
            setter(parameter, float(snapshot["value"]))
        except Exception as exc:
            with self._lock:
                record = self._require_record(proposal_id)
                record.status = AdaptationStatus.FAILED
                record.decision_reason = "rollback_failed"
                record.error = f"{type(exc).__name__}: {exc}"
                return copy.deepcopy(record)

        with self._lock:
            record = self._require_record(proposal_id)
            record.status = AdaptationStatus.ROLLED_BACK
            record.decision_reason = "rolled_back"
            record.rolled_back_wall = time.time()
            self._last_applied.pop(parameter, None)
            return copy.deepcopy(record)

    def audit(self, limit: Optional[int] = None) -> Tuple[AdaptationAuditRecord, ...]:
        with self._lock:
            ids = self._order if limit is None else self._order[-max(0, int(limit)):]
            return tuple(copy.deepcopy(self._records[item]) for item in ids)

    def _validate(self, proposal: AdaptationProposal) -> Optional[str]:
        name = proposal.parameter
        if not name:
            return "parameter_name_empty"
        if name in self.permanent_deny:
            return "parameter_permanently_denied"
        rule = self.rules.get(name)
        if rule is None:
            return "parameter_not_allowlisted"
        if proposal.evidence_samples < rule.minimum_samples:
            return "insufficient_evidence_samples"
        if proposal.confidence < rule.minimum_confidence:
            return "insufficient_confidence"
        if not rule.minimum <= proposal.proposed_value <= rule.maximum:
            return "proposed_value_outside_absolute_range"

        delta = abs(proposal.proposed_value - proposal.current_value)
        if delta > rule.max_delta_per_proposal:
            return "proposal_delta_exceeds_limit"

        with self._lock:
            last = self._last_applied.get(name)
        if last is not None:
            last_value, last_time = last
            elapsed = max(1e-9, proposal.created_monotonic - last_time)
            rate = abs(proposal.proposed_value - last_value) / elapsed
            if rate > rule.max_change_per_second:
                return "proposal_change_rate_exceeds_limit"
        return None

    def _store(self, record: AdaptationAuditRecord) -> None:
        with self._lock:
            proposal_id = record.proposal.proposal_id
            self._records[proposal_id] = record
            self._order.append(proposal_id)
            while len(self._order) > self._audit_capacity:
                removed = self._order.pop(0)
                self._records.pop(removed, None)

    def _require_record(self, proposal_id: str) -> AdaptationAuditRecord:
        record = self._records.get(str(proposal_id))
        if record is None:
            raise KeyError(f"Unknown adaptation proposal {proposal_id!r}")
        return record

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be numeric, not bool")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{name} must be finite")
        return result

    @classmethod
    def _probability(cls, value: Any) -> float:
        result = cls._finite(value, "confidence")
        if not 0.0 <= result <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        return result


__all__ = [
    "AdaptationStatus",
    "ParameterRule",
    "AdaptationProposal",
    "AdaptationAuditRecord",
    "AdaptationGuard",
]
