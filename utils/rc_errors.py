
import time
import json
import hashlib

from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime

class RoboCarErrorType(Enum):
    """Enumeration of RoboCar-specific error categories"""
    HARDWARE_FAILURE = "Hardware Failure"
    SENSOR_FAILURE = "Sensor Malfunction"
    CONTROL_FAILURE = "Control System Failure"
    PLANNING_FAILURE = "Path Planning Failure"
    PERCEPTION_FAILURE = "Perception System Failure"
    LOCALIZATION_FAILURE = "Localization Error"
    SAFETY_VIOLATION = "Safety Protocol Violation"
    COMMUNICATION_FAILURE = "Inter-component Communication Failure"
    POWER_MANAGEMENT = "Power System Error"
    CONFIG_ERROR = "Configuration Error"

class RoboCarError(Exception):
    """Base exception for all autonomous vehicle errors with forensic capabilities"""
    def __init__(
        self,
        error_type: RoboCarErrorType,
        message: str,
        severity: str = "medium",
        context: Optional[Dict[str, Any]] = None,
        component: Optional[str] = None,
        remediation: Optional[str] = None
    ):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.component = component
        self.remediation = remediation
        self.timestamp = time.time()
        self.error_id = self._generate_error_id()
        self.forensic_hash = self._generate_forensic_hash()

    def _generate_error_id(self) -> str:
        """Generate unique error ID using context and timestamp"""
        unique_str = f"{self.timestamp}{self.error_type.value}{json.dumps(self.context)}"
        return hashlib.sha256(unique_str.encode()).hexdigest()[:12]

    def _generate_forensic_hash(self) -> str:
        """Create verifiable hash of error context for auditing"""
        data = {
            "timestamp": self.timestamp,
            "error_id": self.error_id,
            "error_type": self.error_type.value,
            "context": self.context,
            "component": self.component
        }
        return hashlib.sha3_256(json.dumps(data).encode()).hexdigest()

    def to_audit_dict(self) -> Dict[str, Any]:
        """Structured representation for logging and auditing"""
        return {
            "error_id": self.error_id,
            "type": self.error_type.value,
            "severity": self.severity,
            "message": str(self),
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat(),
            "forensic_hash": self.forensic_hash,
            "component": self.component,
            "context": self.context,
            "remediation": self.remediation
        }

# Hardware and Sensor Errors
class HardwareError(RoboCarError):
    """Base for hardware-related failures"""
    def __init__(self, device: str, operation: str, error_details: str):
        super().__init__(
            error_type=RoboCarErrorType.HARDWARE_FAILURE,
            message=f"Hardware failure in {device} during {operation}",
            severity="critical",
            context={
                "device": device,
                "operation": operation,
                "details": error_details
            },
            component="hardware",
            remediation="Check physical connections and device status"
        )

class SensorError(RoboCarError):
    """Base for sensor-related failures"""
    def __init__(self, sensor_type: str, reading: Any, expected_range: tuple):
        super().__init__(
            error_type=RoboCarErrorType.SENSOR_FAILURE,
            message=f"{sensor_type} sensor reading out of range: {reading}",
            severity="high",
            context={
                "sensor_type": sensor_type,
                "reading": reading,
                "expected_min": expected_range[0],
                "expected_max": expected_range[1]
            },
            component="sensors",
            remediation="Calibrate or replace sensor, check wiring"
        )

# Motion Control Errors
class ControlError(RoboCarError):
    """Base for control system failures"""
    def __init__(self, controller: str, command: Any, actual: Any):
        super().__init__(
            error_type=RoboCarErrorType.CONTROL_FAILURE,
            message=f"{controller} control mismatch: Command={command}, Actual={actual}",
            severity="high",
            context={
                "controller": controller,
                "command": command,
                "actual_value": actual
            },
            component="control_system",
            remediation="Check controller calibration and feedback mechanisms"
        )

class ActuatorError(HardwareError):
    """Specific hardware error for actuators"""
    def __init__(self, actuator_type: str, command: float, actual: float):
        super().__init__(
            device=actuator_type,
            operation="actuation",
            error_details=f"Commanded: {command}, Actual: {actual}"
        )
        self.error_type = RoboCarErrorType.HARDWARE_FAILURE

# Planning and Navigation Errors
class PlanningError(RoboCarError):
    """Base for path planning failures"""
    def __init__(self, planner: str, start: tuple, goal: tuple, reason: str):
        super().__init__(
            error_type=RoboCarErrorType.PLANNING_FAILURE,
            message=f"{planner} failed to find path from {start} to {goal}",
            severity="high",
            context={
                "planner": planner,
                "start_position": start,
                "goal_position": goal,
                "failure_reason": reason
            },
            component="planning_system",
            remediation="Verify map data and obstacle configuration"
        )

class ObstacleViolationError(PlanningError):
    """Path planning failed due to obstacles"""
    def __init__(self, position: tuple, obstacle_type: str):
        super().__init__(
            planner="A*",
            start="current_position",
            goal="destination",
            reason=f"Obstacle detected at {position} ({obstacle_type})"
        )
        self.context["obstacle_position"] = position
        self.context["obstacle_type"] = obstacle_type

# Perception and Localization Errors
class PerceptionError(RoboCarError):
    """Base for perception system failures"""
    def __init__(self, detector: str, confidence: float, threshold: float):
        super().__init__(
            error_type=RoboCarErrorType.PERCEPTION_FAILURE,
            message=f"{detector} detection confidence {confidence} < threshold {threshold}",
            severity="medium",
            context={
                "detector": detector,
                "confidence": confidence,
                "threshold": threshold
            },
            component="perception_system",
            remediation="Improve lighting conditions or recalibrate sensor"
        )

class LocalizationError(RoboCarError):
    """Base for localization failures"""
    def __init__(self, estimated: tuple, reference: tuple, max_error: float):
        super().__init__(
            error_type=RoboCarErrorType.LOCALIZATION_FAILURE,
            message=f"Localization drift: {max_error}m at {estimated}",
            severity="high",
            context={
                "estimated_position": estimated,
                "reference_position": reference,
                "position_error": max_error
            },
            component="localization_system",
            remediation="Sensor fusion recalibration needed"
        )

# Safety and Power Errors
class SafetyViolationError(RoboCarError):
    """Base for safety protocol violations"""
    def __init__(self, violation_type: str, threshold: float, actual: float):
        super().__init__(
            error_type=RoboCarErrorType.SAFETY_VIOLATION,
            message=f"Safety violation: {violation_type} ({actual} > {threshold})",
            severity="critical",
            context={
                "violation_type": violation_type,
                "threshold": threshold,
                "actual_value": actual
            },
            component="safety_system",
            remediation="Immediate stop required. Review safety margins"
        )

class PowerError(RoboCarError):
    """Base for power management issues"""
    def __init__(self, voltage: float, min_voltage: float, component: str):
        super().__init__(
            error_type=RoboCarErrorType.POWER_MANAGEMENT,
            message=f"Low voltage {voltage}V for {component} (min {min_voltage}V)",
            severity="critical",
            context={
                "voltage": voltage,
                "min_required": min_voltage,
                "affected_component": component
            },
            component="power_system",
            remediation="Check power supply and battery health"
        )

# Communication and Configuration Errors
class CommunicationError(RoboCarError):
    """Base for inter-component communication failures"""
    def __init__(self, source: str, destination: str, message_type: str):
        super().__init__(
            error_type=RoboCarErrorType.COMMUNICATION_FAILURE,
            message=f"Comm failure: {source} → {destination} ({message_type})",
            severity="high",
            context={
                "source_component": source,
                "destination_component": destination,
                "message_type": message_type
            },
            component="communication_system",
            remediation="Check communication buses and protocols"
        )

class ConfigurationError(RoboCarError):
    """Base for configuration-related errors"""
    def __init__(self, parameter: str, value: Any, valid_range: tuple):
        super().__init__(
            error_type=RoboCarErrorType.CONFIG_ERROR,
            message=f"Invalid config: {parameter}={value}",
            severity="medium",
            context={
                "parameter": parameter,
                "value": value,
                "valid_min": valid_range[0],
                "valid_max": valid_range[1]
            },
            component="configuration",
            remediation="Validate configuration files and parameters"
        )


class GNSSError(RuntimeError):
    pass


class GNSSProtocolError(GNSSError):
    pass


class GNSSTransportError(GNSSError):
    pass


class GNSSConfigurationError(GNSSError):
    pass


__all__ = [
    "RoboCarErrorType",
    "RoboCarError",
    "HardwareError",
    "SensorError",
    "ControlError",
    "ActuatorError",
    "PlanningError",
    "ObstacleViolationError",
    "PerceptionError",
    "LocalizationError",
    "SafetyViolationError",
    "PowerError",
    "CommunicationError",
    "ConfigurationError",
    "GNSSError",
    "GNSSProtocolError",
    "GNSSTransportError",
    "GNSSConfigurationError",
]
