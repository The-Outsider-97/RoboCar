from __future__ import annotations

import math
import time
import heapq
import ezdxf
import hashlib, json

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .motion_controller import MotionController, PIDSpeedController
from .wheel_encoder import WheelEncoder
from .main_sensor import SensorBus, SensorReading
from .modules.edt2d import distance_map_from_occupancy, inflate_obstacles
from .utils.config_loader import load_global_config, get_config_section
from .utils.rc_errors import *
from .utils.rc_helpers import *

from ..src.agents.agent_factory import AgentFactory
from ..src.agents.collaborative.shared_memory import SharedMemory
from ..src.agents.planning.planning_types import Task, TaskType, ResourceProfile
from logs.logger import get_logger, PrettyPrinter # pyright: ignore[reportMissingImports]

logger = get_logger("SLAI AI RC Car")
printer = PrettyPrinter()

MEM_FILE = "robot_memory.pkl"

# -----------------------
# Memory keys (single source of truth)
# -----------------------
K_MAP_LATEST         = "map:latest"                     # occupancy grid (dict or numpy; see OccupancyGrid schema below)
K_DETECTIONS_SIGNS   = "detections:signs"               # camera sign detections
K_GOAL_CURRENT       = "goal:current"                   # {"x":..,"y":..,"theta":..}
K_PLAN_CURRENT       = "plan:current"                   # [Waypoint,...]
K_ROUTE_TRAVELED     = "route:traveled"                 # appended as robot moves
K_SAFETY_STATE       = "safety:state"                   # {"estop":bool, "speed_cap": mps, ...}
K_DIRECTIVES         = "reasoning:directives"           # {"full_stop_until": t, "limit_speed": val, ...}
K_POSE_ESTIMATE      = "pose:estimate"                  # {"x":..,"y":..,"theta":..,"v":..}
K_CONFIG             = "robocar:config"                 # persist user preferences
K_ENCODER_TICKS      = "sensors:encoder:ticks_total"    # monotonically increasing integer
K_ULTRA_FRONT        = "sensors:ultra:front_m"
K_BATTERY_VOLT       = "power:vbat"
K_BATTERY_STATE      = "power:state"

# -------------------------------------------------------------
# robo_car implements:
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
