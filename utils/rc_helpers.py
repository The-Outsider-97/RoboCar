"""
Shared helper primitives for the Robotic Car subsystem.
"""

import hashlib
import json

from collections import deque
from datetime import datetime, timedelta
from typing import Any, Mapping, Optional

import numpy as np # type: ignore
import pandas as pd # type: ignore

from .rc_errors import *


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None

def _to_int(v) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None
