"""RoboCar deterministic domain modules.

``slai_autonomy`` is intentionally not imported here.  Import it explicitly
when the parent SLAI runtime is available:

    from RoboCar.modules.slai_autonomy import build_robocar_autonomy_loop

This keeps ordinary deterministic-module imports free from eager SLAI agent
construction/import dependencies.
"""

from .edt2d import *
from .world_model import *
from .trajectory_control import *
from .kpi_tracker import *
from .watchdog import *
from .adaptation_guard import *

from .edt2d import __all__ as _edt2d_exports
from .world_model import __all__ as _world_model_exports
from .trajectory_control import __all__ as _trajectory_control_exports
from .kpi_tracker import __all__ as _kpi_tracker_exports
from .watchdog import __all__ as _watchdog_exports
from .adaptation_guard import __all__ as _adaptation_guard_exports

__all__ = [
    *_edt2d_exports,
    *_world_model_exports,
    *_trajectory_control_exports,
    *_kpi_tracker_exports,
    *_watchdog_exports,
    *_adaptation_guard_exports,
]
