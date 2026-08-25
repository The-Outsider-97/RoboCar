"""SLAI-root entry point for the nested RoboCar package.

Expected layout:
    SLAI/
      rc_main.py
      RoboCar/
      src/
      logs/

This entry point never starts vehicle motion by itself. It initializes the
hardware boundary at neutral, starts sensor ingestion, initializes the RoboCar
SLAI integration, services the synchronous vehicle watchdog, publishes optional
health summaries, and remains available for higher-level callers.

The watchdog is intentionally serviced from this outer runtime loop rather than
from a hidden background thread. A critical watchdog report therefore follows
RoboCar's stop-first enforcement path before any recovery handling occurs.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

from typing import Optional

from RoboCar.robocar import RoboCar


# Preserve the existing launcher cadence. This is a supervisory/service cadence,
# not a claim that 50 ms is a calibrated motion-control deadline.
_SERVICE_INTERVAL_S = 0.05


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SLAI RoboCar runtime")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to rc_configs.yaml; default is RoboCar/configs/rc_configs.yaml",
    )
    parser.add_argument(
        "--port",
        default=None,
        help="Override Pico serial port (for example /dev/ttyACM0)",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Explicitly allow synthetic sensors/PWM when hardware is unavailable",
    )
    parser.add_argument(
        "--health-interval",
        type=float,
        default=5.0,
        help="Seconds between health summaries; set <=0 to suppress periodic output",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    car: Optional[RoboCar] = None
    stopping = False

    def request_stop(_signum=None, _frame=None) -> None:
        nonlocal stopping
        stopping = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, request_stop)
        except (ValueError, OSError):
            pass

    try:
        car = RoboCar(
            config_path=args.config,
            sensor_port=args.port,
            allow_simulation=args.simulate,
        )
        car.start()
        print(
            "RoboCar initialized at neutral; watchdog supervision active. "
            "Press Ctrl+C to stop."
        )

        interval = float(args.health_interval)
        next_health = time.monotonic()

        while not stopping:
            # VehicleWatchdog is synchronous by design. Service it before
            # diagnostics so a critical condition follows the enforcing
            # stop/recovery path rather than being merely reported by health().
            car.service()

            now = time.monotonic()
            if interval > 0.0 and now >= next_health:
                print(json.dumps(car.health(), default=str, indent=2))
                next_health = now + interval

            time.sleep(_SERVICE_INTERVAL_S)

        return 0
    except Exception as exc:
        print(
            f"RoboCar startup/runtime failure: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        if car is not None:
            try:
                car.close()
            except Exception as exc:
                print(
                    f"RoboCar shutdown failure: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    raise SystemExit(main())
