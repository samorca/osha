from __future__ import annotations

import argparse
import json
import os
import signal
import time

from typing import (
    Optional,
    Sequence,
)


shutdown_requested = False


def request_shutdown(
    signum: int,
    frame: object,
) -> None:
    del signum, frame
    global shutdown_requested
    shutdown_requested = True


def parse_args(
    argv: Optional[
        Sequence[str]
    ] = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OSHA dry-run telemetry worker",
    )
    parser.add_argument(
        "--once",
        action="store_true",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(
            os.environ.get(
                "OSHA_WORKER_POLL_INTERVAL_SECONDS",
                "2",
            )
        ),
    )
    parser.add_argument(
        "--worker-id",
        default=os.environ.get(
            "OSHA_POD_NAME",
            os.environ.get(
                "OSHA_WORKER_ID",
                "worker-1",
            ),
        ),
    )
    parser.add_argument(
        "--json",
        action="store_true",
    )
    arguments = parser.parse_args(argv)

    if arguments.poll_interval < 0.001:
        parser.error(
            "--poll-interval must be >= 0.001"
        )

    return arguments


def main(
    argv: Optional[
        Sequence[str]
    ] = None,
) -> None:
    arguments = parse_args(argv)

    signal.signal(
        signal.SIGTERM,
        request_shutdown,
    )
    signal.signal(
        signal.SIGINT,
        request_shutdown,
    )

    cycle = 0

    while not shutdown_requested:
        cycle += 1
        payload = {
            "worker_id": arguments.worker_id,
            "worker_type": "telemetry",
            "cycle": cycle,
            "dry_run": True,
            "status": "idle",
        }

        if arguments.json:
            print(
                json.dumps(
                    payload,
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            print(
                "worker="
                + str(arguments.worker_id)
                + " cycle="
                + str(cycle)
                + " status=idle dry_run=true",
                flush=True,
            )

        if arguments.once:
            break

        time.sleep(
            arguments.poll_interval
        )


if __name__ == "__main__":
    main()
