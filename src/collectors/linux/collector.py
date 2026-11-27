from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Callable, List

from src.collectors.linux.configuration import (
    collect_configuration,
)
from src.collectors.linux.containers import (
    collect_containers,
)
from src.collectors.linux.journal import (
    collect_journal,
)
from src.collectors.linux.metrics import (
    collect_metrics,
)
from src.collectors.linux.network import (
    collect_network,
)
from src.collectors.linux.packages import (
    collect_packages,
)
from src.collectors.linux.processes import (
    collect_processes,
)
from src.collectors.linux.schema import (
    build_record,
)
from src.collectors.linux.security import (
    collect_security,
)
from src.collectors.linux.services import (
    collect_services,
)
from src.collectors.linux.sockets import (
    collect_sockets,
)
from src.collectors.linux.storage import (
    collect_storage,
)
from src.collectors.linux.system import (
    collect_system_identity,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Collect comprehensive Linux telemetry for OSHA.")
    )

    parser.add_argument(
        "--output",
        default=("dataset/live/linux/telemetry.jsonl"),
    )

    parser.add_argument(
        "--journal-lines",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--since",
        default=None,
    )

    parser.add_argument(
        "--skip-journal",
        action="store_true",
    )

    parser.add_argument(
        "--skip-packages",
        action="store_true",
    )

    parser.add_argument(
        "--skip-processes",
        action="store_true",
    )

    parser.add_argument(
        "--skip-configuration",
        action="store_true",
    )

    parser.add_argument(
        "--truncate-output",
        action="store_true",
        help=("Replace the output file instead of appending to it."),
    )

    return parser.parse_args()


def error_record(
    collector: str,
    exc: Exception,
) -> dict:
    return build_record(
        signal_type="collector_error",
        payload={
            "collector": collector,
            "error": (type(exc).__name__),
            "message": str(exc),
        },
    )


def run_collector(
    name: str,
    function: Callable[
        [],
        List[dict] | dict,
    ],
) -> List[dict]:
    print(f"Collecting {name}...")

    try:
        result = function()
    except Exception as exc:
        print(f"  {name}: failed: {exc}")

        return [
            error_record(
                name,
                exc,
            )
        ]

    if isinstance(result, dict):
        records = [result]
    else:
        records = list(result)

    print(f"  {name}: {len(records)} records")

    return records


def write_records(
    path: Path,
    records: List[dict],
    *,
    truncate: bool,
) -> int:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = "w" if truncate else "a"

    written = 0

    with path.open(
        mode,
        encoding="utf-8",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

            written += 1

    return written


def main():
    args = parse_args()

    if args.journal_lines < 1:
        raise ValueError("--journal-lines must be positive")

    print("Starting comprehensive Linux discovery...")

    records: List[dict] = []

    records.extend(
        run_collector(
            "system",
            lambda: [collect_system_identity()],
        )
    )

    records.extend(
        run_collector(
            "metrics",
            lambda: [collect_metrics()],
        )
    )

    records.extend(
        run_collector(
            "services",
            collect_services,
        )
    )

    if not args.skip_processes:
        records.extend(
            run_collector(
                "processes",
                collect_processes,
            )
        )

    records.extend(
        run_collector(
            "sockets",
            collect_sockets,
        )
    )

    records.extend(
        run_collector(
            "containers",
            collect_containers,
        )
    )

    records.extend(
        run_collector(
            "storage",
            collect_storage,
        )
    )

    records.extend(
        run_collector(
            "network",
            collect_network,
        )
    )

    if not args.skip_packages:
        records.extend(
            run_collector(
                "packages",
                collect_packages,
            )
        )

    if not args.skip_configuration:
        records.extend(
            run_collector(
                "configuration",
                collect_configuration,
            )
        )

    records.extend(
        run_collector(
            "security",
            collect_security,
        )
    )

    if not args.skip_journal:
        records.extend(
            run_collector(
                "journal",
                lambda: collect_journal(
                    lines=(args.journal_lines),
                    since=args.since,
                ),
            )
        )

    output_path = Path(args.output)

    written = write_records(
        output_path,
        records,
        truncate=(args.truncate_output),
    )

    counts = Counter(
        str(
            record.get(
                "signal_type",
                "unknown",
            )
        )
        for record in records
    )

    print("\nLinux discovery complete.")

    print(f"Records written: {written}")

    print(f"Output: {output_path}")

    print("\nSignal counts:")

    for signal, count in counts.most_common():
        print(f"  {signal}: {count}")


if __name__ == "__main__":
    main()
