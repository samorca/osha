from collections import defaultdict

from ...schemas.event import Event


def correlate_events(
    events: list[Event],
) -> dict[str, list[Event]]:

    groups: dict[str, list[Event]] = defaultdict(list)

    for event in events:

        correlation_key = (
            event.correlation_id
            or event.source_id
        )

        groups[correlation_key].append(event)

    for event_group in groups.values():
        event_group.sort(
            key=lambda event: event.timestamp
        )

    return dict(groups)