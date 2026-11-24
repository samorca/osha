from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    TextIO,
)

from src.telemetry.identity import (
    IdentityMatch,
    IdentityResolver,
)

from src.telemetry.redaction import (
    DEFAULT_POLICY,
    RedactionPolicy,
    contains_redaction,
    redact_native_record,
)

from src.telemetry.resource import (
    resource_from_record,
)

from src.telemetry.schema import (
    Provider,
    ResourceIdentity,
    Signal,
    TelemetryRecord,
    build_record,
)

from src.telemetry.signal import (
    signal_from_record,
)

from src.telemetry.validation import (
    ValidationResult,
    require_valid,
    validate_record,
    validation_summary,
)


SUPPORTED_PROVIDERS = {item.value for item in Provider}


@dataclass
class NormalizationResult:
    """
    Result of normalizing one collector record.

    telemetry:
        Canonical telemetry when normalization
        reached the record-construction stage.

    validation:
        Structural and consistency validation.

    error:
        Error details when normalization failed
        before a canonical record was built.

    identity_matches:
        Candidate or merged identity evidence
        produced by IdentityResolver.
    """

    telemetry: Optional[TelemetryRecord] = None

    validation: Optional[ValidationResult] = None

    error: Optional[Dict[str, Any]] = None

    identity_matches: List[IdentityMatch] = None

    source_line: Optional[int] = None

    source_name: Optional[str] = None

    def __post_init__(
        self,
    ) -> None:
        if self.identity_matches is None:
            self.identity_matches = []

    @property
    def valid(self) -> bool:
        return (
            self.telemetry is not None
            and self.validation is not None
            and self.validation.valid
            and self.error is None
        )

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "source_line": (self.source_line),
            "source_name": (self.source_name),
            "telemetry": (
                self.telemetry.to_dict() if self.telemetry is not None else None
            ),
            "validation": (
                self.validation.to_dict() if self.validation is not None else None
            ),
            "error": self.error,
            "identity_matches": [match.to_dict() for match in self.identity_matches],
        }


@dataclass
class PreparedRecord:
    """
    Internal two-pass normalization object.

    Resource identities are built first and
    resolved as a batch before signals are built.
    This prevents a later canonical identity from
    making an earlier signal reference stale.
    """

    native: Dict[str, Any]

    resource: ResourceIdentity

    source_line: Optional[int] = None

    source_name: Optional[str] = None

    identity_matches: List[IdentityMatch] = None

    def __post_init__(
        self,
    ) -> None:
        if self.identity_matches is None:
            self.identity_matches = []


def optional_string(
    value: Any,
) -> Optional[str]:
    if value is None:
        return None

    text = str(value).strip()

    return text if text else None


def normalize_provider(
    value: Any,
) -> str:
    """
    Normalize a collector/provider identifier.

    Unknown provider names remain usable, though
    validation may produce a warning.
    """

    text = optional_string(value)

    if text is None:
        return Provider.OTHER.value

    normalized = (
        text.lower()
        .replace(
            "-",
            "_",
        )
        .replace(
            " ",
            "_",
        )
    )

    aliases = {
        "k8s": (Provider.KUBERNETES.value),
        "kube": (Provider.KUBERNETES.value),
        "open_stack": (Provider.OPENSTACK.value),
        "gcp": (Provider.GOOGLE_CLOUD.value),
        "googlecloud": (Provider.GOOGLE_CLOUD.value),
        "ibmcloud": (Provider.IBM_CLOUD.value),
    }

    return aliases.get(
        normalized,
        normalized,
    )


def infer_provider(
    record_data: Mapping[str, Any],
    *,
    provider_hint: Optional[str] = None,
) -> str:
    """
    Determine the provider using explicit data
    first, then a caller-supplied hint.

    Weak inference from signal names is used only
    as a final compatibility fallback.
    """

    explicit = optional_string(record_data.get("provider"))

    if explicit:
        return normalize_provider(explicit)

    if provider_hint:
        return normalize_provider(provider_hint)

    record_type = (optional_string(record_data.get("record_type")) or "").lower()

    if "openstack" in record_type:
        return Provider.OPENSTACK.value

    if "kubernetes" in record_type:
        return Provider.KUBERNETES.value

    if "linux" in record_type:
        return Provider.LINUX.value

    payload = record_data.get("payload") or {}

    if isinstance(
        payload,
        Mapping,
    ):
        if any(
            key in payload
            for key in (
                "pod_uid",
                "pod_name",
                "node_name",
                "namespace",
                "container_name",
            )
        ):
            return Provider.KUBERNETES.value

        if any(
            key in record_data
            for key in (
                "cloud_id",
                "region",
            )
        ):
            return Provider.OPENSTACK.value

        if any(
            key in payload
            for key in (
                "machine_id",
                "system_uuid",
                "pid",
                "filesystem",
                "mountpoint",
            )
        ):
            return Provider.LINUX.value

    return Provider.OTHER.value


def prepare_native_record(
    record_data: Mapping[str, Any],
    *,
    provider_hint: Optional[str] = None,
    environment_id: Optional[str] = None,
    policy: RedactionPolicy = (DEFAULT_POLICY),
    hash_salt: str = "",
) -> Dict[str, Any]:
    """
    Copy, add required normalization context,
    and redact one native collector record.

    The input mapping is never modified.
    """

    prepared = dict(record_data)

    prepared["provider"] = infer_provider(
        prepared,
        provider_hint=(provider_hint),
    )

    if environment_id:
        prepared["environment_id"] = environment_id

    elif not optional_string(prepared.get("environment_id")):
        prepared["environment_id"] = "default"

    redacted = redact_native_record(
        prepared,
        policy=policy,
        hash_salt=hash_salt,
    )

    return redacted


def prepare_record(
    record_data: Mapping[str, Any],
    *,
    provider_hint: Optional[str] = None,
    environment_id: Optional[str] = None,
    policy: RedactionPolicy = (DEFAULT_POLICY),
    hash_salt: str = "",
    source_line: Optional[int] = None,
    source_name: Optional[str] = None,
) -> PreparedRecord:
    """
    Run the first normalization pass:

      context injection
      redaction
      resource construction
    """

    native = prepare_native_record(
        record_data,
        provider_hint=provider_hint,
        environment_id=(environment_id),
        policy=policy,
        hash_salt=hash_salt,
    )

    resource = resource_from_record(
        native,
        policy=policy,
    )

    return PreparedRecord(
        native=native,
        resource=resource,
        source_line=source_line,
        source_name=source_name,
    )


def finalize_prepared_record(
    prepared: PreparedRecord,
    *,
    policy: RedactionPolicy = (DEFAULT_POLICY),
    strict: bool = False,
) -> NormalizationResult:
    """
    Complete normalization after resource
    identity resolution:

      build signal
      build canonical record
      validate
    """

    try:
        signal = signal_from_record(
            prepared.native,
            prepared.resource,
            policy=policy,
        )

        telemetry = build_record(
            resource=(prepared.resource),
            signal=signal,
            relationships=[],
            native=prepared.native,
        )

        validation = validate_record(telemetry)

        if strict:
            require_valid(telemetry)

        return NormalizationResult(
            telemetry=telemetry,
            validation=validation,
            identity_matches=(prepared.identity_matches),
            source_line=(prepared.source_line),
            source_name=(prepared.source_name),
        )

    except Exception as exc:
        return NormalizationResult(
            error={
                "error_type": (type(exc).__name__),
                "message": str(exc),
                "stage": "finalize",
            },
            identity_matches=(prepared.identity_matches),
            source_line=(prepared.source_line),
            source_name=(prepared.source_name),
        )


def normalize_record(
    record_data: Mapping[str, Any],
    *,
    provider_hint: Optional[str] = None,
    environment_id: Optional[str] = None,
    policy: RedactionPolicy = (DEFAULT_POLICY),
    identity_resolver: Optional[IdentityResolver] = None,
    hash_salt: str = "",
    strict: bool = False,
    source_line: Optional[int] = None,
    source_name: Optional[str] = None,
) -> NormalizationResult:
    """
    Normalize one provider-native record.

    For large batches, normalize_records() is
    preferred because it performs identity
    resolution before building signals.
    """

    try:
        prepared = prepare_record(
            record_data,
            provider_hint=(provider_hint),
            environment_id=(environment_id),
            policy=policy,
            hash_salt=hash_salt,
            source_line=source_line,
            source_name=source_name,
        )

        if identity_resolver:
            (
                prepared.resource,
                prepared.identity_matches,
            ) = identity_resolver.register(prepared.resource)

        return finalize_prepared_record(
            prepared,
            policy=policy,
            strict=strict,
        )

    except Exception as exc:
        return NormalizationResult(
            error={
                "error_type": (type(exc).__name__),
                "message": str(exc),
                "stage": "prepare",
            },
            source_line=source_line,
            source_name=source_name,
        )


def normalize_records(
    records: Iterable[Mapping[str, Any]],
    *,
    provider_hint: Optional[str] = None,
    environment_id: Optional[str] = None,
    policy: RedactionPolicy = (DEFAULT_POLICY),
    identity_resolver: Optional[IdentityResolver] = None,
    hash_salt: str = "",
    strict: bool = False,
    source_name: Optional[str] = None,
) -> List[NormalizationResult]:
    """
    Normalize a batch using two passes.

    Pass 1:
        redact and construct resources

    Pass 2:
        resolve identities, then build signals

    This ensures signals reference the resolved
    resource identity available for the batch.
    """

    resolver = identity_resolver or IdentityResolver()

    prepared_records: List[PreparedRecord] = []

    early_results: List[NormalizationResult] = []

    for line_number, item in enumerate(
        records,
        start=1,
    ):
        if not isinstance(
            item,
            Mapping,
        ):
            early_results.append(
                NormalizationResult(
                    error={
                        "error_type": ("TypeError"),
                        "message": ("Collector record must be a mapping"),
                        "stage": "input",
                    },
                    source_line=line_number,
                    source_name=source_name,
                )
            )

            continue

        try:
            prepared_records.append(
                prepare_record(
                    item,
                    provider_hint=(provider_hint),
                    environment_id=(environment_id),
                    policy=policy,
                    hash_salt=hash_salt,
                    source_line=(line_number),
                    source_name=(source_name),
                )
            )

        except Exception as exc:
            early_results.append(
                NormalizationResult(
                    error={
                        "error_type": (type(exc).__name__),
                        "message": str(exc),
                        "stage": "prepare",
                    },
                    source_line=line_number,
                    source_name=source_name,
                )
            )

    # Register resources before signal creation.
    for prepared in prepared_records:
        (
            prepared.resource,
            prepared.identity_matches,
        ) = resolver.register(prepared.resource)

    results = list(early_results)

    for prepared in prepared_records:
        results.append(
            finalize_prepared_record(
                prepared,
                policy=policy,
                strict=strict,
            )
        )

    results.sort(
        key=lambda result: result.source_line if result.source_line is not None else 0
    )

    return results


def read_jsonl(
    source: TextIO,
) -> Iterator[tuple[int, Any]]:
    """
    Read JSON Lines without silently dropping
    malformed data.

    JSON parsing errors are yielded as structured
    objects for quarantine.
    """

    for line_number, line in enumerate(
        source,
        start=1,
    ):
        text = line.strip()

        if not text:
            continue

        try:
            yield (
                line_number,
                json.loads(text),
            )

        except json.JSONDecodeError as exc:
            yield (
                line_number,
                {
                    "__normalization_error__": {
                        "error_type": ("JSONDecodeError"),
                        "message": str(exc),
                        "raw_line": (text[:4096]),
                    }
                },
            )


def normalize_jsonl(
    input_path: Path,
    *,
    provider_hint: Optional[str] = None,
    environment_id: Optional[str] = None,
    policy: RedactionPolicy = (DEFAULT_POLICY),
    identity_resolver: Optional[IdentityResolver] = None,
    hash_salt: str = "",
    strict: bool = False,
) -> List[NormalizationResult]:
    """
    Read and normalize one JSONL collector file.
    """

    raw_records: List[Mapping[str, Any]] = []

    line_numbers: List[int] = []

    parse_failures: List[NormalizationResult] = []

    with input_path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for (
            line_number,
            item,
        ) in read_jsonl(source):
            if (
                isinstance(
                    item,
                    Mapping,
                )
                and "__normalization_error__" in item
            ):
                parse_failures.append(
                    NormalizationResult(
                        error=dict(item["__normalization_error__"]),
                        source_line=(line_number),
                        source_name=str(input_path),
                    )
                )

                continue

            if not isinstance(
                item,
                Mapping,
            ):
                parse_failures.append(
                    NormalizationResult(
                        error={
                            "error_type": ("TypeError"),
                            "message": ("JSONL record must be an object"),
                            "stage": "input",
                        },
                        source_line=(line_number),
                        source_name=str(input_path),
                    )
                )

                continue

            raw_records.append(item)

            line_numbers.append(line_number)

    normalized = normalize_records(
        raw_records,
        provider_hint=provider_hint,
        environment_id=(environment_id),
        policy=policy,
        identity_resolver=(identity_resolver),
        hash_salt=hash_salt,
        strict=strict,
        source_name=str(input_path),
    )

    # normalize_records assigns sequential source
    # lines for an in-memory batch. Restore the
    # original JSONL line numbers.
    for result, original_line in zip(
        normalized,
        line_numbers,
    ):
        result.source_line = original_line

    normalized.extend(parse_failures)

    normalized.sort(
        key=lambda result: result.source_line if result.source_line is not None else 0
    )

    return normalized


def write_normalized_jsonl(
    path: Path,
    results: Iterable[NormalizationResult],
    *,
    truncate: bool = True,
) -> int:
    """
    Write valid canonical telemetry records.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = "w" if truncate else "a"

    written = 0

    with path.open(
        mode,
        encoding="utf-8",
    ) as output:
        for result in results:
            if not result.valid:
                continue

            if result.telemetry is None:
                continue

            output.write(result.telemetry.to_json() + "\n")

            written += 1

    return written


def write_quarantine_jsonl(
    path: Path,
    results: Iterable[NormalizationResult],
    *,
    truncate: bool = True,
) -> int:
    """
    Write failed or invalid records separately.

    Quarantine output is diagnostic data and
    should receive restricted access.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode = "w" if truncate else "a"

    written = 0

    with path.open(
        mode,
        encoding="utf-8",
    ) as output:
        for result in results:
            if result.valid:
                continue

            output.write(
                json.dumps(
                    result.to_dict(),
                    ensure_ascii=False,
                    separators=(
                        ",",
                        ":",
                    ),
                    default=str,
                )
                + "\n"
            )

            written += 1

    return written


def normalization_summary(
    results: Iterable[NormalizationResult],
) -> Dict[str, Any]:
    """
    Summarize normalization and validation.
    """

    result_list = list(results)

    valid = [result for result in result_list if result.valid]

    invalid = [result for result in result_list if not result.valid]

    validations = [
        result.validation for result in result_list if result.validation is not None
    ]

    provider_counts: Dict[
        str,
        int,
    ] = {}

    signal_counts: Dict[
        str,
        int,
    ] = {}

    resource_counts: Dict[
        str,
        int,
    ] = {}

    redacted_records = 0

    identity_candidates = 0
    identity_merges = 0

    for result in result_list:
        if result.telemetry:
            telemetry = result.telemetry

            provider = telemetry.resource.provider

            signal_type = telemetry.signal.signal_type

            resource_type = telemetry.resource.resource_type

            provider_counts[provider] = (
                provider_counts.get(
                    provider,
                    0,
                )
                + 1
            )

            signal_counts[signal_type] = (
                signal_counts.get(
                    signal_type,
                    0,
                )
                + 1
            )

            resource_counts[resource_type] = (
                resource_counts.get(
                    resource_type,
                    0,
                )
                + 1
            )

            if contains_redaction(telemetry.native):
                redacted_records += 1

        for match in result.identity_matches:
            if match.decision == "candidate":
                identity_candidates += 1

            elif match.decision == "merge":
                identity_merges += 1

    return {
        "records_received": (len(result_list)),
        "valid_records": len(valid),
        "invalid_records": (len(invalid)),
        "redacted_records": (redacted_records),
        "provider_counts": (provider_counts),
        "signal_counts": (signal_counts),
        "resource_counts": (resource_counts),
        "identity": {
            "candidate_matches": (identity_candidates),
            "merged_matches": (identity_merges),
        },
        "validation": (validation_summary(validations)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Normalize OSHA collector JSONL into canonical telemetry.")
    )

    parser.add_argument(
        "--input",
        required=True,
        help=("Collector JSONL input path."),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=("Normalized JSONL output path."),
    )

    parser.add_argument(
        "--quarantine",
        default=None,
        help=("Optional invalid-record JSONL output path."),
    )

    parser.add_argument(
        "--provider",
        default=None,
        help=("Provider hint when the records do not contain a provider field."),
    )

    parser.add_argument(
        "--environment",
        default=None,
        help=("Override environment_id."),
    )

    parser.add_argument(
        "--hash-salt",
        default="",
        help=("Optional telemetry pseudonymization salt."),
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help=("Treat invalid normalized records as failed normalization."),
    )

    parser.add_argument(
        "--append",
        action="store_true",
        help=("Append instead of replacing output files."),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)

    output_path = Path(args.output)

    quarantine_path = Path(args.quarantine) if args.quarantine else None

    if not input_path.is_file():
        print(
            (f"Input file does not exist: {input_path}"),
            file=sys.stderr,
        )

        return 2

    resolver = IdentityResolver()

    try:
        results = normalize_jsonl(
            input_path,
            provider_hint=(args.provider),
            environment_id=(args.environment),
            identity_resolver=(resolver),
            hash_salt=(args.hash_salt),
            strict=args.strict,
        )

    except OSError as exc:
        print(
            (f"Unable to read input: {exc}"),
            file=sys.stderr,
        )

        return 1

    written = write_normalized_jsonl(
        output_path,
        results,
        truncate=(not args.append),
    )

    quarantined = 0

    if quarantine_path:
        quarantined = write_quarantine_jsonl(
            quarantine_path,
            results,
            truncate=(not args.append),
        )

    summary = normalization_summary(results)

    print("Telemetry normalization complete.")

    print(f"Input: {input_path}")

    print(f"Normalized records: {written}")

    print(f"Output: {output_path}")

    print(f"Invalid records: {summary['invalid_records']}")

    if quarantine_path:
        print((f"Quarantined records: {quarantined}"))

        print(f"Quarantine: {quarantine_path}")

    print("\nProvider counts:")

    for provider, count in sorted(summary["provider_counts"].items()):
        print(f"  {provider}: {count}")

    print("\nSignal counts:")

    for signal_type, count in sorted(summary["signal_counts"].items()):
        print(f"  {signal_type}: {count}")

    if summary["invalid_records"] > 0 and args.strict:
        return 3

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
