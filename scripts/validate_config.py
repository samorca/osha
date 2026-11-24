#!/usr/bin/env python3
"""
OSHA configuration validator.

Validates configuration files under ./config without modifying them.

Checks:
    - Required configuration files
    - YAML syntax
    - YAML root structure
    - Required top-level sections
    - Basic value types
    - Numeric ranges
    - Production safety invariants
    - Remediation guardrails
    - Security guardrails
    - Infrastructure mutation boundaries
    - Secret-like configuration keys
    - Referenced local directories

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --environment production
    python scripts/validate_config.py --strict
    python scripts/validate_config.py --config-dir ./config
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    import yaml
except ImportError:
    print(
        "[ERROR] PyYAML is required.\n"
        "Install project dependencies first or run:\n"
        "  python -m pip install PyYAML",
        file=sys.stderr,
    )
    raise SystemExit(2)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG_DIR = PROJECT_ROOT / "config"


REQUIRED_FILES = (
    "base.yaml",
    "production.yaml",
    "logging.yaml",
    "models.yaml",
    "collectors.yaml",
    "correlation.yaml",
    "ai.yaml",
    "remediation.yaml",
    "security.yaml",
    "observability.yaml",
    "infrastructure.yaml",
)


EXPECTED_ROOTS = {
    "base.yaml": None,
    "production.yaml": None,
    "logging.yaml": "logging",
    "models.yaml": "models",
    "collectors.yaml": "collectors",
    "correlation.yaml": "correlation",
    "ai.yaml": "ai",
    "remediation.yaml": "remediation",
    "security.yaml": "security",
    "observability.yaml": "observability",
    "infrastructure.yaml": "infrastructure",
}


SECRET_KEY_PATTERN = re.compile(
    r"(^|_)(password|passwd|token|secret|api_key|apikey|private_key|"
    r"access_key|secret_key|client_secret)($|_)",
    re.IGNORECASE,
)


ALLOWED_SECRET_POLICY_KEYS = {
    "allow_plaintext",
    "allow_in_configuration_files",
    "allow_in_logs",
    "allow_in_artifacts",
    "allow_in_model_metadata",
    "allow_in_error_messages",
    "allow_plaintext_credentials",
    "allow_inline_credentials",
    "include_credentials",
    "include_secrets",
    "include_tokens",
    "expose_credentials",
    "expose_secrets",
    "log_credentials",
    "log_secrets",
    "redact_sensitive_data",
    "redact_credentials",
}


@dataclass
class ValidationResult:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    @property
    def valid(self) -> bool:
        return not self.errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate OSHA YAML configuration."
    )

    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_CONFIG_DIR,
        help="Configuration directory. Default: ./config",
    )

    parser.add_argument(
        "--environment",
        choices=("production", "development"),
        default="production",
        help="Environment to validate. Default: production",
    )

    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as validation failures.",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print errors and final status.",
    )

    return parser.parse_args()


def load_yaml(
    path: Path,
    result: ValidationResult,
) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)

    except yaml.YAMLError as exc:
        result.error(f"{path.name}: invalid YAML: {exc}")
        return None

    except OSError as exc:
        result.error(f"{path.name}: cannot read file: {exc}")
        return None

    if data is None:
        result.error(f"{path.name}: configuration is empty.")
        return None

    if not isinstance(data, dict):
        result.error(
            f"{path.name}: root YAML value must be a mapping/object."
        )
        return None

    return data


def get_path(
    data: Mapping[str, Any],
    dotted_path: str,
    default: Any = None,
) -> Any:
    current: Any = data

    for part in dotted_path.split("."):
        if not isinstance(current, Mapping):
            return default

        if part not in current:
            return default

        current = current[part]

    return current


def require_path(
    data: Mapping[str, Any],
    dotted_path: str,
    expected_type: type | tuple[type, ...],
    filename: str,
    result: ValidationResult,
) -> Any:
    value = get_path(data, dotted_path)

    if value is None:
        result.error(
            f"{filename}: missing required setting '{dotted_path}'."
        )
        return None

    if not isinstance(value, expected_type):
        if isinstance(expected_type, tuple):
            type_name = "/".join(t.__name__ for t in expected_type)
        else:
            type_name = expected_type.__name__

        result.error(
            f"{filename}: '{dotted_path}' must be {type_name}, "
            f"got {type(value).__name__}."
        )

    return value


def require_bool(
    data: Mapping[str, Any],
    dotted_path: str,
    filename: str,
    result: ValidationResult,
) -> bool | None:
    value = require_path(
        data,
        dotted_path,
        bool,
        filename,
        result,
    )

    return value if isinstance(value, bool) else None


def validate_required_files(
    config_dir: Path,
    result: ValidationResult,
) -> None:
    if not config_dir.exists():
        result.error(
            f"Configuration directory does not exist: {config_dir}"
        )
        return

    if not config_dir.is_dir():
        result.error(
            f"Configuration path is not a directory: {config_dir}"
        )
        return

    for filename in REQUIRED_FILES:
        path = config_dir / filename

        if not path.is_file():
            result.error(f"Missing required configuration: {filename}")


def validate_expected_root(
    filename: str,
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    expected_root = EXPECTED_ROOTS.get(filename)

    if expected_root is None:
        return

    if expected_root not in data:
        result.error(
            f"{filename}: expected top-level section "
            f"'{expected_root}'."
        )
        return

    if not isinstance(data[expected_root], dict):
        result.error(
            f"{filename}: '{expected_root}' must be a mapping/object."
        )


def walk_values(
    value: Any,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}" if prefix else key_string

            yield path, child
            yield from walk_values(child, path)

    elif isinstance(value, list):
        for index, child in enumerate(value):
            path = f"{prefix}[{index}]"
            yield path, child
            yield from walk_values(child, path)


def validate_possible_secrets(
    filename: str,
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    for path, value in walk_values(data):
        leaf = path.split(".")[-1]
        leaf = leaf.split("[", 1)[0]

        if leaf in ALLOWED_SECRET_POLICY_KEYS:
            continue

        if not SECRET_KEY_PATTERN.search(leaf):
            continue

        if value is None:
            continue

        if isinstance(value, bool):
            continue

        if isinstance(value, (dict, list)):
            continue

        if isinstance(value, str):
            normalized = value.strip()

            if not normalized:
                continue

            if normalized.startswith("${") and normalized.endswith("}"):
                continue

        result.error(
            f"{filename}: possible secret value stored at '{path}'. "
            "Secrets must not be committed to YAML configuration."
        )


def validate_base(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "base.yaml"

    require_path(data, "application", dict, filename, result)
    require_path(data, "runtime", dict, filename, result)
    require_path(data, "logging", dict, filename, result)
    require_path(data, "models", dict, filename, result)
    require_path(data, "collectors", dict, filename, result)
    require_path(data, "ai", dict, filename, result)
    require_path(data, "security", dict, filename, result)

    debug = require_bool(
        data,
        "application.debug",
        filename,
        result,
    )

    automatic_remediation = require_bool(
        data,
        "features.automatic_remediation",
        filename,
        result,
    )

    if debug is True:
        result.warning(
            "base.yaml: application.debug is enabled."
        )

    if automatic_remediation is True:
        result.warning(
            "base.yaml: features.automatic_remediation is enabled."
        )


def validate_production(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "production.yaml"

    environment = require_path(
        data,
        "application.environment",
        str,
        filename,
        result,
    )

    if environment is not None and environment != "production":
        result.error(
            "production.yaml: application.environment must be 'production'."
        )

    safety_expectations = {
        "application.debug": False,
        "remediation.execution_enabled": False,
        "remediation.dry_run": True,
        "remediation.require_approval": True,
        "security.fail_closed": True,
        "security.secrets.allow_plaintext": False,
        "security.allow_shell_execution": False,
        "security.allow_privileged_operations": False,
        "features.automatic_remediation": False,
        "safety.fail_closed": True,
        "safety.require_human_approval_for_mutation": True,
    }

    for dotted_path, expected in safety_expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: unsafe production setting "
                f"'{dotted_path}={value}'. Expected {expected}."
            )


def validate_remediation(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "remediation.yaml"

    expectations = {
        "remediation.execution_enabled": False,
        "remediation.dry_run": True,
        "remediation.fail_closed": True,
        "remediation.require_approval": True,
        "remediation.actions.require_allowlist": True,
        "remediation.actions.allow_unknown": False,
        "remediation.actions.allow_destructive": False,
        "remediation.actions.allow_arbitrary_commands": False,
        "remediation.execution.enabled": False,
        "remediation.execution.dry_run": True,
        "remediation.safety.deny_by_default": True,
        "remediation.safety.require_human_approval": True,
        "remediation.safety.block_unknown_actions": True,
        "remediation.safety.block_destructive_actions": True,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: '{dotted_path}' must be {expected} "
                "for the current production policy."
            )

    allowed_actions = get_path(
        data,
        "remediation.actions.maximum_actions_per_incident",
    )

    if allowed_actions is not None:
        if (
            not isinstance(allowed_actions, int)
            or isinstance(allowed_actions, bool)
            or allowed_actions < 1
        ):
            result.error(
                f"{filename}: maximum_actions_per_incident "
                "must be a positive integer."
            )


def validate_security(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "security.yaml"

    expectations = {
        "security.enabled": True,
        "security.fail_closed": True,
        "security.deny_by_default": True,
        "security.allow_security_bypass": False,
        "security.authentication.required": True,
        "security.authorization.required": True,
        "security.secrets.allow_plaintext": False,
        "security.secrets.allow_in_configuration_files": False,
        "security.credentials.expose_to_ai": False,
        "security.infrastructure.mutation.enabled": False,
        "security.privileged_operations.enabled": False,
        "security.command_execution.enabled": False,
        "security.ai.allow_direct_infrastructure_access": False,
        "security.ai.allow_direct_remediation": False,
        "security.policy.allow_ai_policy_override": False,
        "security.network.allow_insecure_connections": False,
        "security.failure.fail_closed": True,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: '{dotted_path}' violates "
                "the production security policy."
            )

    authorization_policy = require_path(
        data,
        "security.authorization.default_policy",
        str,
        filename,
        result,
    )

    if (
        authorization_policy is not None
        and authorization_policy.lower() != "deny"
    ):
        result.error(
            f"{filename}: authorization.default_policy must be 'deny'."
        )


def validate_infrastructure(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "infrastructure.yaml"

    expectations = {
        "infrastructure.fail_closed": True,
        "infrastructure.connection.verify_tls": True,
        "infrastructure.connection.allow_insecure_tls": False,
        "infrastructure.mutation.enabled": False,
        "infrastructure.mutation.require_remediation_layer": True,
        "infrastructure.mutation.require_authorization": True,
        "infrastructure.mutation.require_approval": True,
        "infrastructure.mutation.direct_mutation_from_collectors": False,
        "infrastructure.mutation.direct_mutation_from_ai": False,
        "infrastructure.safety.read_only_by_default": True,
        "infrastructure.safety.block_direct_ai_mutation": True,
        "infrastructure.safety.block_direct_collector_mutation": True,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: '{dotted_path}' violates "
                "the infrastructure safety boundary."
            )


def validate_ai(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "ai.yaml"

    expectations = {
        "ai.fail_safe": True,
        "ai.fail_closed": True,
        "ai.allow_direct_infrastructure_actions": False,
        "ai.allow_shell_execution": False,
        "ai.allow_privileged_execution": False,
        "ai.safety.require_decision_engine_for_actions": True,
        "ai.safety.require_remediation_layer_for_execution": True,
        "ai.safety.prohibit_direct_infrastructure_mutation": True,
        "ai.security.expose_credentials": False,
        "ai.security.expose_secrets": False,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: '{dotted_path}' violates the AI safety boundary."
            )

    validate_probability(
        data,
        "ai.confidence.minimum",
        filename,
        result,
    )

    validate_probability(
        data,
        "ai.confidence.decision_threshold",
        filename,
        result,
    )

    validate_probability(
        data,
        "ai.confidence.high_confidence",
        filename,
        result,
    )


def validate_probability(
    data: Mapping[str, Any],
    dotted_path: str,
    filename: str,
    result: ValidationResult,
) -> None:
    value = get_path(data, dotted_path)

    if value is None:
        result.error(
            f"{filename}: missing required setting '{dotted_path}'."
        )
        return

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        result.error(
            f"{filename}: '{dotted_path}' must be numeric."
        )
        return

    if not 0.0 <= float(value) <= 1.0:
        result.error(
            f"{filename}: '{dotted_path}' must be between 0 and 1."
        )


def validate_collectors(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "collectors.yaml"

    expectations = {
        "collectors.security.read_only_by_default": True,
        "collectors.security.allow_shell_commands": False,
        "collectors.security.allow_remote_execution": False,
        "collectors.security.allow_resource_mutation": False,
        "collectors.kubernetes.security.read_only": True,
        "collectors.kubernetes.security.allow_exec": False,
        "collectors.openstack.security.read_only": True,
        "collectors.openstack.security.allow_resource_mutation": False,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: collector safety setting "
                f"'{dotted_path}' must be {expected}."
            )


def validate_models(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "models.yaml"

    expectations = {
        "models.enabled": True,
        "models.fail_on_load_error": True,
        "models.allow_unvalidated_models": False,
        "models.security.validate_model_source": True,
        "models.security.require_local_models": True,
        "models.security.allow_remote_model_download": False,
        "models.security.allow_arbitrary_model_loading": False,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: model security setting "
                f"'{dotted_path}' must be {expected}."
            )


def validate_logging(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "logging.yaml"

    if require_bool(
        data,
        "logging.redaction.enabled",
        filename,
        result,
    ) is not True:
        result.error(
            f"{filename}: sensitive-data redaction must be enabled."
        )

    if require_bool(
        data,
        "logging.audit.enabled",
        filename,
        result,
    ) is not True:
        result.error(
            f"{filename}: audit logging must be enabled."
        )

    if get_path(data, "logging.audit.include_sensitive_data") is not False:
        result.error(
            f"{filename}: audit.include_sensitive_data must be false."
        )


def validate_observability(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "observability.yaml"

    if require_bool(
        data,
        "observability.redact_sensitive_data",
        filename,
        result,
    ) is not True:
        result.error(
            f"{filename}: redact_sensitive_data must be enabled."
        )

    sensitive_expectations = {
        "observability.tracing.include_credentials": False,
        "observability.tracing.include_secrets": False,
        "observability.diagnostics.include_environment_variables": False,
        "observability.diagnostics.include_credentials": False,
        "observability.diagnostics.include_secrets": False,
        "observability.privacy.include_credentials": False,
        "observability.privacy.include_tokens": False,
        "observability.privacy.include_secrets": False,
    }

    for dotted_path, expected in sensitive_expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: '{dotted_path}' must be {expected}."
            )


def validate_correlation(
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    filename = "correlation.yaml"

    expectations = {
        "correlation.enabled": True,
        "correlation.safety.correlation_cannot_execute_actions": True,
        "correlation.safety.correlation_cannot_modify_infrastructure": True,
    }

    for dotted_path, expected in expectations.items():
        value = require_bool(
            data,
            dotted_path,
            filename,
            result,
        )

        if value is not None and value is not expected:
            result.error(
                f"{filename}: '{dotted_path}' must be {expected}."
            )

    validate_probability(
        data,
        "correlation.scoring.minimum_score",
        filename,
        result,
    )


def run_file_specific_validation(
    filename: str,
    data: Mapping[str, Any],
    result: ValidationResult,
) -> None:
    validators = {
        "base.yaml": validate_base,
        "production.yaml": validate_production,
        "logging.yaml": validate_logging,
        "models.yaml": validate_models,
        "collectors.yaml": validate_collectors,
        "correlation.yaml": validate_correlation,
        "ai.yaml": validate_ai,
        "remediation.yaml": validate_remediation,
        "security.yaml": validate_security,
        "observability.yaml": validate_observability,
        "infrastructure.yaml": validate_infrastructure,
    }

    validator = validators.get(filename)

    if validator is not None:
        validator(data, result)


def validate_cross_file_invariants(
    configs: Mapping[str, Mapping[str, Any]],
    result: ValidationResult,
) -> None:
    remediation = configs.get("remediation.yaml")
    infrastructure = configs.get("infrastructure.yaml")
    security = configs.get("security.yaml")
    ai = configs.get("ai.yaml")

    if remediation and infrastructure:
        remediation_execution = get_path(
            remediation,
            "remediation.execution_enabled",
        )

        infrastructure_mutation = get_path(
            infrastructure,
            "infrastructure.mutation.enabled",
        )

        if (
            remediation_execution is False
            and infrastructure_mutation is True
        ):
            result.error(
                "Cross-file policy conflict: infrastructure mutation is "
                "enabled while remediation execution is disabled."
            )

    if remediation and security:
        remediation_approval = get_path(
            remediation,
            "remediation.require_approval",
        )

        security_approval = get_path(
            security,
            "security.remediation.require_approval",
        )

        if remediation_approval != security_approval:
            result.error(
                "Cross-file policy conflict: remediation approval settings "
                "do not agree between remediation.yaml and security.yaml."
            )

    if ai and infrastructure:
        ai_direct_mutation = get_path(
            ai,
            "ai.allow_direct_infrastructure_actions",
        )

        block_ai_mutation = get_path(
            infrastructure,
            "infrastructure.safety.block_direct_ai_mutation",
        )

        if ai_direct_mutation is True or block_ai_mutation is not True:
            result.error(
                "Cross-file policy conflict: direct AI infrastructure "
                "mutation is not fully blocked."
            )


def find_local_paths(
    data: Any,
    prefix: str = "",
) -> Iterable[tuple[str, str]]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            key_string = str(key)
            path = f"{prefix}.{key_string}" if prefix else key_string

            if (
                isinstance(value, str)
                and (
                    key_string.endswith("_directory")
                    or key_string == "directory"
                    or key_string.endswith("_path")
                )
            ):
                if value.startswith("./"):
                    yield path, value

            yield from find_local_paths(value, path)

    elif isinstance(data, list):
        for index, value in enumerate(data):
            yield from find_local_paths(
                value,
                f"{prefix}[{index}]",
            )


def validate_local_paths(
    configs: Mapping[str, Mapping[str, Any]],
    result: ValidationResult,
) -> None:
    checked: set[str] = set()

    for filename, data in configs.items():
        for config_path, path_string in find_local_paths(data):
            identifier = f"{filename}:{config_path}:{path_string}"

            if identifier in checked:
                continue

            checked.add(identifier)

            candidate = (PROJECT_ROOT / path_string).resolve()

            try:
                candidate.relative_to(PROJECT_ROOT.resolve())
            except ValueError:
                result.error(
                    f"{filename}: '{config_path}' resolves outside "
                    f"the project root: {path_string}"
                )
                continue

            if not candidate.exists():
                result.warning(
                    f"{filename}: referenced local path does not currently "
                    f"exist: {path_string}"
                )


def load_configurations(
    config_dir: Path,
    result: ValidationResult,
    quiet: bool,
) -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}

    for filename in REQUIRED_FILES:
        path = config_dir / filename

        if not path.is_file():
            continue

        data = load_yaml(path, result)

        if data is None:
            continue

        configs[filename] = data

        validate_expected_root(
            filename,
            data,
            result,
        )

        validate_possible_secrets(
            filename,
            data,
            result,
        )

        run_file_specific_validation(
            filename,
            data,
            result,
        )

        if not quiet:
            print(f"[OK] loaded {filename}")

    return configs


def print_report(
    result: ValidationResult,
    config_dir: Path,
    strict: bool,
) -> None:
    print()

    if result.warnings:
        print("Warnings:")

        for warning in result.warnings:
            print(f"  [WARN] {warning}")

        print()

    if result.errors:
        print("Errors:")

        for error in result.errors:
            print(f"  [ERROR] {error}")

        print()

    print(f"Configuration directory: {config_dir}")
    print(f"Errors: {len(result.errors)}")
    print(f"Warnings: {len(result.warnings)}")

    print()

    if result.errors:
        print("[FAILED] OSHA configuration validation failed.")

    elif strict and result.warnings:
        print(
            "[FAILED] Configuration produced warnings "
            "and --strict was requested."
        )

    else:
        print("[OK] OSHA configuration validation passed.")


def main() -> int:
    args = parse_args()

    config_dir = args.config_dir.expanduser()

    if not config_dir.is_absolute():
        config_dir = (PROJECT_ROOT / config_dir).resolve()
    else:
        config_dir = config_dir.resolve()

    result = ValidationResult()

    if not args.quiet:
        print("OSHA Configuration Validator")
        print(f"Project:     {PROJECT_ROOT}")
        print(f"Environment: {args.environment}")
        print(f"Config:      {config_dir}")
        print()

    validate_required_files(
        config_dir,
        result,
    )

    configs = load_configurations(
        config_dir,
        result,
        quiet=args.quiet,
    )

    validate_cross_file_invariants(
        configs,
        result,
    )

    validate_local_paths(
        configs,
        result,
    )

    print_report(
        result,
        config_dir,
        strict=args.strict,
    )

    if result.errors:
        return 1

    if args.strict and result.warnings:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())