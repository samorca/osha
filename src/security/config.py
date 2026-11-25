from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AuthenticationConfig:
    enabled: bool = True
    require_authentication: bool = True
    accepted_methods: tuple[str, ...] = ("certificate", "service", "token")
    session_lifetime_seconds: int = 3600

    def validate(self) -> None:
        if self.session_lifetime_seconds < 1:
            raise ValueError("session_lifetime_seconds must be >= 1")
        if not self.accepted_methods or any(not item.strip() for item in self.accepted_methods):
            raise ValueError("accepted_methods must contain non-empty method names")


@dataclass(frozen=True)
class AuthorizationConfig:
    enabled: bool = True
    deny_by_default: bool = True
    require_explicit_allow: bool = True
    allow_system_principal: bool = False

    def validate(self) -> None:
        return None


@dataclass(frozen=True)
class ApprovalConfig:
    enabled: bool = True
    required_for_mutating_actions: bool = True
    default_roles: tuple[str, ...] = ("security_admin",)

    def validate(self) -> None:
        if self.enabled and self.required_for_mutating_actions and not self.default_roles:
            raise ValueError("Mutating-action approval requires at least one approval role")


@dataclass(frozen=True)
class SecretsConfig:
    enabled: bool = True
    default_provider: str = "environment"
    allowed_providers: tuple[str, ...] = ("environment",)
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300

    def validate(self) -> None:
        if not self.allowed_providers or any(not item.strip() for item in self.allowed_providers):
            raise ValueError("allowed_providers must contain non-empty provider names")
        if self.default_provider not in self.allowed_providers:
            raise ValueError("default_provider must be included in allowed_providers")
        if self.cache_ttl_seconds < 1:
            raise ValueError("cache_ttl_seconds must be >= 1")


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = True
    maximum_buffered_records: int = 10_000
    record_success: bool = True
    record_failures: bool = True
    record_policy_blocks: bool = True

    def validate(self) -> None:
        if self.maximum_buffered_records < 1:
            raise ValueError("maximum_buffered_records must be >= 1")


@dataclass(frozen=True)
class SecurityConfig:
    authentication: AuthenticationConfig = AuthenticationConfig()
    authorization: AuthorizationConfig = AuthorizationConfig()
    approval: ApprovalConfig = ApprovalConfig()
    secrets: SecretsConfig = SecretsConfig()
    audit: AuditConfig = AuditConfig()
    redact_sensitive: bool = True

    def validate(self) -> None:
        self.authentication.validate()
        self.authorization.validate()
        self.approval.validate()
        self.secrets.validate()
        self.audit.validate()


def safe_config_summary(config: SecurityConfig) -> dict[str, Any]:
    config.validate()
    return {
        "authentication": {
            "enabled": config.authentication.enabled,
            "require_authentication": config.authentication.require_authentication,
            "accepted_methods": list(config.authentication.accepted_methods),
            "session_lifetime_seconds": config.authentication.session_lifetime_seconds,
        },
        "authorization": {
            "enabled": config.authorization.enabled,
            "deny_by_default": config.authorization.deny_by_default,
            "require_explicit_allow": config.authorization.require_explicit_allow,
            "allow_system_principal": config.authorization.allow_system_principal,
        },
        "approval": {
            "enabled": config.approval.enabled,
            "required_for_mutating_actions": config.approval.required_for_mutating_actions,
            "default_roles": list(config.approval.default_roles),
        },
        "secrets": {
            "enabled": config.secrets.enabled,
            "default_provider": config.secrets.default_provider,
            "allowed_providers": list(config.secrets.allowed_providers),
            "cache_enabled": config.secrets.cache_enabled,
            "cache_ttl_seconds": config.secrets.cache_ttl_seconds,
        },
        "audit": {
            "enabled": config.audit.enabled,
            "maximum_buffered_records": config.audit.maximum_buffered_records,
            "record_success": config.audit.record_success,
            "record_failures": config.audit.record_failures,
            "record_policy_blocks": config.audit.record_policy_blocks,
        },
        "redact_sensitive": config.redact_sensitive,
    }


__all__ = [
    "ApprovalConfig",
    "AuditConfig",
    "AuthenticationConfig",
    "AuthorizationConfig",
    "SecretsConfig",
    "SecurityConfig",
    "safe_config_summary",
]