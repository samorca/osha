"""
OSHA Database Security.

Provides the public security interface for OSHA's database subsystem.

Security responsibilities include:

    - request/task-local database security context
    - tenant identity validation
    - PostgreSQL tenant binding
    - tenant isolation guards
    - security-oriented input validation
    - support for PostgreSQL Row-Level Security integration

Security architecture:

    Application / Worker
            |
            v
    DatabaseSecurityContext
            |
            v
    Validation + Tenant Guard
            |
            v
    PostgreSQL Transaction
            |
            v
    Row-Level Security
            |
            v
    Tenant-isolated data

Database security is defence in depth. Application validation and tenant
checks do not replace PostgreSQL permissions, Row-Level Security,
authentication, authorisation, TLS or secret management.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Security context
# ---------------------------------------------------------------------------

from .context import (
    DatabaseSecurityContext,
    PrivilegedContextRequiredError,
    SecurityContextError,
    SecurityContextMissingError,
    TenantContextMismatchError,
    clear_security_context,
    get_security_context,
    require_privileged_context,
    require_security_context,
    require_tenant,
    reset_security_context,
    security_context,
    set_security_context,
)


# ---------------------------------------------------------------------------
# PostgreSQL tenant security
# ---------------------------------------------------------------------------

from .tenant import (
    ACTOR_ID_SETTING,
    ACTOR_TYPE_SETTING,
    CORRELATION_ID_SETTING,
    PRIVILEGED_SETTING,
    REQUEST_ID_SETTING,
    TENANT_SETTING,
    PostgreSQLTenantState,
    TenantBindingError,
    TenantSecurityError,
    TenantVerificationError,
    UnsupportedTenantDatabaseError,
    bind_tenant_context,
    read_tenant_state,
    require_bound_tenant,
    tenant_rls_check_expression,
    tenant_rls_expression,
    verify_tenant_state,
)


# ---------------------------------------------------------------------------
# Security validation
# ---------------------------------------------------------------------------

from .validation import (
    MAX_AUDIT_MESSAGE_LENGTH,
    MAX_DESCRIPTION_LENGTH,
    MAX_IDENTIFIER_LENGTH,
    MAX_JSON_DEPTH,
    MAX_JSON_ITEMS,
    MAX_JSON_STRING_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PAGE_SIZE,
    MAX_PORT,
    MIN_PORT,
    InvalidChoiceError,
    InvalidHostnameError,
    InvalidIPAddressError,
    InvalidIdentifierError,
    InvalidMetadataError,
    InvalidPaginationError,
    InvalidPortError,
    InvalidRangeError,
    InvalidStringError,
    InvalidUUIDError,
    RequiredValueError,
    SecurityValidationError,
    require_same_tenant,
    validate_audit_message,
    validate_choice,
    validate_description,
    validate_enum_value,
    validate_finite_number,
    validate_hostname,
    validate_identifier,
    validate_ip_address,
    validate_metadata,
    validate_optional_hostname,
    validate_optional_ip_address,
    validate_optional_port,
    validate_optional_probability,
    validate_optional_string,
    validate_optional_uuid,
    validate_pagination,
    validate_port,
    validate_probability,
    validate_range,
    validate_required_string,
    validate_tenant_id,
    validate_time_window,
    validate_uuid,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Security context
    "DatabaseSecurityContext",
    "PrivilegedContextRequiredError",
    "SecurityContextError",
    "SecurityContextMissingError",
    "TenantContextMismatchError",
    "clear_security_context",
    "get_security_context",
    "require_privileged_context",
    "require_security_context",
    "require_tenant",
    "reset_security_context",
    "security_context",
    "set_security_context",

    # PostgreSQL tenant security
    "ACTOR_ID_SETTING",
    "ACTOR_TYPE_SETTING",
    "CORRELATION_ID_SETTING",
    "PRIVILEGED_SETTING",
    "REQUEST_ID_SETTING",
    "TENANT_SETTING",
    "PostgreSQLTenantState",
    "TenantBindingError",
    "TenantSecurityError",
    "TenantVerificationError",
    "UnsupportedTenantDatabaseError",
    "bind_tenant_context",
    "read_tenant_state",
    "require_bound_tenant",
    "tenant_rls_check_expression",
    "tenant_rls_expression",
    "verify_tenant_state",

    # Validation limits
    "MAX_AUDIT_MESSAGE_LENGTH",
    "MAX_DESCRIPTION_LENGTH",
    "MAX_IDENTIFIER_LENGTH",
    "MAX_JSON_DEPTH",
    "MAX_JSON_ITEMS",
    "MAX_JSON_STRING_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_PAGE_SIZE",
    "MAX_PORT",
    "MIN_PORT",

    # Validation exceptions
    "InvalidChoiceError",
    "InvalidHostnameError",
    "InvalidIPAddressError",
    "InvalidIdentifierError",
    "InvalidMetadataError",
    "InvalidPaginationError",
    "InvalidPortError",
    "InvalidRangeError",
    "InvalidStringError",
    "InvalidUUIDError",
    "RequiredValueError",
    "SecurityValidationError",

    # Validation helpers
    "require_same_tenant",
    "validate_audit_message",
    "validate_choice",
    "validate_description",
    "validate_enum_value",
    "validate_finite_number",
    "validate_hostname",
    "validate_identifier",
    "validate_ip_address",
    "validate_metadata",
    "validate_optional_hostname",
    "validate_optional_ip_address",
    "validate_optional_port",
    "validate_optional_probability",
    "validate_optional_string",
    "validate_optional_uuid",
    "validate_pagination",
    "validate_port",
    "validate_probability",
    "validate_range",
    "validate_required_string",
    "validate_tenant_id",
    "validate_time_window",
    "validate_uuid",
]
