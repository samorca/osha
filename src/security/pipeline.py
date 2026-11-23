from __future__ import annotations

"""
Composition facade for OSHA security services.
"""

from dataclasses import (
    dataclass,
)

from typing import (
    Any,
    Dict,
    Optional,
)

from src.security.audit import (
    SecurityAuditLog,
)

from src.security.authentication import (
    AuthenticationService,
)

from src.security.authorization import (
    AuthorizationService,
)

from src.security.config import (
    SecurityConfig,
    safe_config_summary,
)

from src.security.identity import (
    IdentityRegistry,
)

from src.security.permissions import (
    PermissionCatalog,
)

from src.security.policies import (
    PolicySet,
)

from src.security.schema import (
    AuthenticationContext,
    AuthorizationRequest,
    AuthorizationResult,
    Permission,
    SecurityContext,
    SecurityPrincipal,
    SecretReference,
)

from src.security.secrets import (
    SecretManager,
    SecretValue,
)


# ---------------------------------------------------------------
# Security check result
# ---------------------------------------------------------------

@dataclass
class SecurityCheckResult:
    """
    Combined security context and authorization
    result for one protected OSHA operation.
    """

    context: SecurityContext

    authorization: AuthorizationResult

    @property
    def allowed(
        self,
    ) -> bool:
        return (
            self.authorization.
            allowed
        )

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "allowed": (
                self.allowed
            ),

            "context": (
                self.context.to_dict()
            ),

            "authorization": (
                self.authorization.
                to_dict()
            ),
        }


# ---------------------------------------------------------------
# Security pipeline
# ---------------------------------------------------------------

class SecurityPipeline:
    """
    Unified security facade for OSHA.

    Composes:

      identity registry
      authentication
      permissions
      policies
      authorization
      secret management
      security audit

    This layer decides whether an OSHA operation
    is permitted. It does not execute remediation
    or infrastructure-changing operations.
    """

    def __init__(
        self,
        *,
        config: Optional[
            SecurityConfig
        ] = None,
    ) -> None:
        self.config = (
            config
            or SecurityConfig()
        )

        self.config.validate()

        # -----------------------------------------------------
        # Identity
        # -----------------------------------------------------

        self.identities = (
            IdentityRegistry()
        )

        # -----------------------------------------------------
        # Permissions
        # -----------------------------------------------------

        self.permissions = (
            PermissionCatalog()
        )

        # -----------------------------------------------------
        # Policies
        # -----------------------------------------------------

        self.policies = (
            PolicySet()
        )

        # -----------------------------------------------------
        # Authentication
        # -----------------------------------------------------

        self.authentication = (
            AuthenticationService(
                config=(
                    self.config.
                    authentication
                ),
                redact_sensitive=(
                    self.config.
                    redact_sensitive
                ),
            )
        )

        # -----------------------------------------------------
        # Authorization
        # -----------------------------------------------------

        self.authorization = (
            AuthorizationService(
                permissions=(
                    self.permissions
                ),
                config=(
                    self.config.
                    authorization
                ),
                approval=(
                    self.config.approval
                ),
                policy_evaluator=(
                    self.policies.evaluate
                ),
            )
        )

        # -----------------------------------------------------
        # Secrets
        # -----------------------------------------------------

        self.secrets = (
            SecretManager(
                config=(
                    self.config.secrets
                )
            )
        )

        # -----------------------------------------------------
        # Audit
        # -----------------------------------------------------

        self.audit = (
            SecurityAuditLog(
                config=(
                    self.config.audit
                )
            )
        )

    # ---------------------------------------------------------
    # Authenticate
    # ---------------------------------------------------------

    def authenticate(
        self,
        *,
        method: str,
        credential: Any,
    ) -> AuthenticationContext:
        """
        Authenticate using a registered provider
        and automatically create a bounded
        security audit record.

        Credential material is never passed to
        the audit layer.
        """

        try:
            context = (
                self.authentication.
                authenticate(
                    method=method,
                    credential=credential,
                )
            )

            principal_id = (
                context.principal.
                principal_id
                if (
                    context.principal
                    is not None
                )
                else None
            )

            self.audit.authentication(
                success=True,
                principal_id=(
                    principal_id
                ),
                method=method,
            )

            return context

        except Exception:
            self.audit.authentication(
                success=False,
                principal_id=None,
                method=method,
            )

            raise

    # ---------------------------------------------------------
    # Authorize
    # ---------------------------------------------------------

    def authorize(
        self,
        *,
        authentication: AuthenticationContext,
        environment_id: str,
        permission: Permission,
        resource_id: Optional[
            str
        ] = None,
        mutating: bool = False,
        correlation_id: Optional[
            str
        ] = None,
    ) -> SecurityCheckResult:
        """
        Evaluate one protected OSHA operation.

        Authentication must already have produced
        a principal.

        A result of ALLOW still does not bypass
        downstream remediation guardrails.
        """

        authentication.finalize()

        if (
            authentication.principal
            is None
        ):
            raise ValueError(
                "Authorization requires "
                "an authenticated "
                "principal."
            )

        request = (
            AuthorizationRequest(
                principal=(
                    authentication.
                    principal
                ),
                permission=(
                    permission
                ),
                resource_id=(
                    resource_id
                ),
                attributes={
                    "mutating": (
                        mutating
                    ),
                    "correlation_id": (
                        correlation_id
                    ),
                },
            ).finalize()
        )

        result = (
            self.authorization.
            authorize(
                request,
                mutating=mutating,
            )
        )

        context = (
            SecurityContext(
                environment_id=(
                    environment_id
                ),
                authentication=(
                    authentication
                ),
                authorization=(
                    result
                ),
                correlation_id=(
                    correlation_id
                ),
                attributes={
                    "authorization_request_id": (
                        request.request_id
                    ),
                },
            ).finalize()
        )

        self.audit.authorization(
            decision=(
                result.decision
            ),

            principal_id=(
                authentication.
                principal.
                principal_id
            ),

            environment_id=(
                environment_id
            ),

            resource=(
                permission.resource
            ),

            action=(
                permission.action
            ),

            reference_ids=(
                [
                    result.decision_id
                ]
                if result.decision_id
                else []
            ),

            attributes={
                "mutating": (
                    mutating
                ),

                "request_id": (
                    request.request_id
                ),
            },
        )

        return SecurityCheckResult(
            context=context,
            authorization=result,
        )

    # ---------------------------------------------------------
    # Resolve secret
    # ---------------------------------------------------------

    def resolve_secret(
        self,
        reference: SecretReference,
        *,
        principal: Optional[
            SecurityPrincipal
        ] = None,
    ) -> SecretValue:
        """
        Resolve one secret reference and record
        only bounded reference-level audit
        information.

        The secret value itself is never sent to
        the audit system.
        """

        try:
            value = (
                self.secrets.resolve(
                    reference
                )
            )

            self.audit.secret_access(
                principal_id=(
                    principal.principal_id
                    if principal
                    else None
                ),

                provider=(
                    reference.provider
                ),

                reference_id=(
                    reference.reference_id
                ),

                success=True,
            )

            return value

        except Exception:
            self.audit.secret_access(
                principal_id=(
                    principal.principal_id
                    if principal
                    else None
                ),

                provider=(
                    reference.provider
                ),

                reference_id=(
                    reference.reference_id
                ),

                success=False,
            )

            raise

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        """
        Return non-secret operational metadata
        for the security subsystem.
        """

        return {
            "config": (
                safe_config_summary(
                    self.config
                )
            ),

            "identities": (
                self.identities.
                summary()
            ),

            "permissions": (
                self.permissions.
                summary()
            ),

            "policies": (
                self.policies.summary()
            ),

            "authentication": (
                self.authentication.
                summary()
            ),

            "authorization": (
                self.authorization.
                summary()
            ),

            "secrets": (
                self.secrets.summary()
            ),

            "audit": (
                self.audit.summary()
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "SecurityCheckResult",

    "SecurityPipeline",
]
