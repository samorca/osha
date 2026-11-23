from __future__ import annotations

"""
Final authorization decision service for OSHA
security.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Optional,
)

from src.security.config import (
    ApprovalConfig,
    AuthorizationConfig,
)

from src.security.permissions import (
    PermissionCatalog,
)

from src.security.schema import (
    AuthorizationDecision,
    AuthorizationRequest,
    AuthorizationResult,
    PrincipalType,
)


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------

class AuthorizationErrorBase(
    RuntimeError
):
    pass


# ---------------------------------------------------------------
# Policy evaluator contract
# ---------------------------------------------------------------
#
# policies.py will eventually implement this
# contract.
#
# Returning None means the policy evaluator has
# no explicit result for the request.
#
# ---------------------------------------------------------------

PolicyEvaluator = Callable[
    [
        AuthorizationRequest
    ],
    Optional[
        AuthorizationResult
    ],
]


# ---------------------------------------------------------------
# Authorization service
# ---------------------------------------------------------------

class AuthorizationService:
    """
    Combine:

      principal
      requested permission
      direct permission grants
      policy evaluation
      approval requirements
      deny-by-default configuration

    into one AuthorizationResult.

    This layer authorizes OSHA operations. It
    does not perform the operation itself.
    """

    def __init__(
        self,
        *,
        permissions: Optional[
            PermissionCatalog
        ] = None,
        config: Optional[
            AuthorizationConfig
        ] = None,
        approval: Optional[
            ApprovalConfig
        ] = None,
        policy_evaluator: Optional[
            PolicyEvaluator
        ] = None,
    ) -> None:
        self.permissions = (
            permissions
            or PermissionCatalog()
        )

        self.config = (
            config
            or AuthorizationConfig()
        )

        self.approval = (
            approval
            or ApprovalConfig()
        )

        self.policy_evaluator = (
            policy_evaluator
        )

        self.config.validate()
        self.approval.validate()

    # ---------------------------------------------------------
    # Authorize
    # ---------------------------------------------------------

    def authorize(
        self,
        request: AuthorizationRequest,
        *,
        mutating: bool = False,
    ) -> AuthorizationResult:
        """
        Produce the final authorization decision
        for one request.

        Decision precedence:

          explicit DENY
              ↓
          REQUIRE_APPROVAL
              ↓
          configured mutation approval
              ↓
          explicit ALLOW / direct grant
              ↓
          deny-by-default
        """

        request.finalize()

        # -----------------------------------------------------
        # Authorization subsystem disabled
        # -----------------------------------------------------

        if not self.config.enabled:
            return (
                AuthorizationResult(
                    decision=(
                        AuthorizationDecision.
                        DENY.value
                    ),
                    request_id=(
                        request.request_id
                    ),
                    reason=(
                        "Authorization "
                        "services are "
                        "disabled."
                    ),
                ).finalize()
            )

        principal = (
            request.principal
        )

        # -----------------------------------------------------
        # System principal policy
        # -----------------------------------------------------

        if (
            principal.principal_type
            == PrincipalType.SYSTEM.value
            and not self.config.
            allow_system_principal
        ):
            return (
                AuthorizationResult(
                    decision=(
                        AuthorizationDecision.
                        DENY.value
                    ),
                    request_id=(
                        request.request_id
                    ),
                    reason=(
                        "System principals "
                        "are not allowed by "
                        "authorization "
                        "configuration."
                    ),
                ).finalize()
            )

        # -----------------------------------------------------
        # Direct permission grants
        # -----------------------------------------------------

        direct_grants = (
            self.permissions.
            matching_grants(
                principal.principal_id,
                request.permission,
            )
        )

        # -----------------------------------------------------
        # Policy evaluation
        # -----------------------------------------------------

        policy_result: Optional[
            AuthorizationResult
        ] = None

        if (
            self.policy_evaluator
            is not None
        ):
            policy_result = (
                self.policy_evaluator(
                    request
                )
            )

            if (
                policy_result
                is not None
            ):
                policy_result.finalize()

        # -----------------------------------------------------
        # Explicit policy denial wins
        # -----------------------------------------------------

        if (
            policy_result
            is not None
            and policy_result.decision
            == AuthorizationDecision.
            DENY.value
        ):
            return policy_result

        # -----------------------------------------------------
        # Explicit approval requirement
        # -----------------------------------------------------

        if (
            policy_result
            is not None
            and policy_result.decision
            == AuthorizationDecision.
            REQUIRE_APPROVAL.value
        ):
            return policy_result

        # -----------------------------------------------------
        # Mutation approval boundary
        # -----------------------------------------------------

        if (
            mutating
            and self.approval.enabled
            and self.approval.
            required_for_mutating_actions
        ):
            # Important:
            #
            # required_approval_roles identifies
            # which role classes may satisfy the
            # FUTURE approval workflow.
            #
            # Merely having one of these roles
            # does not count as approval.
            return (
                AuthorizationResult(
                    decision=(
                        AuthorizationDecision.
                        REQUIRE_APPROVAL.value
                    ),
                    request_id=(
                        request.request_id
                    ),
                    reason=(
                        "Mutating operation "
                        "requires approval."
                    ),
                    policy_ids=(
                        list(
                            policy_result.
                            policy_ids
                        )
                        if (
                            policy_result
                            is not None
                        )
                        else []
                    ),
                    required_approval_roles=(
                        list(
                            self.approval.
                            default_roles
                        )
                    ),
                    attributes={
                        "direct_grant_count": (
                            len(
                                direct_grants
                            )
                        ),
                    },
                ).finalize()
            )

        # -----------------------------------------------------
        # Determine explicit allow
        # -----------------------------------------------------

        explicit_policy_allow = bool(
            policy_result
            is not None
            and policy_result.decision
            == AuthorizationDecision.
            ALLOW.value
        )

        direct_allow = bool(
            direct_grants
        )

        if (
            self.config.
            require_explicit_allow
        ):
            allowed = (
                explicit_policy_allow
                or direct_allow
            )

        else:
            allowed = (
                explicit_policy_allow
                or direct_allow
                or not self.config.
                deny_by_default
            )

        # -----------------------------------------------------
        # Allow
        # -----------------------------------------------------

        if allowed:
            return (
                AuthorizationResult(
                    decision=(
                        AuthorizationDecision.
                        ALLOW.value
                    ),
                    request_id=(
                        request.request_id
                    ),
                    reason=(
                        "Authorization policy "
                        "allowed the request."
                        if (
                            explicit_policy_allow
                        )
                        else (
                            "A matching "
                            "permission grant "
                            "allowed the "
                            "request."
                        )
                    ),
                    policy_ids=(
                        list(
                            policy_result.
                            policy_ids
                        )
                        if (
                            policy_result
                            is not None
                        )
                        else []
                    ),
                    attributes={
                        "direct_grant_count": (
                            len(
                                direct_grants
                            )
                        ),
                        "explicit_policy_allow": (
                            explicit_policy_allow
                        ),
                    },
                ).finalize()
            )

        # -----------------------------------------------------
        # Default denial
        # -----------------------------------------------------

        if (
            self.config.
            deny_by_default
        ):
            reason = (
                "No explicit "
                "authorization allowed "
                "the request."
            )

        else:
            reason = (
                "Authorization request "
                "was not allowed."
            )

        return (
            AuthorizationResult(
                decision=(
                    AuthorizationDecision.
                    DENY.value
                ),
                request_id=(
                    request.request_id
                ),
                reason=reason,
                policy_ids=(
                    list(
                        policy_result.
                        policy_ids
                    )
                    if (
                        policy_result
                        is not None
                    )
                    else []
                ),
                attributes={
                    "direct_grant_count": (
                        len(
                            direct_grants
                        )
                    ),
                },
            ).finalize()
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        return {
            "enabled": (
                self.config.enabled
            ),

            "deny_by_default": (
                self.config.
                deny_by_default
            ),

            "require_explicit_allow": (
                self.config.
                require_explicit_allow
            ),

            "allow_system_principal": (
                self.config.
                allow_system_principal
            ),

            "approval_enabled": (
                self.approval.enabled
            ),

            "approval_required_for_mutating_actions": (
                self.approval.
                required_for_mutating_actions
            ),

            "permission_catalog": (
                self.permissions.summary()
            ),

            "policy_evaluator_registered": (
                self.policy_evaluator
                is not None
            ),
        }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "AuthorizationErrorBase",

    "PolicyEvaluator",

    "AuthorizationService",
]
