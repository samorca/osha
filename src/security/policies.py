from __future__ import annotations

"""
Deterministic policy matching and evaluation for
OSHA authorization.
"""

from threading import RLock

from typing import (
    Any,
    Dict,
    List,
    Optional,
)

from src.security.identity import (
    normalize_role,
)

from src.security.schema import (
    AuthorizationDecision,
    AuthorizationRequest,
    AuthorizationResult,
    PolicyEffect,
    SecurityPolicy,
)


# ---------------------------------------------------------------
# Wildcard
# ---------------------------------------------------------------

WILDCARD = "*"


# ---------------------------------------------------------------
# Errors
# ---------------------------------------------------------------


class PolicyErrorBase(RuntimeError):
    pass


class PolicyConflictError(PolicyErrorBase):
    pass


# ---------------------------------------------------------------
# Generic list/value matching
# ---------------------------------------------------------------


def _matches_value(
    allowed: List[str],
    requested: Optional[str],
) -> bool:
    """
    Policy-list matching semantics.

    Empty policy list:
        no restriction

    "*" in list:
        wildcard

    otherwise:
        exact value match
    """

    if not allowed:
        return True

    if WILDCARD in allowed:
        return True

    if requested is None:
        return False

    return requested in allowed


# ---------------------------------------------------------------
# Policy matching
# ---------------------------------------------------------------


def policy_matches(
    policy: SecurityPolicy,
    request: AuthorizationRequest,
) -> bool:
    """
    Determine whether a policy applies to an
    authorization request.

    Current policy dimensions:

      enabled
      resource
      action
      environment
      role

    Scope-specific policy rules can be added
    later once their semantics are explicitly
    defined.
    """

    policy.finalize()
    request.finalize()

    if not policy.enabled:
        return False

    # ---------------------------------------------------------
    # Resource
    # ---------------------------------------------------------

    if policy.resource not in {
        WILDCARD,
        request.permission.resource,
    }:
        return False

    # ---------------------------------------------------------
    # Action
    # ---------------------------------------------------------

    if policy.actions and WILDCARD not in policy.actions:
        if request.permission.action not in policy.actions:
            return False

    # ---------------------------------------------------------
    # Environment
    # ---------------------------------------------------------

    if not _matches_value(
        policy.environments,
        request.permission.environment_id,
    ):
        return False

    # ---------------------------------------------------------
    # Principal roles
    # ---------------------------------------------------------

    if policy.roles:
        principal_roles = {normalize_role(role) for role in request.principal.roles}

        required_roles = {normalize_role(role) for role in policy.roles}

        if WILDCARD not in required_roles and not principal_roles.intersection(
            required_roles
        ):
            return False

    return True


# ---------------------------------------------------------------
# Policy set
# ---------------------------------------------------------------


class PolicySet:
    """
    Thread-safe collection and deterministic
    evaluator for SecurityPolicy records.

    Policy conflict precedence:

        DENY
          ↓
        REQUIRE_APPROVAL
          ↓
        ALLOW

    This means an allow rule cannot override an
    applicable explicit deny rule.
    """

    def __init__(
        self,
    ) -> None:
        self._policies: Dict[
            str,
            SecurityPolicy,
        ] = {}

        self._lock = RLock()

    # ---------------------------------------------------------
    # Register
    # ---------------------------------------------------------

    def register(
        self,
        policy: SecurityPolicy,
    ) -> SecurityPolicy:
        policy.finalize()

        with self._lock:
            if policy.policy_id in self._policies:
                raise PolicyConflictError(
                    f"Policy already registered: {policy.policy_id}"
                )

            self._policies[policy.policy_id] = policy

        return policy

    # ---------------------------------------------------------
    # Upsert
    # ---------------------------------------------------------

    def upsert(
        self,
        policy: SecurityPolicy,
    ) -> SecurityPolicy:
        policy.finalize()

        with self._lock:
            self._policies[policy.policy_id] = policy

        return policy

    # ---------------------------------------------------------
    # Remove
    # ---------------------------------------------------------

    def remove(
        self,
        policy_id: str,
    ) -> Optional[SecurityPolicy]:
        with self._lock:
            return self._policies.pop(
                policy_id,
                None,
            )

    # ---------------------------------------------------------
    # List
    # ---------------------------------------------------------

    def list(
        self,
        *,
        enabled_only: bool = False,
    ) -> List[SecurityPolicy]:
        with self._lock:
            values = list(self._policies.values())

        if enabled_only:
            values = [value for value in values if value.enabled]

        # Lower priority numbers are evaluated
        # first for deterministic ordering.
        return sorted(
            values,
            key=lambda item: (
                item.priority,
                item.policy_id,
            ),
        )

    # ---------------------------------------------------------
    # Matching policies
    # ---------------------------------------------------------

    def matching(
        self,
        request: AuthorizationRequest,
    ) -> List[SecurityPolicy]:
        request.finalize()

        return [
            policy
            for policy in self.list(enabled_only=True)
            if policy_matches(
                policy,
                request,
            )
        ]

    # ---------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------

    def evaluate(
        self,
        request: AuthorizationRequest,
    ) -> Optional[AuthorizationResult]:
        """
        Evaluate all applicable policies.

        Returns None when no policy matches.

        Decision precedence:

            DENY
            REQUIRE_APPROVAL
            ALLOW
        """

        matches = self.matching(request)

        if not matches:
            return None

        denies = [
            policy for policy in matches if (policy.effect == PolicyEffect.DENY.value)
        ]

        approvals = [
            policy
            for policy in matches
            if (policy.effect == PolicyEffect.REQUIRE_APPROVAL.value)
        ]

        allows = [
            policy for policy in matches if (policy.effect == PolicyEffect.ALLOW.value)
        ]

        # -----------------------------------------------------
        # DENY wins
        # -----------------------------------------------------

        if denies:
            selected = denies

            decision = AuthorizationDecision.DENY.value

            reason = "A matching security policy denied the request."

        # -----------------------------------------------------
        # Approval requirement wins over allow
        # -----------------------------------------------------

        elif approvals:
            selected = approvals

            decision = AuthorizationDecision.REQUIRE_APPROVAL.value

            reason = "A matching security policy requires approval."

        # -----------------------------------------------------
        # Allow
        # -----------------------------------------------------

        elif allows:
            selected = allows

            decision = AuthorizationDecision.ALLOW.value

            reason = "A matching security policy allowed the request."

        else:
            return None

        # Keep all matching policies for
        # provenance, not merely the selected
        # effect category.
        policy_ids = [policy.policy_id for policy in matches]

        # For an approval decision, policy roles
        # describe possible approval-role
        # requirements. They do not constitute
        # actual approval.
        required_approval_roles = (
            sorted(
                {
                    normalize_role(role)
                    for policy in selected
                    for role in policy.roles
                    if role != WILDCARD
                }
            )
            if (decision == AuthorizationDecision.REQUIRE_APPROVAL.value)
            else []
        )

        return AuthorizationResult(
            decision=decision,
            request_id=(request.request_id),
            reason=reason,
            policy_ids=policy_ids,
            required_approval_roles=(required_approval_roles),
            attributes={
                "matching_policy_count": (len(matches)),
                "selected_effect": (
                    PolicyEffect.DENY.value
                    if denies
                    else (
                        PolicyEffect.REQUIRE_APPROVAL.value
                        if approvals
                        else (PolicyEffect.ALLOW.value)
                    )
                ),
            },
        ).finalize()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    def summary(
        self,
    ) -> Dict[str, Any]:
        with self._lock:
            effects: Dict[
                str,
                int,
            ] = {}

            enabled = 0

            for policy in self._policies.values():
                effects[policy.effect] = (
                    effects.get(
                        policy.effect,
                        0,
                    )
                    + 1
                )

                if policy.enabled:
                    enabled += 1

            return {
                "policy_count": (len(self._policies)),
                "enabled_policy_count": (enabled),
                "policies_by_effect": (effects),
            }


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

__all__ = [
    "WILDCARD",
    "PolicyErrorBase",
    "PolicyConflictError",
    "policy_matches",
    "PolicySet",
]
