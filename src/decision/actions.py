from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
)

from src.decision.schema import (
    ActionCandidate,
    ActionCategory,
    DecisionContext,
    DecisionEvidence,
    Reversibility,
)


@dataclass(frozen=True)
class ActionDefinition:
    """
    One declarative OSHA action definition.

    This describes an action that may be proposed.
    It provides no execution capability.
    """

    action_type: str

    category: str

    title: str

    description: str

    reversibility: str

    required_privilege: str

    mutating: bool

    rollback_action_type: Optional[str] = None

    expected_effect: Optional[str] = None

    enabled: bool = True

    def validate(
        self,
    ) -> None:
        if not self.action_type.strip():
            raise ValueError("action_type is required")

        if not self.category.strip():
            raise ValueError("category is required")

        if not self.title.strip():
            raise ValueError("title is required")

        if (
            self.mutating
            and self.reversibility == Reversibility.FULL.value
            and not self.rollback_action_type
        ):
            raise ValueError(
                "Fully reversible mutating action requires rollback_action_type"
            )


# ---------------------------------------------------------------
# Default action catalog
# ---------------------------------------------------------------

DEFAULT_ACTION_DEFINITIONS: Dict[
    str,
    ActionDefinition,
] = {
    "observe_resource": ActionDefinition(
        action_type=("observe_resource"),
        category=(ActionCategory.OBSERVATION.value),
        title=("Observe affected resources"),
        description=("Continue collecting bounded health and state observations."),
        reversibility=(Reversibility.FULL.value),
        required_privilege="read",
        mutating=False,
        rollback_action_type=("stop_observation"),
        expected_effect=("Additional evidence without changing infrastructure state."),
    ),
    "collect_diagnostics": (
        ActionDefinition(
            action_type=("collect_diagnostics"),
            category=(ActionCategory.INVESTIGATION.value),
            title=("Collect diagnostics"),
            description=(
                "Collect bounded logs, state "
                "and health diagnostics for "
                "affected resources."
            ),
            reversibility=(Reversibility.FULL.value),
            required_privilege="read",
            mutating=False,
            rollback_action_type=("discard_diagnostic_session"),
            expected_effect=("More evidence for incident and root-cause validation."),
        )
    ),
    "notify_operator": (
        ActionDefinition(
            action_type=("notify_operator"),
            category=(ActionCategory.NOTIFICATION.value),
            title="Notify operator",
            description=("Prepare an operator-facing incident notification."),
            reversibility=(Reversibility.FULL.value),
            required_privilege="none",
            mutating=False,
            rollback_action_type=("withdraw_notification"),
            expected_effect=("Human awareness and review of the incident candidate."),
        )
    ),
    "restart_service": (
        ActionDefinition(
            action_type=("restart_service"),
            category=(ActionCategory.SERVICE_CONTROL.value),
            title=("Restart affected service"),
            description=(
                "Propose a controlled restart of one explicitly targeted service."
            ),
            reversibility=(Reversibility.FULL.value),
            required_privilege=("operator"),
            mutating=True,
            rollback_action_type=("restore_service_state"),
            expected_effect=("Restore service operation if the failure is transient."),
        )
    ),
    "reload_service": (
        ActionDefinition(
            action_type=("reload_service"),
            category=(ActionCategory.SERVICE_CONTROL.value),
            title=("Reload affected service"),
            description=("Propose a controlled service configuration reload."),
            reversibility=(Reversibility.FULL.value),
            required_privilege=("operator"),
            mutating=True,
            rollback_action_type=("restore_previous_service_configuration"),
            expected_effect=(
                "Apply validated configuration without a full service restart."
            ),
        )
    ),
    "rollback_configuration": (
        ActionDefinition(
            action_type=("rollback_configuration"),
            category=(ActionCategory.CONFIGURATION.value),
            title=("Roll back configuration"),
            description=(
                "Propose restoring a previously verified configuration version."
            ),
            reversibility=(Reversibility.FULL.value),
            required_privilege=("administrator"),
            mutating=True,
            rollback_action_type=("restore_replaced_configuration"),
            expected_effect=("Return the resource to a known configuration state."),
        )
    ),
    "isolate_resource": (
        ActionDefinition(
            action_type=("isolate_resource"),
            category=(ActionCategory.SECURITY_CONTROL.value),
            title=("Isolate affected resource"),
            description=(
                "Propose temporary isolation of one explicitly targeted resource."
            ),
            reversibility=(Reversibility.FULL.value),
            required_privilege=("administrator"),
            mutating=True,
            rollback_action_type=("remove_resource_isolation"),
            expected_effect=("Limit propagation while preserving a rollback path."),
        )
    ),
    "scale_capacity": (
        ActionDefinition(
            action_type=("scale_capacity"),
            category=(ActionCategory.CAPACITY.value),
            title="Adjust capacity",
            description=(
                "Propose a bounded capacity "
                "adjustment for an explicitly "
                "targeted service."
            ),
            reversibility=(Reversibility.FULL.value),
            required_privilege=("operator"),
            mutating=True,
            rollback_action_type=("restore_previous_capacity"),
            expected_effect=("Reduce verified capacity pressure."),
        )
    ),
}


# ---------------------------------------------------------------
# Catalog lookup
# ---------------------------------------------------------------


def action_definition(
    action_type: str,
    *,
    definitions: Optional[
        Dict[
            str,
            ActionDefinition,
        ]
    ] = None,
) -> ActionDefinition:
    catalog = definitions or DEFAULT_ACTION_DEFINITIONS

    definition = catalog.get(action_type)

    if definition is None:
        raise KeyError(f"Unknown action type: {action_type}")

    definition.validate()

    if not definition.enabled:
        raise ValueError(f"Action type is disabled: {action_type}")

    return definition


# ---------------------------------------------------------------
# Candidate creation
# ---------------------------------------------------------------


def create_action(
    action_type: str,
    context: DecisionContext,
    *,
    target_resource_ids: Iterable[str] = (),
    target_service_ids: Iterable[str] = (),
    parameters: Optional[Dict[str, Any]] = None,
    rationale: Optional[str] = None,
    definitions: Optional[
        Dict[
            str,
            ActionDefinition,
        ]
    ] = None,
) -> ActionCandidate:
    """
    Create one ActionCandidate from a catalog
    definition.

    The resulting object is a proposal only.
    """

    definition = action_definition(
        action_type,
        definitions=definitions,
    )

    context.finalize()

    resources = sorted(set(target_resource_ids))

    services = sorted(set(target_service_ids))

    evidence = [
        DecisionEvidence(
            evidence_type=("action_generation"),
            description=(
                "Action candidate was generated from structured decision context."
            ),
            score=(context.incident_confidence),
            incident_id=(context.incident_id),
            resource_ids=resources,
            attributes={
                "context_id": (context.context_id),
                "action_type": (action_type),
                "mutating": (definition.mutating),
            },
        )
    ]

    return ActionCandidate(
        action_type=(definition.action_type),
        category=(definition.category),
        title=(definition.title),
        environment_id=(context.environment_id),
        target_resource_ids=(resources),
        target_service_ids=(services),
        description=(definition.description),
        rationale=rationale,
        parameters=dict(parameters or {}),
        reversibility=(definition.reversibility),
        rollback_action_type=(definition.rollback_action_type),
        expected_effect=(definition.expected_effect),
        evidence=evidence,
        attributes={
            "required_privilege": (definition.required_privilege),
            "mutating": (definition.mutating),
            "context_id": (context.context_id),
        },
    ).finalize()


# ---------------------------------------------------------------
# Default action generation
# ---------------------------------------------------------------


def default_actions_for_context(
    context: DecisionContext,
    *,
    include_mutating_candidates: bool = True,
) -> List[ActionCandidate]:
    """
    Generate conservative default candidates.

    Read-only candidates are always generated.

    A service restart candidate is generated only
    when both a root-cause resource and a service
    target exist. Generation still does not make
    it eligible, approved or executable.
    """

    context.finalize()

    actions = [
        create_action(
            "observe_resource",
            context,
            target_resource_ids=(context.resource_ids),
            target_service_ids=(context.service_ids),
            rationale=(
                "Continue observing the affected scope while evidence is evaluated."
            ),
        ),
        create_action(
            "collect_diagnostics",
            context,
            target_resource_ids=(context.resource_ids),
            target_service_ids=(context.service_ids),
            rationale=(
                "Collect evidence needed to "
                "validate incident and "
                "root-cause hypotheses."
            ),
        ),
        create_action(
            "notify_operator",
            context,
            target_resource_ids=(context.resource_ids),
            target_service_ids=(context.service_ids),
            rationale=(
                "Present the incident "
                "candidate, limitations and "
                "proposed next steps to an "
                "operator."
            ),
        ),
    ]

    if (
        include_mutating_candidates
        and context.root_cause_resource_id
        and context.service_ids
    ):
        actions.append(
            create_action(
                "restart_service",
                context,
                target_resource_ids=[context.root_cause_resource_id],
                target_service_ids=(context.service_ids[:1]),
                parameters={
                    "maximum_targets": 1,
                },
                rationale=(
                    "Candidate only: consider "
                    "a controlled service "
                    "restart after policy, "
                    "risk, eligibility and "
                    "approval checks."
                ),
            )
        )

    return deduplicate_actions(actions)


# ---------------------------------------------------------------
# Deduplication and summaries
# ---------------------------------------------------------------


def deduplicate_actions(
    actions: Iterable[ActionCandidate],
) -> List[ActionCandidate]:
    output: Dict[
        str,
        ActionCandidate,
    ] = {}

    for action in actions:
        action.finalize()

        if action.action_id:
            output[action.action_id] = action

    return sorted(
        output.values(),
        key=lambda item: (
            item.category,
            item.action_type,
            item.action_id or "",
        ),
    )


def action_catalog_summary(
    definitions: Optional[
        Dict[
            str,
            ActionDefinition,
        ]
    ] = None,
) -> Dict[str, Any]:
    catalog = definitions or DEFAULT_ACTION_DEFINITIONS

    enabled = [item for item in catalog.values() if item.enabled]

    categories: Dict[
        str,
        int,
    ] = {}

    for item in enabled:
        categories[item.category] = (
            categories.get(
                item.category,
                0,
            )
            + 1
        )

    return {
        "action_type_count": len(catalog),
        "enabled_action_type_count": (len(enabled)),
        "mutating_action_type_count": (sum(1 for item in enabled if item.mutating)),
        "read_only_action_type_count": (
            sum(1 for item in enabled if not item.mutating)
        ),
        "categories": categories,
    }


# ---------------------------------------------------------------
# Catalog class
# ---------------------------------------------------------------


class ActionCatalog:
    """
    Reusable declarative action catalog.

    The catalog cannot execute actions.
    """

    def __init__(
        self,
        *,
        definitions: Optional[Iterable[ActionDefinition]] = None,
    ) -> None:
        source = definitions or DEFAULT_ACTION_DEFINITIONS.values()

        self._definitions: Dict[
            str,
            ActionDefinition,
        ] = {}

        for definition in source:
            definition.validate()

            if definition.action_type in self._definitions:
                raise ValueError(f"Duplicate action type: {definition.action_type}")

            self._definitions[definition.action_type] = definition

    def get(
        self,
        action_type: str,
    ) -> ActionDefinition:
        return action_definition(
            action_type,
            definitions=(self._definitions),
        )

    def create(
        self,
        action_type: str,
        context: DecisionContext,
        *,
        target_resource_ids: Iterable[str] = (),
        target_service_ids: Iterable[str] = (),
        parameters: Optional[Dict[str, Any]] = None,
        rationale: Optional[str] = None,
    ) -> ActionCandidate:
        return create_action(
            action_type,
            context,
            target_resource_ids=(target_resource_ids),
            target_service_ids=(target_service_ids),
            parameters=parameters,
            rationale=rationale,
            definitions=(self._definitions),
        )

    def defaults(
        self,
        context: DecisionContext,
        *,
        include_mutating_candidates: bool = True,
    ) -> List[ActionCandidate]:
        """
        Generate default candidates using this
        catalog's definitions.
        """

        context.finalize()

        actions = [
            self.create(
                "observe_resource",
                context,
                target_resource_ids=(context.resource_ids),
                target_service_ids=(context.service_ids),
                rationale=(
                    "Continue observing the affected scope while evidence is evaluated."
                ),
            ),
            self.create(
                "collect_diagnostics",
                context,
                target_resource_ids=(context.resource_ids),
                target_service_ids=(context.service_ids),
                rationale=(
                    "Collect evidence needed "
                    "to validate incident and "
                    "root-cause hypotheses."
                ),
            ),
            self.create(
                "notify_operator",
                context,
                target_resource_ids=(context.resource_ids),
                target_service_ids=(context.service_ids),
                rationale=("Present the incident candidate to an operator."),
            ),
        ]

        if (
            include_mutating_candidates
            and context.root_cause_resource_id
            and context.service_ids
        ):
            actions.append(
                self.create(
                    "restart_service",
                    context,
                    target_resource_ids=[context.root_cause_resource_id],
                    target_service_ids=(context.service_ids[:1]),
                    parameters={
                        "maximum_targets": 1,
                    },
                    rationale=(
                        "Candidate only: "
                        "controlled service "
                        "restart subject to all "
                        "decision gates."
                    ),
                )
            )

        return deduplicate_actions(actions)

    def definitions(
        self,
    ) -> List[ActionDefinition]:
        return sorted(
            self._definitions.values(),
            key=lambda item: item.action_type,
        )
