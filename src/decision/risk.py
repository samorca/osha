from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
)

from src.decision.schema import (
    ActionCandidate,
    ActionCategory,
    DecisionContext,
    DecisionEvidence,
    Reversibility,
    RiskAssessment,
    RiskLevel,
    checked_score,
)


CATEGORY_BASE_RISK: Dict[
    str,
    float,
] = {
    ActionCategory.OBSERVATION.value: 0.02,
    ActionCategory.INVESTIGATION.value: 0.08,
    ActionCategory.NOTIFICATION.value: 0.05,
    ActionCategory.CONFIGURATION.value: 0.50,
    ActionCategory.SERVICE_CONTROL.value: 0.55,
    ActionCategory.RESOURCE_CONTROL.value: 0.75,
    ActionCategory.NETWORK_CONTROL.value: 0.75,
    ActionCategory.SECURITY_CONTROL.value: 0.80,
    ActionCategory.CAPACITY.value: 0.60,
    ActionCategory.RECOVERY.value: 0.65,
    ActionCategory.OTHER.value: 0.90,
}


PRIVILEGE_RISK: Dict[
    str,
    float,
] = {
    "none": 0.00,
    "read": 0.05,
    "user": 0.20,
    "service": 0.40,
    "operator": 0.55,
    "administrator": 0.75,
    "root": 0.90,
    "cluster_admin": 0.95,
    "unknown": 0.70,
}


REVERSIBILITY_RISK: Dict[
    str,
    float,
] = {
    Reversibility.FULL.value: 0.10,
    Reversibility.PARTIAL.value: 0.55,
    Reversibility.NONE.value: 1.00,
    Reversibility.UNKNOWN.value: 0.80,
}


@dataclass(frozen=True)
class RiskConfig:
    """
    OSHA action-risk configuration.

    These values are heuristic policy defaults,
    not calibrated probabilities.
    """

    category_weight: float = 0.20

    blast_radius_weight: float = 0.20

    irreversibility_weight: float = 0.20

    service_impact_weight: float = 0.15

    uncertainty_weight: float = 0.15

    privilege_weight: float = 0.10

    low_threshold: float = 0.20

    medium_threshold: float = 0.45

    high_threshold: float = 0.70

    critical_threshold: float = 0.90

    maximum_resources_for_low_risk: int = 1

    maximum_services_for_low_risk: int = 1

    def validate(
        self,
    ) -> None:
        weights = (
            self.category_weight,
            self.blast_radius_weight,
            self.irreversibility_weight,
            self.service_impact_weight,
            self.uncertainty_weight,
            self.privilege_weight,
        )

        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("Risk weights must be non-negative and total > 0")

        thresholds = (
            self.low_threshold,
            self.medium_threshold,
            self.high_threshold,
            self.critical_threshold,
        )

        if any(value < 0 or value > 1 for value in thresholds):
            raise ValueError("Risk thresholds must be between 0 and 1")

        if not (
            self.low_threshold
            <= self.medium_threshold
            <= self.high_threshold
            <= self.critical_threshold
        ):
            raise ValueError("Risk thresholds must be ordered")

        if self.maximum_resources_for_low_risk < 1:
            raise ValueError("maximum_resources_for_low_risk must be >= 1")

        if self.maximum_services_for_low_risk < 1:
            raise ValueError("maximum_services_for_low_risk must be >= 1")


# ---------------------------------------------------------------
# Individual risk dimensions
# ---------------------------------------------------------------


def category_risk(
    action: ActionCandidate,
) -> float:
    """
    Base risk associated with the action category.

    Unknown categories receive a conservative
    high score.
    """

    return checked_score(
        CATEGORY_BASE_RISK.get(
            action.category,
            0.90,
        )
    )


def scale_count(
    count: int,
    boundaries: tuple[
        int,
        int,
        int,
    ],
) -> float:
    """
    Convert a target count into bounded scope
    risk.
    """

    if count <= 0:
        return 0.0

    low, medium, high = boundaries

    if count <= low:
        return 0.15

    if count <= medium:
        return 0.40

    if count <= high:
        return 0.70

    return 1.0


def blast_radius_risk(
    action: ActionCandidate,
    context: DecisionContext,
) -> float:
    """
    Estimate the action's possible operational
    scope.

    This is based on the proposed targets and
    current incident impact context.
    """

    target_resources = set(action.target_resource_ids)

    target_services = set(action.target_service_ids)

    resource_component = scale_count(
        len(target_resources),
        (
            1,
            3,
            10,
        ),
    )

    service_component = scale_count(
        len(target_services),
        (
            1,
            2,
            5,
        ),
    )

    impact_component = checked_score(context.impact_score)

    return checked_score(
        (resource_component * 0.45)
        + (service_component * 0.25)
        + (impact_component * 0.30)
    )


def irreversibility_risk(
    action: ActionCandidate,
) -> float:
    """
    Calculate risk introduced by difficulty of
    undoing the action.
    """

    score = REVERSIBILITY_RISK.get(
        action.reversibility,
        0.80,
    )

    if (
        action.reversibility == Reversibility.FULL.value
        and not action.rollback_action_type
    ):
        # An action claiming full reversibility
        # without an explicit rollback type gets
        # a small uncertainty increase.
        score = max(
            score,
            0.25,
        )

    return checked_score(score)


def service_impact_risk(
    action: ActionCandidate,
    context: DecisionContext,
) -> float:
    """
    Estimate risk to operational services.

    Known user impact increases caution. Unknown
    impact still carries non-zero uncertainty.
    """

    service_count = len(set(action.target_service_ids))

    spread = scale_count(
        service_count,
        (
            1,
            2,
            5,
        ),
    )

    known_user_impact = 0.85 if context.user_impact_known else 0.25

    return checked_score((spread * 0.60) + (known_user_impact * 0.40))


def uncertainty_risk(
    context: DecisionContext,
) -> float:
    """
    Convert evidence gaps into action risk.

    Low intelligence confidence and an
    unverified root cause should increase the
    risk of mutating infrastructure.
    """

    confidence_uncertainty = 1.0 - checked_score(context.incident_confidence)

    root_cause_uncertainty = 1.0 - checked_score(context.root_cause_score)

    verification_uncertainty = 0.0 if context.root_cause_verified else 0.50

    limitation_component = min(
        1.0,
        len(context.limitations) / 5.0,
    )

    return checked_score(
        (confidence_uncertainty * 0.35)
        + (root_cause_uncertainty * 0.30)
        + (verification_uncertainty * 0.20)
        + (limitation_component * 0.15)
    )


def privilege_risk(
    action: ActionCandidate,
) -> float:
    """
    Score the privilege required by the proposed
    action.

    actions.py should place the required
    privilege under:

        action.attributes[
            "required_privilege"
        ]
    """

    raw = (
        str(
            action.attributes.get(
                "required_privilege",
                "unknown",
            )
        )
        .strip()
        .lower()
    )

    return checked_score(
        PRIVILEGE_RISK.get(
            raw,
            PRIVILEGE_RISK["unknown"],
        )
    )


# ---------------------------------------------------------------
# Risk classification
# ---------------------------------------------------------------


def risk_level(
    score: float,
    *,
    config: RiskConfig,
) -> str:
    value = checked_score(score)

    if value >= config.critical_threshold:
        return RiskLevel.CRITICAL.value

    if value >= config.high_threshold:
        return RiskLevel.HIGH.value

    if value >= config.medium_threshold:
        return RiskLevel.MEDIUM.value

    if value >= config.low_threshold:
        return RiskLevel.LOW.value

    return RiskLevel.NONE.value


def risk_reasons(
    action: ActionCandidate,
    context: DecisionContext,
    dimensions: Mapping[
        str,
        float,
    ],
) -> List[str]:
    reasons: List[str] = []

    if dimensions["category"] >= 0.70:
        reasons.append(
            "The action category can materially change infrastructure state."
        )

    if dimensions["blast_radius"] >= 0.60:
        reasons.append("The proposed action has a broad target or impact scope.")

    if dimensions["irreversibility"] >= 0.60:
        reasons.append(
            "The action is not fully reversible or has no explicit rollback action."
        )

    if dimensions["service_impact"] >= 0.60:
        reasons.append("The action may affect one or more operational services.")

    if dimensions["uncertainty"] >= 0.60:
        reasons.append(
            "Incident or root-cause evidence contains substantial uncertainty."
        )

    if dimensions["privilege"] >= 0.70:
        reasons.append("The action requires elevated infrastructure privileges.")

    if not reasons:
        reasons.append(
            "The proposed action has limited scope and low measured risk dimensions."
        )

    return reasons


# ---------------------------------------------------------------
# Complete assessment
# ---------------------------------------------------------------


def assess_action_risk(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: Optional[RiskConfig] = None,
) -> RiskAssessment:
    """
    Assess the operational risk of one action.

    This function does not determine eligibility,
    grant approval or execute the action.
    """

    configuration = config or RiskConfig()

    configuration.validate()

    action.finalize()

    context.finalize()

    dimensions = {
        "category": category_risk(action),
        "blast_radius": (
            blast_radius_risk(
                action,
                context,
            )
        ),
        "irreversibility": (irreversibility_risk(action)),
        "service_impact": (
            service_impact_risk(
                action,
                context,
            )
        ),
        "uncertainty": (uncertainty_risk(context)),
        "privilege": (privilege_risk(action)),
    }

    weights = {
        "category": (configuration.category_weight),
        "blast_radius": (configuration.blast_radius_weight),
        "irreversibility": (configuration.irreversibility_weight),
        "service_impact": (configuration.service_impact_weight),
        "uncertainty": (configuration.uncertainty_weight),
        "privilege": (configuration.privilege_weight),
    }

    total_weight = sum(weights.values())

    overall = checked_score(
        sum(dimensions[name] * weights[name] for name in weights) / total_weight
    )

    level = risk_level(
        overall,
        config=configuration,
    )

    reasons = risk_reasons(
        action,
        context,
        dimensions,
    )

    evidence = [
        DecisionEvidence(
            evidence_type=("action_risk"),
            description=(
                "Risk was calculated from "
                "action scope, reversibility, "
                "service impact, uncertainty, "
                "privilege and category."
            ),
            score=overall,
            incident_id=(context.incident_id),
            resource_ids=list(action.target_resource_ids),
            attributes={
                "dimensions": (dimensions),
                "weights": weights,
            },
        )
    ]

    return RiskAssessment(
        level=level,
        score=overall,
        blast_radius_score=(dimensions["blast_radius"]),
        irreversibility_score=(dimensions["irreversibility"]),
        service_impact_score=(dimensions["service_impact"]),
        uncertainty_score=(dimensions["uncertainty"]),
        privilege_score=(dimensions["privilege"]),
        reasons=reasons,
        evidence=evidence,
        attributes={
            "category_score": (dimensions["category"]),
            "dimension_scores": (dimensions),
            "weights": weights,
            "target_resource_count": (len(set(action.target_resource_ids))),
            "target_service_count": (len(set(action.target_service_ids))),
        },
    ).finalize()


def apply_risk(
    action: ActionCandidate,
    context: DecisionContext,
    *,
    config: Optional[RiskConfig] = None,
) -> ActionCandidate:
    """
    Attach a RiskAssessment to one action.
    """

    action.risk = assess_action_risk(
        action,
        context,
        config=config,
    )

    action.evidence.extend(action.risk.evidence)

    return action.finalize()


def assess_actions(
    actions: Iterable[ActionCandidate],
    context: DecisionContext,
    *,
    config: Optional[RiskConfig] = None,
) -> List[ActionCandidate]:
    return [
        apply_risk(
            action,
            context,
            config=config,
        )
        for action in actions
    ]


# ---------------------------------------------------------------
# Summary
# ---------------------------------------------------------------


def risk_summary(
    actions: Iterable[ActionCandidate],
) -> Dict[str, Any]:
    action_list = list(actions)

    levels: Dict[
        str,
        int,
    ] = {}

    for action in action_list:
        levels[action.risk.level] = (
            levels.get(
                action.risk.level,
                0,
            )
            + 1
        )

    scores = [action.risk.score for action in action_list]

    return {
        "action_count": (len(action_list)),
        "risk_levels": levels,
        "average_risk": (checked_score(sum(scores) / len(scores)) if scores else 0.0),
        "highest_risk": max(
            scores,
            default=0.0,
        ),
    }


class RiskEngine:
    """
    Reusable action-risk evaluator.

    This class has no infrastructure execution
    capabilities.
    """

    def __init__(
        self,
        *,
        config: Optional[RiskConfig] = None,
    ) -> None:
        self.config = config or RiskConfig()

        self.config.validate()

    def assess(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> RiskAssessment:
        return assess_action_risk(
            action,
            context,
            config=self.config,
        )

    def apply(
        self,
        action: ActionCandidate,
        context: DecisionContext,
    ) -> ActionCandidate:
        return apply_risk(
            action,
            context,
            config=self.config,
        )

    def apply_many(
        self,
        actions: Iterable[ActionCandidate],
        context: DecisionContext,
    ) -> List[ActionCandidate]:
        return assess_actions(
            actions,
            context,
            config=self.config,
        )
