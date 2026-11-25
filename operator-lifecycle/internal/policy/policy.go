package policy

import (
	"errors"
	"fmt"
	"strings"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
)

var (
	ErrPlanNil               = errors.New("install plan is nil")
	ErrPlanNotApproved       = errors.New("install plan requires approval")
	ErrActionDenied          = errors.New("install plan action denied")
	ErrDestructiveStepDenied = errors.New("destructive step denied")
	ErrDeleteDenied          = errors.New("resource deletion denied")
	ErrNamespaceDenied       = errors.New("target namespace denied")
	ErrClusterScopedDenied   = errors.New("cluster-scoped resource denied")
	ErrTooManySteps          = errors.New("install plan exceeds step limit")
	ErrInvalidStep           = errors.New("invalid install plan step")
	ErrDuplicateStep         = errors.New("duplicate install plan step")
)

type Decision string

const (
	DecisionAllow           Decision = "Allow"
	DecisionDeny            Decision = "Deny"
	DecisionRequireApproval Decision = "RequireApproval"
)

type Config struct {
	AllowedNamespaces map[string]struct{}

	AllowInstall  bool
	AllowUpgrade  bool
	AllowRollback bool

	AllowDelete           bool
	AllowDestructiveSteps bool
	AllowClusterScoped    bool

	MaxSteps int
}

type Evaluation struct {
	Decision Decision
	Reason   string
	Message  string
}

type Engine struct {
	config Config
}

func DefaultConfig() Config {
	return Config{
		AllowedNamespaces:     map[string]struct{}{},
		AllowInstall:          true,
		AllowUpgrade:          true,
		AllowRollback:         false,
		AllowDelete:           false,
		AllowDestructiveSteps: false,
		AllowClusterScoped:    false,
		MaxSteps:              100,
	}
}

func New(config Config) *Engine {
	if config.MaxSteps <= 0 {
		config.MaxSteps = 100
	}

	if config.AllowedNamespaces == nil {
		config.AllowedNamespaces = map[string]struct{}{}
	}

	return &Engine{
		config: config,
	}
}

func (e *Engine) Evaluate(
	plan *lifecyclev1alpha1.OperatorInstallPlan,
) Evaluation {
	err := e.Validate(plan)

	switch {
	case err == nil:
		return Evaluation{
			Decision: DecisionAllow,
			Reason:   "PolicyAllowed",
			Message:  "install plan satisfies lifecycle policy",
		}

	case errors.Is(err, ErrPlanNotApproved):
		return Evaluation{
			Decision: DecisionRequireApproval,
			Reason:   "ApprovalRequired",
			Message:  err.Error(),
		}

	default:
		return Evaluation{
			Decision: DecisionDeny,
			Reason:   "PolicyDenied",
			Message:  err.Error(),
		}
	}
}

func (e *Engine) Validate(
	plan *lifecyclev1alpha1.OperatorInstallPlan,
) error {
	if plan == nil {
		return ErrPlanNil
	}

	if err := e.validateAction(plan.Spec.Action); err != nil {
		return err
	}

	if strings.TrimSpace(plan.Spec.Package) == "" {
		return fmt.Errorf(
			"%w: package is required",
			ErrInvalidStep,
		)
	}

	if strings.TrimSpace(plan.Spec.TargetVersion) == "" {
		return fmt.Errorf(
			"%w: target version is required",
			ErrInvalidStep,
		)
	}

	if strings.TrimSpace(plan.Spec.ResolutionHash) == "" {
		return fmt.Errorf(
			"%w: resolution hash is required",
			ErrInvalidStep,
		)
	}

	if err := e.validateNamespace(
		plan.Spec.TargetNamespace,
	); err != nil {
		return err
	}

	if len(plan.Spec.Steps) == 0 {
		return fmt.Errorf(
			"%w: plan contains no execution steps",
			ErrInvalidStep,
		)
	}

	if len(plan.Spec.Steps) > e.config.MaxSteps {
		return fmt.Errorf(
			"%w: got=%d maximum=%d",
			ErrTooManySteps,
			len(plan.Spec.Steps),
			e.config.MaxSteps,
		)
	}

	if plan.Spec.Approval == lifecyclev1alpha1.ApprovalModeManual &&
		!plan.Spec.Approved {

		return ErrPlanNotApproved
	}

	stepIDs := make(
		map[string]struct{},
		len(plan.Spec.Steps),
	)

	for _, step := range plan.Spec.Steps {
		if err := e.validateStep(
			step,
			plan.Spec.TargetNamespace,
		); err != nil {
			return fmt.Errorf(
				"step %q: %w",
				step.ID,
				err,
			)
		}

		if _, exists := stepIDs[step.ID]; exists {
			return fmt.Errorf(
				"%w: %q",
				ErrDuplicateStep,
				step.ID,
			)
		}

		stepIDs[step.ID] = struct{}{}
	}

	for _, step := range plan.Spec.Steps {
		for _, dependencyID := range step.DependsOn {
			if dependencyID == step.ID {
				return fmt.Errorf(
					"%w: step %q depends on itself",
					ErrInvalidStep,
					step.ID,
				)
			}

			if _, exists := stepIDs[dependencyID]; !exists {
				return fmt.Errorf(
					"%w: step %q depends on unknown step %q",
					ErrInvalidStep,
					step.ID,
					dependencyID,
				)
			}
		}
	}

	return nil
}

func (e *Engine) validateAction(
	action lifecyclev1alpha1.InstallPlanAction,
) error {
	switch action {
	case lifecyclev1alpha1.InstallPlanActionInstall:
		if !e.config.AllowInstall {
			return fmt.Errorf(
				"%w: install disabled by policy",
				ErrActionDenied,
			)
		}

	case lifecyclev1alpha1.InstallPlanActionUpgrade:
		if !e.config.AllowUpgrade {
			return fmt.Errorf(
				"%w: upgrade disabled by policy",
				ErrActionDenied,
			)
		}

	case lifecyclev1alpha1.InstallPlanActionRollback:
		if !e.config.AllowRollback {
			return fmt.Errorf(
				"%w: rollback disabled by policy",
				ErrActionDenied,
			)
		}

	default:
		return fmt.Errorf(
			"%w: unsupported action %q",
			ErrActionDenied,
			action,
		)
	}

	return nil
}

func (e *Engine) validateStep(
	step lifecyclev1alpha1.InstallPlanStep,
	targetNamespace string,
) error {
	if strings.TrimSpace(step.ID) == "" {
		return fmt.Errorf(
			"%w: step ID is required",
			ErrInvalidStep,
		)
	}

	if strings.TrimSpace(step.Resource.Version) == "" ||
		strings.TrimSpace(step.Resource.Kind) == "" ||
		strings.TrimSpace(step.Resource.Name) == "" {

		return fmt.Errorf(
			"%w: resource version, kind and name are required",
			ErrInvalidStep,
		)
	}

	if step.Destructive &&
		!e.config.AllowDestructiveSteps {

		return ErrDestructiveStepDenied
	}

	if step.Action == lifecyclev1alpha1.InstallStepDelete &&
		!e.config.AllowDelete {

		return ErrDeleteDenied
	}

	switch step.Action {
	case lifecyclev1alpha1.InstallStepCreate,
		lifecyclev1alpha1.InstallStepUpdate:

		if strings.TrimSpace(step.Manifest) == "" {
			return fmt.Errorf(
				"%w: action %s requires a manifest",
				ErrInvalidStep,
				step.Action,
			)
		}

	case lifecyclev1alpha1.InstallStepPatch:
		if strings.TrimSpace(step.Manifest) == "" {
			return fmt.Errorf(
				"%w: patch requires manifest data",
				ErrInvalidStep,
			)
		}

	case lifecyclev1alpha1.InstallStepDelete:
		// Delete requires an explicit resource reference but no manifest.

	default:
		return fmt.Errorf(
			"%w: unsupported step action %q",
			ErrInvalidStep,
			step.Action,
		)
	}

	resourceNamespace := strings.TrimSpace(
		step.Resource.Namespace,
	)

	if resourceNamespace == "" {
		if !e.config.AllowClusterScoped {
			return fmt.Errorf(
				"%w: %s %s",
				ErrClusterScopedDenied,
				step.Resource.Kind,
				step.Resource.Name,
			)
		}

		return nil
	}

	if targetNamespace != "" &&
		resourceNamespace != targetNamespace {

		return fmt.Errorf(
			"%w: resource namespace %q differs from target namespace %q",
			ErrNamespaceDenied,
			resourceNamespace,
			targetNamespace,
		)
	}

	return e.validateNamespace(resourceNamespace)
}

func (e *Engine) validateNamespace(namespace string) error {
	namespace = strings.TrimSpace(namespace)

	if namespace == "" {
		return fmt.Errorf(
			"%w: namespace cannot be empty",
			ErrNamespaceDenied,
		)
	}

	if len(e.config.AllowedNamespaces) == 0 {
		return nil
	}

	if _, allowed := e.config.AllowedNamespaces[namespace]; !allowed {
		return fmt.Errorf(
			"%w: %q",
			ErrNamespaceDenied,
			namespace,
		)
	}

	return nil
}
