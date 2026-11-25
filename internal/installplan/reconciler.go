package installplan

import (
	"context"
	"errors"
	"time"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/installer"
	"github.com/samorca/osha/operator-lifecycle/internal/policy"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	ConditionApproved = "Approved"
	ConditionExecuted = "Executed"

	defaultRequeueInterval = 30 * time.Second
)

var (
	ErrClientRequired   = errors.New("Kubernetes client is required")
	ErrExecutorRequired = errors.New("install plan executor is required")
	ErrPolicyRequired   = errors.New("policy engine is required")
)

type Reconciler struct {
	client.Client
	Scheme   *runtime.Scheme
	Executor *installer.Executor
	Policy   *policy.Engine

	RequeueInterval time.Duration
	Now             func() time.Time
}

// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorinstallplans,verbs=get;list;watch
// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorinstallplans/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete

func (r *Reconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues(
		"operatorInstallPlan",
		req.NamespacedName,
	)

	if err := r.validateDependencies(); err != nil {
		return ctrl.Result{}, err
	}

	var plan lifecyclev1alpha1.OperatorInstallPlan
	if err := r.Get(ctx, req.NamespacedName, &plan); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	switch plan.Status.Phase {
	case lifecyclev1alpha1.InstallPlanPhaseComplete,
		lifecyclev1alpha1.InstallPlanPhaseFailed,
		lifecyclev1alpha1.InstallPlanPhaseCancelled:
		return ctrl.Result{}, nil
	}

	now := metav1.NewTime(r.now())

	if plan.Spec.Approval == lifecyclev1alpha1.ApprovalModeManual &&
		!plan.Spec.Approved {

		plan.Status.ObservedGeneration = plan.Generation
		plan.Status.Phase =
			lifecyclev1alpha1.InstallPlanPhaseWaitingForApproval

		apimeta.SetStatusCondition(
			&plan.Status.Conditions,
			metav1.Condition{
				Type:               ConditionApproved,
				Status:             metav1.ConditionFalse,
				Reason:             "ApprovalRequired",
				Message:            "The install plan requires explicit approval",
				ObservedGeneration: plan.Generation,
				LastTransitionTime: now,
			},
		)

		if err := r.Status().Update(ctx, &plan); err != nil {
			return ctrl.Result{}, err
		}

		return ctrl.Result{
			RequeueAfter: r.requeueInterval(),
		}, nil
	}

	plan.Status.ObservedGeneration = plan.Generation
	plan.Status.Phase = lifecyclev1alpha1.InstallPlanPhaseInstalling

	if plan.Status.ApprovedAt == nil {
		plan.Status.ApprovedAt = &now
	}
	plan.Status.StartedAt = &now

	apimeta.SetStatusCondition(
		&plan.Status.Conditions,
		metav1.Condition{
			Type:               ConditionApproved,
			Status:             metav1.ConditionTrue,
			Reason:             "ApprovalSatisfied",
			Message:            "The install plan is approved or automatic",
			ObservedGeneration: plan.Generation,
			LastTransitionTime: now,
		},
	)
	apimeta.SetStatusCondition(
		&plan.Status.Conditions,
		metav1.Condition{
			Type:               ConditionExecuted,
			Status:             metav1.ConditionFalse,
			Reason:             "ExecutionStarting",
			Message:            "The install plan execution is starting",
			ObservedGeneration: plan.Generation,
			LastTransitionTime: now,
		},
	)

	if err := r.Status().Update(ctx, &plan); err != nil {
		return ctrl.Result{}, err
	}

	result, err := r.Executor.Execute(ctx, &plan)

	completedAt := metav1.NewTime(r.now())
	plan.Status.CompletedAt = &completedAt

	plan.Status.StepStatuses = make(
		[]lifecyclev1alpha1.InstallStepStatus,
		0,
	)

	for _, step := range resultSteps(result) {
		status := lifecyclev1alpha1.InstallStepStatus{
			ID:      step.ID,
			Phase:   step.Phase,
			Message: step.Message,
		}

		if !step.StartedAt.IsZero() {
			started := metav1.NewTime(step.StartedAt)
			status.StartedAt = &started
		}
		if !step.EndedAt.IsZero() {
			ended := metav1.NewTime(step.EndedAt)
			status.CompletedAt = &ended
		}

		plan.Status.StepStatuses = append(
			plan.Status.StepStatuses,
			status,
		)
	}

	if err != nil {
		plan.Status.Phase = lifecyclev1alpha1.InstallPlanPhaseFailed
		apimeta.SetStatusCondition(
			&plan.Status.Conditions,
			metav1.Condition{
				Type:               ConditionExecuted,
				Status:             metav1.ConditionFalse,
				Reason:             "ExecutionFailed",
				Message:            err.Error(),
				ObservedGeneration: plan.Generation,
				LastTransitionTime: completedAt,
			},
		)

		if statusErr := r.Status().Update(ctx, &plan); statusErr != nil {
			return ctrl.Result{}, statusErr
		}

		logger.Error(err, "install plan execution failed")
		return ctrl.Result{
			RequeueAfter: r.requeueInterval(),
		}, nil
	}

	plan.Status.Phase = lifecyclev1alpha1.InstallPlanPhaseComplete
	apimeta.SetStatusCondition(
		&plan.Status.Conditions,
		metav1.Condition{
			Type:               ConditionExecuted,
			Status:             metav1.ConditionTrue,
			Reason:             "ExecutionComplete",
			Message:            "All install plan steps completed",
			ObservedGeneration: plan.Generation,
			LastTransitionTime: completedAt,
		},
	)

	if err := r.Status().Update(ctx, &plan); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info(
		"install plan execution complete",
		"package",
		plan.Spec.Package,
		"targetVersion",
		plan.Spec.TargetVersion,
	)

	return ctrl.Result{}, nil
}

func resultSteps(result *installer.Result) []installer.StepResult {
	if result == nil {
		return nil
	}
	return result.Steps
}

func (r *Reconciler) validateDependencies() error {
	if r.Client == nil {
		return ErrClientRequired
	}
	if r.Executor == nil {
		return ErrExecutorRequired
	}
	if r.Policy == nil {
		return ErrPolicyRequired
	}
	if r.Scheme == nil {
		return errors.New("runtime scheme is required")
	}
	return nil
}

func (r *Reconciler) now() time.Time {
	if r.Now != nil {
		return r.Now().UTC()
	}
	return time.Now().UTC()
}

func (r *Reconciler) requeueInterval() time.Duration {
	if r.RequeueInterval > 0 {
		return r.RequeueInterval
	}
	return defaultRequeueInterval
}

func (r *Reconciler) SetupWithManager(
	manager ctrl.Manager,
) error {
	if err := r.validateDependencies(); err != nil {
		return err
	}
	return ctrl.NewControllerManagedBy(manager).
		For(&lifecyclev1alpha1.OperatorInstallPlan{}).
		Complete(r)
}
