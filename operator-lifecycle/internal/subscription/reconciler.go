package subscription

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"time"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/catalog"
	"github.com/samorca/osha/operator-lifecycle/internal/resolver"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

const (
	ConditionResolved         = "Resolved"
	ConditionInstallPlanReady = "InstallPlanReady"
	ConditionReady            = "Ready"

	labelSubscriptionName = "lifecycle.osha.io/subscription-name"
	labelSubscriptionUID  = "lifecycle.osha.io/subscription-uid"
	labelPackage          = "lifecycle.osha.io/package"
	labelTargetVersion    = "lifecycle.osha.io/target-version"

	defaultRequeueInterval = 5 * time.Minute
)

var (
	ErrCatalogStoreRequired = errors.New("catalog store is required")
	ErrResolverRequired     = errors.New("resolver is required")
	ErrPlanBuilderRequired  = errors.New("install plan builder is required")
)

type PlanBuildResult struct {
	Steps        []lifecyclev1alpha1.InstallPlanStep
	Dependencies []lifecyclev1alpha1.ResolvedDependency
}

type PlanBuilder interface {
	Build(
		ctx context.Context,
		subscription *lifecyclev1alpha1.OperatorSubscription,
		resolution *resolver.Result,
	) (*PlanBuildResult, error)
}

type Reconciler struct {
	client.Client
	Scheme *runtime.Scheme

	CatalogStore catalog.Store
	Resolver     *resolver.Resolver
	PlanBuilder  PlanBuilder

	RequeueInterval time.Duration
	Now             func() time.Time
}

// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorsubscriptions,verbs=get;list;watch
// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorsubscriptions/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorcatalogs,verbs=get;list;watch
// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorpackages,verbs=get;list;watch
// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorinstallplans,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=lifecycle.osha.io,resources=operatorinstallplans/status,verbs=get

func (r *Reconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues(
		"operatorSubscription",
		req.NamespacedName,
	)

	var subscription lifecyclev1alpha1.OperatorSubscription

	if err := r.Get(ctx, req.NamespacedName, &subscription); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if !subscription.DeletionTimestamp.IsZero() {
		return ctrl.Result{}, nil
	}

	if err := r.validateDependencies(); err != nil {
		return ctrl.Result{}, err
	}

	now := metav1.NewTime(r.now())

	if subscription.Spec.Paused {
		return r.handlePaused(ctx, &subscription, now)
	}

	if err := r.loadCatalogAndPackage(
		ctx,
		&subscription,
	); err != nil {
		r.setFailureStatus(
			&subscription,
			lifecyclev1alpha1.SubscriptionPhaseDegraded,
			"CatalogOrPackageUnavailable",
			err.Error(),
			now,
		)

		if statusErr := r.updateStatus(
			ctx,
			&subscription,
		); statusErr != nil {
			return ctrl.Result{}, statusErr
		}

		return ctrl.Result{
			RequeueAfter: r.requeueInterval(),
		}, nil
	}

	resolution, err := r.Resolver.Resolve(
		resolver.Request{
			Namespace:         subscription.Namespace,
			Package:           subscription.Spec.Package,
			Channel:           subscription.Spec.Channel,
			VersionConstraint: subscription.Spec.VersionConstraint,
			StartingVersion:   subscription.Spec.StartingVersion,
			InstalledVersions: installedVersions(&subscription),
		},
	)
	if err != nil {
		r.setFailureStatus(
			&subscription,
			lifecyclev1alpha1.SubscriptionPhaseDegraded,
			"ResolutionFailed",
			err.Error(),
			now,
		)

		if statusErr := r.updateStatus(
			ctx,
			&subscription,
		); statusErr != nil {
			return ctrl.Result{}, statusErr
		}

		return ctrl.Result{
			RequeueAfter: r.requeueInterval(),
		}, nil
	}

	subscription.Status.ResolvedChannel = resolution.Primary.Channel
	subscription.Status.ResolvedVersion =
		resolution.Primary.Release.Version
	subscription.Status.LastResolutionTime = &now

	apimeta.SetStatusCondition(
		&subscription.Status.Conditions,
		metav1.Condition{
			Type:               ConditionResolved,
			Status:             metav1.ConditionTrue,
			Reason:             "ResolutionSucceeded",
			Message:            "A compatible Operator release was resolved",
			ObservedGeneration: subscription.Generation,
			LastTransitionTime: now,
		},
	)

	if subscription.Status.InstalledVersion ==
		resolution.Primary.Release.Version {

		subscription.Status.Phase =
			lifecyclev1alpha1.SubscriptionPhaseReady
		subscription.Status.InstallPlanRef = nil

		apimeta.SetStatusCondition(
			&subscription.Status.Conditions,
			metav1.Condition{
				Type:               ConditionReady,
				Status:             metav1.ConditionTrue,
				Reason:             "DesiredVersionInstalled",
				Message:            "The resolved Operator version is installed",
				ObservedGeneration: subscription.Generation,
				LastTransitionTime: now,
			},
		)

		if err := r.updateStatus(
			ctx,
			&subscription,
		); err != nil {
			return ctrl.Result{}, err
		}

		return ctrl.Result{
			RequeueAfter: r.requeueInterval(),
		}, nil
	}

	plan, created, err := r.ensureInstallPlan(
		ctx,
		&subscription,
		resolution,
	)
	if err != nil {
		r.setFailureStatus(
			&subscription,
			lifecyclev1alpha1.SubscriptionPhaseDegraded,
			"InstallPlanCreationFailed",
			err.Error(),
			now,
		)

		if statusErr := r.updateStatus(
			ctx,
			&subscription,
		); statusErr != nil {
			return ctrl.Result{}, statusErr
		}

		return ctrl.Result{}, err
	}

	subscription.Status.InstallPlanRef =
		&lifecyclev1alpha1.NamespacedObjectReference{
			Name:      plan.Name,
			Namespace: plan.Namespace,
		}

	switch plan.Status.Phase {
	case lifecyclev1alpha1.InstallPlanPhaseComplete:
		subscription.Status.InstalledVersion =
			plan.Spec.TargetVersion
		subscription.Status.Phase =
			lifecyclev1alpha1.SubscriptionPhaseReady

	case lifecyclev1alpha1.InstallPlanPhaseInstalling,
		lifecyclev1alpha1.InstallPlanPhaseVerifying:

		if plan.Spec.Action ==
			lifecyclev1alpha1.InstallPlanActionUpgrade {

			subscription.Status.Phase =
				lifecyclev1alpha1.SubscriptionPhaseUpgrading
		} else {
			subscription.Status.Phase =
				lifecyclev1alpha1.SubscriptionPhaseInstalling
		}

	case lifecyclev1alpha1.InstallPlanPhaseFailed,
		lifecyclev1alpha1.InstallPlanPhaseCancelled:

		subscription.Status.Phase =
			lifecyclev1alpha1.SubscriptionPhaseDegraded

	default:
		subscription.Status.Phase =
			lifecyclev1alpha1.SubscriptionPhaseInstallPlanPending
	}

	apimeta.SetStatusCondition(
		&subscription.Status.Conditions,
		metav1.Condition{
			Type:   ConditionInstallPlanReady,
			Status: metav1.ConditionTrue,
			Reason: installPlanReason(created),
			Message: fmt.Sprintf(
				"InstallPlan %s targets version %s",
				plan.Name,
				plan.Spec.TargetVersion,
			),
			ObservedGeneration: subscription.Generation,
			LastTransitionTime: now,
		},
	)

	if err := r.updateStatus(
		ctx,
		&subscription,
	); err != nil {
		return ctrl.Result{}, err
	}

	logger.Info(
		"subscription reconciled",
		"package",
		subscription.Spec.Package,
		"resolvedVersion",
		subscription.Status.ResolvedVersion,
		"installedVersion",
		subscription.Status.InstalledVersion,
		"installPlan",
		plan.Name,
		"created",
		created,
	)

	return ctrl.Result{
		RequeueAfter: r.requeueInterval(),
	}, nil
}

func (r *Reconciler) loadCatalogAndPackage(
	ctx context.Context,
	subscription *lifecyclev1alpha1.OperatorSubscription,
) error {
	catalogNamespace := subscription.Spec.Catalog.Namespace
	if catalogNamespace == "" {
		catalogNamespace = subscription.Namespace
	}

	var operatorCatalog lifecyclev1alpha1.OperatorCatalog

	if err := r.Get(
		ctx,
		types.NamespacedName{
			Name:      subscription.Spec.Catalog.Name,
			Namespace: catalogNamespace,
		},
		&operatorCatalog,
	); err != nil {
		return fmt.Errorf(
			"get OperatorCatalog %s/%s: %w",
			catalogNamespace,
			subscription.Spec.Catalog.Name,
			err,
		)
	}

	if !operatorCatalog.Spec.Enabled {
		return fmt.Errorf(
			"OperatorCatalog %s/%s is disabled",
			catalogNamespace,
			operatorCatalog.Name,
		)
	}

	var operatorPackage lifecyclev1alpha1.OperatorPackage

	if err := r.Get(
		ctx,
		types.NamespacedName{
			Name:      subscription.Spec.Package,
			Namespace: subscription.Namespace,
		},
		&operatorPackage,
	); err != nil {
		return fmt.Errorf(
			"get OperatorPackage %s/%s: %w",
			subscription.Namespace,
			subscription.Spec.Package,
			err,
		)
	}

	if err := r.CatalogStore.Put(&operatorPackage); err != nil {
		return fmt.Errorf(
			"index OperatorPackage %s/%s: %w",
			operatorPackage.Namespace,
			operatorPackage.Name,
			err,
		)
	}

	return nil
}

func (r *Reconciler) ensureInstallPlan(
	ctx context.Context,
	subscription *lifecyclev1alpha1.OperatorSubscription,
	resolution *resolver.Result,
) (
	*lifecyclev1alpha1.OperatorInstallPlan,
	bool,
	error,
) {
	name := installPlanName(
		subscription.Name,
		resolution.Primary.Release.Version,
	)

	key := types.NamespacedName{
		Name:      name,
		Namespace: subscription.Namespace,
	}

	var existing lifecyclev1alpha1.OperatorInstallPlan

	if err := r.Get(ctx, key, &existing); err == nil {
		return &existing, false, nil
	} else if !apierrors.IsNotFound(err) {
		return nil, false, fmt.Errorf(
			"get existing OperatorInstallPlan %s: %w",
			key,
			err,
		)
	}

	buildResult, err := r.PlanBuilder.Build(
		ctx,
		subscription,
		resolution,
	)
	if err != nil {
		return nil, false, fmt.Errorf(
			"build install plan: %w",
			err,
		)
	}

	if buildResult == nil || len(buildResult.Steps) == 0 {
		return nil, false, errors.New(
			"plan builder returned no installation steps",
		)
	}

	currentVersion := strings.TrimSpace(
		subscription.Status.InstalledVersion,
	)

	action := lifecyclev1alpha1.InstallPlanActionInstall
	if currentVersion != "" {
		action = lifecyclev1alpha1.InstallPlanActionUpgrade
	}

	approved := subscription.Spec.Approval ==
		lifecyclev1alpha1.ApprovalModeAutomatic

	plan := &lifecyclev1alpha1.OperatorInstallPlan{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: subscription.Namespace,
			Labels: map[string]string{
				labelSubscriptionName: subscription.Name,
				labelSubscriptionUID:  string(subscription.UID),
				labelPackage:          subscription.Spec.Package,
				labelTargetVersion:    resolution.Primary.Release.Version,
			},
		},
		Spec: lifecyclev1alpha1.OperatorInstallPlanSpec{
			Subscription: lifecyclev1alpha1.NamespacedObjectReference{
				Name:      subscription.Name,
				Namespace: subscription.Namespace,
			},
			Package:         subscription.Spec.Package,
			Action:          action,
			CurrentVersion:  currentVersion,
			TargetVersion:   resolution.Primary.Release.Version,
			TargetNamespace: targetNamespace(subscription),
			Approval:        subscription.Spec.Approval,
			Approved:        approved,
			Dependencies:    buildResult.Dependencies,
			Steps:           buildResult.Steps,
			ResolutionHash: resolutionHash(
				subscription,
				resolution,
				buildResult,
			),
		},
	}

	if err := controllerutil.SetControllerReference(
		subscription,
		plan,
		r.Scheme,
	); err != nil {
		return nil, false, fmt.Errorf(
			"set subscription as InstallPlan owner: %w",
			err,
		)
	}

	if err := r.Create(ctx, plan); err != nil {
		return nil, false, fmt.Errorf(
			"create OperatorInstallPlan %s/%s: %w",
			plan.Namespace,
			plan.Name,
			err,
		)
	}

	return plan, true, nil
}

func (r *Reconciler) handlePaused(
	ctx context.Context,
	subscription *lifecyclev1alpha1.OperatorSubscription,
	now metav1.Time,
) (ctrl.Result, error) {
	subscription.Status.Phase =
		lifecyclev1alpha1.SubscriptionPhasePaused
	subscription.Status.ObservedGeneration =
		subscription.Generation

	apimeta.SetStatusCondition(
		&subscription.Status.Conditions,
		metav1.Condition{
			Type:               ConditionReady,
			Status:             metav1.ConditionFalse,
			Reason:             "SubscriptionPaused",
			Message:            "Lifecycle resolution and plan creation are paused",
			ObservedGeneration: subscription.Generation,
			LastTransitionTime: now,
		},
	)

	if err := r.updateStatus(ctx, subscription); err != nil {
		return ctrl.Result{}, err
	}

	return ctrl.Result{
		RequeueAfter: r.requeueInterval(),
	}, nil
}

func (r *Reconciler) setFailureStatus(
	subscription *lifecyclev1alpha1.OperatorSubscription,
	phase lifecyclev1alpha1.OperatorSubscriptionPhase,
	reason string,
	message string,
	now metav1.Time,
) {
	subscription.Status.Phase = phase

	apimeta.SetStatusCondition(
		&subscription.Status.Conditions,
		metav1.Condition{
			Type:               ConditionReady,
			Status:             metav1.ConditionFalse,
			Reason:             reason,
			Message:            message,
			ObservedGeneration: subscription.Generation,
			LastTransitionTime: now,
		},
	)
}

func (r *Reconciler) updateStatus(
	ctx context.Context,
	subscription *lifecyclev1alpha1.OperatorSubscription,
) error {
	subscription.Status.ObservedGeneration =
		subscription.Generation

	if err := r.Status().Update(
		ctx,
		subscription,
	); err != nil {
		return fmt.Errorf(
			"update OperatorSubscription status %s/%s: %w",
			subscription.Namespace,
			subscription.Name,
			err,
		)
	}

	return nil
}

func (r *Reconciler) validateDependencies() error {
	if r.CatalogStore == nil {
		return ErrCatalogStoreRequired
	}

	if r.Resolver == nil {
		return ErrResolverRequired
	}

	if r.PlanBuilder == nil {
		return ErrPlanBuilderRequired
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
	return ctrl.NewControllerManagedBy(manager).
		For(&lifecyclev1alpha1.OperatorSubscription{}).
		Owns(&lifecyclev1alpha1.OperatorInstallPlan{}).
		Complete(r)
}

func installedVersions(
	subscription *lifecyclev1alpha1.OperatorSubscription,
) map[string]string {
	result := make(map[string]string)

	if subscription.Status.InstalledVersion != "" {
		result[subscription.Spec.Package] =
			subscription.Status.InstalledVersion
	}

	return result
}

func targetNamespace(
	subscription *lifecyclev1alpha1.OperatorSubscription,
) string {
	if subscription.Spec.TargetNamespace != "" {
		return subscription.Spec.TargetNamespace
	}

	return subscription.Namespace
}

func installPlanName(
	subscriptionName string,
	targetVersion string,
) string {
	sum := sha256.Sum256(
		[]byte(subscriptionName + ":" + targetVersion),
	)

	suffix := hex.EncodeToString(sum[:])[:10]

	base := sanitizeName(subscriptionName)
	maxBaseLength := 63 - len("-plan-") - len(suffix)

	if len(base) > maxBaseLength {
		base = strings.TrimRight(
			base[:maxBaseLength],
			"-",
		)
	}

	return base + "-plan-" + suffix
}

func sanitizeName(value string) string {
	value = strings.ToLower(value)

	var builder strings.Builder

	for _, character := range value {
		switch {
		case character >= 'a' && character <= 'z':
			builder.WriteRune(character)

		case character >= '0' && character <= '9':
			builder.WriteRune(character)

		case character == '-':
			builder.WriteRune(character)

		default:
			builder.WriteRune('-')
		}
	}

	result := strings.Trim(builder.String(), "-")

	if result == "" {
		return "operator"
	}

	return result
}

func resolutionHash(
	subscription *lifecyclev1alpha1.OperatorSubscription,
	resolution *resolver.Result,
	buildResult *PlanBuildResult,
) string {
	var builder strings.Builder

	builder.WriteString(subscription.Spec.Package)
	builder.WriteString("|")
	builder.WriteString(resolution.Primary.Channel)
	builder.WriteString("|")
	builder.WriteString(resolution.Primary.Release.Version)
	builder.WriteString("|")

	for _, dependency := range resolution.Dependencies {
		builder.WriteString(dependency.Package)
		builder.WriteString("@")
		builder.WriteString(dependency.Release.Version)
		builder.WriteString(";")
	}

	for _, step := range buildResult.Steps {
		builder.WriteString(step.ID)
		builder.WriteString(":")
		builder.WriteString(string(step.Action))
		builder.WriteString(":")
		builder.WriteString(step.Resource.Group)
		builder.WriteString("/")
		builder.WriteString(step.Resource.Version)
		builder.WriteString("/")
		builder.WriteString(step.Resource.Kind)
		builder.WriteString("/")
		builder.WriteString(step.Resource.Namespace)
		builder.WriteString("/")
		builder.WriteString(step.Resource.Name)
		builder.WriteString(";")
	}

	sum := sha256.Sum256([]byte(builder.String()))

	return hex.EncodeToString(sum[:])
}

func installPlanReason(created bool) string {
	if created {
		return "InstallPlanCreated"
	}

	return "InstallPlanAvailable"
}
