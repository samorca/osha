package controller

import (
	"context"
	"fmt"
	"time"

	platformv1alpha1 "github.com/samorca/osha/operators/api/v1alpha1"

	appsv1 "k8s.io/api/apps/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	apiMeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

type OSHAPlatformReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=platform.osha.io,resources=oshaplatforms,verbs=get;list;watch
// +kubebuilder:rbac:groups=platform.osha.io,resources=oshaplatforms/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=statefulsets,verbs=get;list;watch
// +kubebuilder:rbac:groups=apps,resources=daemonsets,verbs=get;list;watch

func (r *OSHAPlatformReconciler) Reconcile(
	ctx context.Context,
	req ctrl.Request,
) (ctrl.Result, error) {
	logger := log.FromContext(ctx).WithValues(
		"oshaplatform",
		req.NamespacedName,
	)

	var platform platformv1alpha1.OSHAPlatform

	if err := r.Get(ctx, req.NamespacedName, &platform); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}

		return ctrl.Result{}, fmt.Errorf(
			"get OSHAPlatform %s: %w",
			req.NamespacedName,
			err,
		)
	}

	interval := observationInterval(platform.Spec.ObservationIntervalSeconds)

	if !platform.DeletionTimestamp.IsZero() {
		logger.Info("OSHAPlatform is terminating")

		return ctrl.Result{}, nil
	}

	statuses := make(
		[]platformv1alpha1.OSHAComponentStatus,
		0,
		len(platform.Spec.Components),
	)

	var readyComponents int32
	requiredReady := true
	progressing := false
	degraded := false

	for _, component := range platform.Spec.Components {
		status, err := r.observeComponent(
			ctx,
			platform.Namespace,
			component,
		)

		if err != nil {
			logger.Error(
				err,
				"unable to observe component",
				"component",
				component.Name,
				"kind",
				component.Kind,
			)

			return ctrl.Result{}, err
		}

		statuses = append(statuses, status)

		if status.Phase == platformv1alpha1.ComponentPhaseReady {
			readyComponents++
		}

		if component.Required &&
			status.Phase != platformv1alpha1.ComponentPhaseReady {
			requiredReady = false
		}

		switch status.Phase {
		case platformv1alpha1.ComponentPhaseNotFound,
			platformv1alpha1.ComponentPhaseDegraded:
			if component.Required {
				degraded = true
			}

		case platformv1alpha1.ComponentPhaseProgressing,
			platformv1alpha1.ComponentPhaseUnknown:
			if component.Required {
				progressing = true
			}
		}
	}

	phase := platformv1alpha1.OSHAPlatformPhaseReady

	switch {
	case degraded:
		phase = platformv1alpha1.OSHAPlatformPhaseDegraded
	case progressing || !requiredReady:
		phase = platformv1alpha1.OSHAPlatformPhaseProgressing
	}

	now := metav1.Now()

	platform.Status.ObservedGeneration = platform.Generation
	platform.Status.Phase = phase
	platform.Status.ReadyComponents = readyComponents
	platform.Status.TotalComponents = int32(len(statuses))
	platform.Status.Components = statuses
	platform.Status.LastReconcileTime = &now

	setConditions(
		&platform,
		requiredReady,
		progressing,
		degraded,
		now,
	)

	if err := r.Status().Update(ctx, &platform); err != nil {
		return ctrl.Result{}, fmt.Errorf(
			"update OSHAPlatform status %s: %w",
			req.NamespacedName,
			err,
		)
	}

	logger.Info(
		"OSHAPlatform observation completed",
		"phase",
		phase,
		"readyComponents",
		readyComponents,
		"totalComponents",
		len(statuses),
		"managementPolicy",
		platform.Spec.ManagementPolicy,
		"paused",
		platform.Spec.Paused,
	)

	return ctrl.Result{
		RequeueAfter: interval,
	}, nil
}

func (r *OSHAPlatformReconciler) observeComponent(
	ctx context.Context,
	defaultNamespace string,
	ref platformv1alpha1.WorkloadReference,
) (platformv1alpha1.OSHAComponentStatus, error) {
	namespace := ref.Namespace

	if namespace == "" {
		namespace = defaultNamespace
	}

	now := metav1.Now()

	status := platformv1alpha1.OSHAComponentStatus{
		Name:             ref.Name,
		Namespace:        namespace,
		Kind:             ref.Kind,
		Required:         ref.Required,
		DesiredReplicas:  ref.DesiredReplicas,
		Phase:            platformv1alpha1.ComponentPhaseUnknown,
		LastObservedTime: now,
	}

	key := types.NamespacedName{
		Name:      ref.Name,
		Namespace: namespace,
	}

	switch ref.Kind {
	case platformv1alpha1.WorkloadKindDeployment:
		return r.observeDeployment(ctx, key, status)

	case platformv1alpha1.WorkloadKindStatefulSet:
		return r.observeStatefulSet(ctx, key, status)

	case platformv1alpha1.WorkloadKindDaemonSet:
		return r.observeDaemonSet(ctx, key, status)

	default:
		status.Phase = platformv1alpha1.ComponentPhaseDegraded
		status.Message = fmt.Sprintf(
			"unsupported workload kind %q",
			ref.Kind,
		)

		return status, nil
	}
}

func (r *OSHAPlatformReconciler) observeDeployment(
	ctx context.Context,
	key types.NamespacedName,
	status platformv1alpha1.OSHAComponentStatus,
) (platformv1alpha1.OSHAComponentStatus, error) {
	var deployment appsv1.Deployment

	if err := r.Get(ctx, key, &deployment); err != nil {
		if apierrors.IsNotFound(err) {
			status.Phase = platformv1alpha1.ComponentPhaseNotFound
			status.Message = "Deployment not found"

			return status, nil
		}

		return status, fmt.Errorf(
			"get Deployment %s: %w",
			key,
			err,
		)
	}

	status.ResourceVersion = deployment.ResourceVersion
	status.ObservedGeneration = deployment.Status.ObservedGeneration
	status.CurrentReplicas = deployment.Status.Replicas
	status.ReadyReplicas = deployment.Status.ReadyReplicas
	status.AvailableReplicas = deployment.Status.AvailableReplicas
	status.UpdatedReplicas = deployment.Status.UpdatedReplicas

	desired := int32(1)

	if deployment.Spec.Replicas != nil {
		desired = *deployment.Spec.Replicas
	}

	if status.DesiredReplicas != nil {
		desired = *status.DesiredReplicas
	} else {
		status.DesiredReplicas = int32Pointer(desired)
	}

	if desired == 0 {
		status.Phase = platformv1alpha1.ComponentPhaseReady
		status.Message = "Deployment intentionally scaled to zero"

		return status, nil
	}

	if deployment.Status.ObservedGeneration < deployment.Generation {
		status.Phase = platformv1alpha1.ComponentPhaseProgressing
		status.Message = "Deployment controller has not observed the latest generation"

		return status, nil
	}

	if deployment.Status.ReadyReplicas < desired ||
		deployment.Status.AvailableReplicas < desired ||
		deployment.Status.UpdatedReplicas < desired {

		status.Phase = platformv1alpha1.ComponentPhaseProgressing
		status.Message = fmt.Sprintf(
			"Deployment ready=%d available=%d updated=%d desired=%d",
			deployment.Status.ReadyReplicas,
			deployment.Status.AvailableReplicas,
			deployment.Status.UpdatedReplicas,
			desired,
		)

		return status, nil
	}

	status.Phase = platformv1alpha1.ComponentPhaseReady
	status.Message = "Deployment is ready"

	return status, nil
}

func (r *OSHAPlatformReconciler) observeStatefulSet(
	ctx context.Context,
	key types.NamespacedName,
	status platformv1alpha1.OSHAComponentStatus,
) (platformv1alpha1.OSHAComponentStatus, error) {
	var statefulSet appsv1.StatefulSet

	if err := r.Get(ctx, key, &statefulSet); err != nil {
		if apierrors.IsNotFound(err) {
			status.Phase = platformv1alpha1.ComponentPhaseNotFound
			status.Message = "StatefulSet not found"

			return status, nil
		}

		return status, fmt.Errorf(
			"get StatefulSet %s: %w",
			key,
			err,
		)
	}

	status.ResourceVersion = statefulSet.ResourceVersion
	status.ObservedGeneration = statefulSet.Status.ObservedGeneration
	status.CurrentReplicas = statefulSet.Status.CurrentReplicas
	status.ReadyReplicas = statefulSet.Status.ReadyReplicas
	status.UpdatedReplicas = statefulSet.Status.UpdatedReplicas
	status.AvailableReplicas = statefulSet.Status.AvailableReplicas

	desired := *statefulSet.Spec.Replicas

	if status.DesiredReplicas != nil {
		desired = *status.DesiredReplicas
	} else {
		status.DesiredReplicas = int32Pointer(desired)
	}

	if desired == 0 {
		status.Phase = platformv1alpha1.ComponentPhaseReady
		status.Message = "StatefulSet intentionally scaled to zero"

		return status, nil
	}

	if statefulSet.Status.ObservedGeneration < statefulSet.Generation {
		status.Phase = platformv1alpha1.ComponentPhaseProgressing
		status.Message = "StatefulSet controller has not observed the latest generation"

		return status, nil
	}

	if statefulSet.Status.ReadyReplicas < desired ||
		statefulSet.Status.UpdatedReplicas < desired {

		status.Phase = platformv1alpha1.ComponentPhaseProgressing
		status.Message = fmt.Sprintf(
			"StatefulSet ready=%d updated=%d desired=%d",
			statefulSet.Status.ReadyReplicas,
			statefulSet.Status.UpdatedReplicas,
			desired,
		)

		return status, nil
	}

	status.Phase = platformv1alpha1.ComponentPhaseReady
	status.Message = "StatefulSet is ready"

	return status, nil
}

func (r *OSHAPlatformReconciler) observeDaemonSet(
	ctx context.Context,
	key types.NamespacedName,
	status platformv1alpha1.OSHAComponentStatus,
) (platformv1alpha1.OSHAComponentStatus, error) {
	var daemonSet appsv1.DaemonSet

	if err := r.Get(ctx, key, &daemonSet); err != nil {
		if apierrors.IsNotFound(err) {
			status.Phase = platformv1alpha1.ComponentPhaseNotFound
			status.Message = "DaemonSet not found"

			return status, nil
		}

		return status, fmt.Errorf(
			"get DaemonSet %s: %w",
			key,
			err,
		)
	}

	status.ResourceVersion = daemonSet.ResourceVersion
	status.ObservedGeneration = daemonSet.Status.ObservedGeneration
	status.CurrentReplicas = daemonSet.Status.CurrentNumberScheduled
	status.ReadyReplicas = daemonSet.Status.NumberReady
	status.AvailableReplicas = daemonSet.Status.NumberAvailable
	status.UpdatedReplicas = daemonSet.Status.UpdatedNumberScheduled

	desired := daemonSet.Status.DesiredNumberScheduled
	status.DesiredReplicas = int32Pointer(desired)

	if daemonSet.Status.ObservedGeneration < daemonSet.Generation {
		status.Phase = platformv1alpha1.ComponentPhaseProgressing
		status.Message = "DaemonSet controller has not observed the latest generation"

		return status, nil
	}

	if daemonSet.Status.NumberUnavailable > 0 ||
		daemonSet.Status.NumberReady < desired ||
		daemonSet.Status.UpdatedNumberScheduled < desired {

		status.Phase = platformv1alpha1.ComponentPhaseProgressing
		status.Message = fmt.Sprintf(
			"DaemonSet ready=%d updated=%d unavailable=%d desired=%d",
			daemonSet.Status.NumberReady,
			daemonSet.Status.UpdatedNumberScheduled,
			daemonSet.Status.NumberUnavailable,
			desired,
		)

		return status, nil
	}

	status.Phase = platformv1alpha1.ComponentPhaseReady
	status.Message = "DaemonSet is ready"

	return status, nil
}

func setConditions(
	platform *platformv1alpha1.OSHAPlatform,
	ready bool,
	progressing bool,
	degraded bool,
	now metav1.Time,
) {
	availableStatus := metav1.ConditionFalse
	availableReason := "ComponentsUnavailable"
	availableMessage := "One or more required OSHA components are unavailable"

	if ready {
		availableStatus = metav1.ConditionTrue
		availableReason = "AllRequiredComponentsReady"
		availableMessage = "All required OSHA components are ready"
	}

	progressingStatus := metav1.ConditionFalse
	progressingReason := "Stable"
	progressingMessage := "No required OSHA component is progressing"

	if progressing {
		progressingStatus = metav1.ConditionTrue
		progressingReason = "ComponentsProgressing"
		progressingMessage = "One or more required OSHA components are progressing"
	}

	degradedStatus := metav1.ConditionFalse
	degradedReason := "Healthy"
	degradedMessage := "No required OSHA component is degraded"

	if degraded {
		degradedStatus = metav1.ConditionTrue
		degradedReason = "ComponentsDegraded"
		degradedMessage = "One or more required OSHA components are degraded or missing"
	}

	apiMeta.SetStatusCondition(
		&platform.Status.Conditions,
		metav1.Condition{
			Type:               platformv1alpha1.ConditionTypeAvailable,
			Status:             availableStatus,
			Reason:             availableReason,
			Message:            availableMessage,
			ObservedGeneration: platform.Generation,
			LastTransitionTime: now,
		},
	)

	apiMeta.SetStatusCondition(
		&platform.Status.Conditions,
		metav1.Condition{
			Type:               platformv1alpha1.ConditionTypeProgressing,
			Status:             progressingStatus,
			Reason:             progressingReason,
			Message:            progressingMessage,
			ObservedGeneration: platform.Generation,
			LastTransitionTime: now,
		},
	)

	apiMeta.SetStatusCondition(
		&platform.Status.Conditions,
		metav1.Condition{
			Type:               platformv1alpha1.ConditionTypeDegraded,
			Status:             degradedStatus,
			Reason:             degradedReason,
			Message:            degradedMessage,
			ObservedGeneration: platform.Generation,
			LastTransitionTime: now,
		},
	)

	apiMeta.SetStatusCondition(
		&platform.Status.Conditions,
		metav1.Condition{
			Type:               platformv1alpha1.ConditionTypeReady,
			Status:             availableStatus,
			Reason:             availableReason,
			Message:            availableMessage,
			ObservedGeneration: platform.Generation,
			LastTransitionTime: now,
		},
	)
}

func observationInterval(seconds int32) time.Duration {
	if seconds < 5 {
		seconds = 30
	}

	return time.Duration(seconds) * time.Second
}

func int32Pointer(value int32) *int32 {
	return &value
}

func (r *OSHAPlatformReconciler) SetupWithManager(
	mgr ctrl.Manager,
) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&platformv1alpha1.OSHAPlatform{}).
		Complete(r)
}
