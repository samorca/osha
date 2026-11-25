package controller

import (
	"context"
	"testing"
	"time"

	platformv1alpha1 "github.com/samorca/osha/operators/api/v1alpha1"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestObservationInterval(t *testing.T) {
	tests := []struct {
		name     string
		seconds  int32
		expected time.Duration
	}{
		{
			name:     "uses configured interval",
			seconds:  60,
			expected: 60 * time.Second,
		},
		{
			name:     "defaults when zero",
			seconds:  0,
			expected: 30 * time.Second,
		},
		{
			name:     "defaults when below minimum",
			seconds:  4,
			expected: 30 * time.Second,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actual := observationInterval(test.seconds)

			if actual != test.expected {
				t.Fatalf(
					"expected %s, got %s",
					test.expected,
					actual,
				)
			}
		})
	}
}

func TestInt32Pointer(t *testing.T) {
	value := int32(3)

	result := int32Pointer(value)

	if result == nil {
		t.Fatal("expected non-nil pointer")
	}

	if *result != value {
		t.Fatalf(
			"expected %d, got %d",
			value,
			*result,
		)
	}
}

func TestReconcileReadyDeployment(t *testing.T) {
	ctx := context.Background()
	scheme := newTestScheme(t)

	replicas := int32(2)

	platform := &platformv1alpha1.OSHAPlatform{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "osha",
			Namespace:  "osha-system",
			Generation: 1,
		},
		Spec: platformv1alpha1.OSHAPlatformSpec{
			ManagementPolicy:           platformv1alpha1.ManagementPolicyObserveOnly,
			ObservationIntervalSeconds: 30,
			Components: []platformv1alpha1.WorkloadReference{
				{
					Name:            "osha-api",
					Namespace:       "osha-system",
					Kind:            platformv1alpha1.WorkloadKindDeployment,
					Required:        true,
					DesiredReplicas: &replicas,
				},
			},
		},
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "osha-api",
			Namespace:  "osha-system",
			Generation: 1,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
		},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 1,
			Replicas:           2,
			ReadyReplicas:      2,
			AvailableReplicas:  2,
			UpdatedReplicas:    2,
		},
	}

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&platformv1alpha1.OSHAPlatform{}).
		WithObjects(platform, deployment).
		Build()

	reconciler := &OSHAPlatformReconciler{
		Client: k8sClient,
		Scheme: scheme,
	}

	result, err := reconciler.Reconcile(
		ctx,
		ctrl.Request{
			NamespacedName: types.NamespacedName{
				Name:      "osha",
				Namespace: "osha-system",
			},
		},
	)

	if err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	if result.RequeueAfter != 30*time.Second {
		t.Fatalf(
			"expected requeue after 30s, got %s",
			result.RequeueAfter,
		)
	}

	var updated platformv1alpha1.OSHAPlatform

	if err := k8sClient.Get(
		ctx,
		types.NamespacedName{
			Name:      "osha",
			Namespace: "osha-system",
		},
		&updated,
	); err != nil {
		t.Fatalf("get reconciled OSHAPlatform: %v", err)
	}

	if updated.Status.Phase != platformv1alpha1.OSHAPlatformPhaseReady {
		t.Fatalf(
			"expected phase %q, got %q",
			platformv1alpha1.OSHAPlatformPhaseReady,
			updated.Status.Phase,
		)
	}

	if updated.Status.ReadyComponents != 1 {
		t.Fatalf(
			"expected 1 ready component, got %d",
			updated.Status.ReadyComponents,
		)
	}

	if updated.Status.TotalComponents != 1 {
		t.Fatalf(
			"expected 1 total component, got %d",
			updated.Status.TotalComponents,
		)
	}

	if len(updated.Status.Components) != 1 {
		t.Fatalf(
			"expected 1 component status, got %d",
			len(updated.Status.Components),
		)
	}

	component := updated.Status.Components[0]

	if component.Phase != platformv1alpha1.ComponentPhaseReady {
		t.Fatalf(
			"expected component Ready, got %q",
			component.Phase,
		)
	}

	assertCondition(
		t,
		updated.Status.Conditions,
		platformv1alpha1.ConditionTypeReady,
		metav1.ConditionTrue,
	)

	assertCondition(
		t,
		updated.Status.Conditions,
		platformv1alpha1.ConditionTypeAvailable,
		metav1.ConditionTrue,
	)

	assertCondition(
		t,
		updated.Status.Conditions,
		platformv1alpha1.ConditionTypeDegraded,
		metav1.ConditionFalse,
	)
}

func TestReconcileMissingRequiredDeployment(t *testing.T) {
	ctx := context.Background()
	scheme := newTestScheme(t)

	platform := &platformv1alpha1.OSHAPlatform{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "osha",
			Namespace:  "osha-system",
			Generation: 1,
		},
		Spec: platformv1alpha1.OSHAPlatformSpec{
			ManagementPolicy:           platformv1alpha1.ManagementPolicyObserveOnly,
			ObservationIntervalSeconds: 30,
			Components: []platformv1alpha1.WorkloadReference{
				{
					Name:      "osha-worker",
					Namespace: "osha-system",
					Kind:      platformv1alpha1.WorkloadKindDeployment,
					Required:  true,
				},
			},
		},
	}

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&platformv1alpha1.OSHAPlatform{}).
		WithObjects(platform).
		Build()

	reconciler := &OSHAPlatformReconciler{
		Client: k8sClient,
		Scheme: scheme,
	}

	_, err := reconciler.Reconcile(
		ctx,
		ctrl.Request{
			NamespacedName: types.NamespacedName{
				Name:      "osha",
				Namespace: "osha-system",
			},
		},
	)

	if err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var updated platformv1alpha1.OSHAPlatform

	if err := k8sClient.Get(
		ctx,
		types.NamespacedName{
			Name:      "osha",
			Namespace: "osha-system",
		},
		&updated,
	); err != nil {
		t.Fatalf("get reconciled OSHAPlatform: %v", err)
	}

	if updated.Status.Phase != platformv1alpha1.OSHAPlatformPhaseDegraded {
		t.Fatalf(
			"expected phase %q, got %q",
			platformv1alpha1.OSHAPlatformPhaseDegraded,
			updated.Status.Phase,
		)
	}

	if updated.Status.ReadyComponents != 0 {
		t.Fatalf(
			"expected 0 ready components, got %d",
			updated.Status.ReadyComponents,
		)
	}

	if len(updated.Status.Components) != 1 {
		t.Fatalf(
			"expected 1 component status, got %d",
			len(updated.Status.Components),
		)
	}

	if updated.Status.Components[0].Phase !=
		platformv1alpha1.ComponentPhaseNotFound {

		t.Fatalf(
			"expected component NotFound, got %q",
			updated.Status.Components[0].Phase,
		)
	}

	assertCondition(
		t,
		updated.Status.Conditions,
		platformv1alpha1.ConditionTypeReady,
		metav1.ConditionFalse,
	)

	assertCondition(
		t,
		updated.Status.Conditions,
		platformv1alpha1.ConditionTypeDegraded,
		metav1.ConditionTrue,
	)
}

func TestReconcileProgressingDeployment(t *testing.T) {
	ctx := context.Background()
	scheme := newTestScheme(t)

	replicas := int32(3)

	platform := &platformv1alpha1.OSHAPlatform{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "osha",
			Namespace:  "osha-system",
			Generation: 1,
		},
		Spec: platformv1alpha1.OSHAPlatformSpec{
			ManagementPolicy:           platformv1alpha1.ManagementPolicyObserveOnly,
			ObservationIntervalSeconds: 15,
			Components: []platformv1alpha1.WorkloadReference{
				{
					Name:            "osha-runtime",
					Kind:            platformv1alpha1.WorkloadKindDeployment,
					Required:        true,
					DesiredReplicas: &replicas,
				},
			},
		},
	}

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "osha-runtime",
			Namespace:  "osha-system",
			Generation: 2,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
		},
		Status: appsv1.DeploymentStatus{
			ObservedGeneration: 2,
			Replicas:           3,
			ReadyReplicas:      1,
			AvailableReplicas:  1,
			UpdatedReplicas:    3,
		},
	}

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&platformv1alpha1.OSHAPlatform{}).
		WithObjects(platform, deployment).
		Build()

	reconciler := &OSHAPlatformReconciler{
		Client: k8sClient,
		Scheme: scheme,
	}

	_, err := reconciler.Reconcile(
		ctx,
		ctrl.Request{
			NamespacedName: types.NamespacedName{
				Name:      "osha",
				Namespace: "osha-system",
			},
		},
	)

	if err != nil {
		t.Fatalf("reconcile returned error: %v", err)
	}

	var updated platformv1alpha1.OSHAPlatform

	if err := k8sClient.Get(
		ctx,
		types.NamespacedName{
			Name:      "osha",
			Namespace: "osha-system",
		},
		&updated,
	); err != nil {
		t.Fatalf("get reconciled OSHAPlatform: %v", err)
	}

	if updated.Status.Phase != platformv1alpha1.OSHAPlatformPhaseProgressing {
		t.Fatalf(
			"expected phase %q, got %q",
			platformv1alpha1.OSHAPlatformPhaseProgressing,
			updated.Status.Phase,
		)
	}

	assertCondition(
		t,
		updated.Status.Conditions,
		platformv1alpha1.ConditionTypeProgressing,
		metav1.ConditionTrue,
	)
}

func newTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()

	scheme := runtime.NewScheme()

	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add Kubernetes types to scheme: %v", err)
	}

	if err := platformv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add OSHA API types to scheme: %v", err)
	}

	return scheme
}

func assertCondition(
	t *testing.T,
	conditions []metav1.Condition,
	conditionType string,
	expected metav1.ConditionStatus,
) {
	t.Helper()

	for _, condition := range conditions {
		if condition.Type != conditionType {
			continue
		}

		if condition.Status != expected {
			t.Fatalf(
				"condition %q expected %q, got %q",
				conditionType,
				expected,
				condition.Status,
			)
		}

		return
	}

	t.Fatalf(
		"condition %q was not found",
		conditionType,
	)
}
