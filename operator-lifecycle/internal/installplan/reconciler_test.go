package installplan

import (
	"context"
	"testing"
	"time"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/installer"
	"github.com/samorca/osha/operator-lifecycle/internal/policy"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestReconcileWaitingForApproval(t *testing.T) {
	scheme := testScheme(t)
	plan := testPlan(
		lifecyclev1alpha1.ApprovalModeManual,
		false,
	)

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&lifecyclev1alpha1.OperatorInstallPlan{}).
		WithObjects(plan).
		Build()

	policyEngine := policy.New(policy.DefaultConfig())
	executor, err := installer.New(k8sClient, policyEngine)
	if err != nil {
		t.Fatalf("installer.New() error = %v", err)
	}

	reconciler := &Reconciler{
		Client:   k8sClient,
		Scheme:   scheme,
		Executor: executor,
		Policy:   policyEngine,
		Now:      func() time.Time { return time.Unix(100, 0).UTC() },
	}

	result, err := reconciler.Reconcile(
		context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{
			Name:      plan.Name,
			Namespace: plan.Namespace,
		}},
	)
	if err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}
	if result.RequeueAfter != 30*time.Second {
		t.Fatalf("expected 30s requeue, got %s", result.RequeueAfter)
	}

	var updated lifecyclev1alpha1.OperatorInstallPlan
	if err := k8sClient.Get(
		context.Background(),
		types.NamespacedName{Name: plan.Name, Namespace: plan.Namespace},
		&updated,
	); err != nil {
		t.Fatalf("get plan: %v", err)
	}

	if updated.Status.Phase != lifecyclev1alpha1.InstallPlanPhaseWaitingForApproval {
		t.Fatalf(
			"expected WaitingForApproval, got %q",
			updated.Status.Phase,
		)
	}
}

func TestReconcileAutomaticPlanExecutesDeployment(t *testing.T) {
	scheme := testScheme(t)
	plan := testPlan(
		lifecyclev1alpha1.ApprovalModeAutomatic,
		false,
	)

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithStatusSubresource(&lifecyclev1alpha1.OperatorInstallPlan{}).
		WithObjects(plan).
		Build()

	policyEngine := policy.New(policy.DefaultConfig())
	executor, err := installer.New(k8sClient, policyEngine)
	if err != nil {
		t.Fatalf("installer.New() error = %v", err)
	}

	reconciler := &Reconciler{
		Client:   k8sClient,
		Scheme:   scheme,
		Executor: executor,
		Policy:   policyEngine,
		Now:      func() time.Time { return time.Unix(200, 0).UTC() },
	}

	_, err = reconciler.Reconcile(
		context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{
			Name:      plan.Name,
			Namespace: plan.Namespace,
		}},
	)
	if err != nil {
		t.Fatalf("Reconcile() error = %v", err)
	}

	var updated lifecyclev1alpha1.OperatorInstallPlan
	if err := k8sClient.Get(
		context.Background(),
		types.NamespacedName{Name: plan.Name, Namespace: plan.Namespace},
		&updated,
	); err != nil {
		t.Fatalf("get plan: %v", err)
	}

	if updated.Status.Phase != lifecyclev1alpha1.InstallPlanPhaseComplete {
		t.Fatalf("expected Complete, got %q", updated.Status.Phase)
	}

	var deployment appsv1.Deployment
	if err := k8sClient.Get(
		context.Background(),
		types.NamespacedName{Name: "osha-test", Namespace: "osha-system"},
		&deployment,
	); err != nil {
		t.Fatalf("get deployed workload: %v", err)
	}
}

func testScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add Kubernetes scheme: %v", err)
	}
	if err := lifecyclev1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add lifecycle scheme: %v", err)
	}
	return scheme
}

func testPlan(
	approval lifecyclev1alpha1.ApprovalMode,
	approved bool,
) *lifecyclev1alpha1.OperatorInstallPlan {
	return &lifecyclev1alpha1.OperatorInstallPlan{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "osha-test-plan",
			Namespace:  "osha-system",
			Generation: 1,
		},
		Spec: lifecyclev1alpha1.OperatorInstallPlanSpec{
			Package:         "osha-test",
			Action:          lifecyclev1alpha1.InstallPlanActionInstall,
			TargetVersion:   "1.0.0",
			TargetNamespace: "osha-system",
			Approval:        approval,
			Approved:        approved,
			ResolutionHash:  "test-resolution",
			Steps: []lifecyclev1alpha1.InstallPlanStep{
				{
					ID:     "operator-deployment",
					Action: lifecyclev1alpha1.InstallStepCreate,
					Resource: lifecyclev1alpha1.ResourceReference{
						Group:     "apps",
						Version:   "v1",
						Kind:      "Deployment",
						Name:      "osha-test",
						Namespace: "osha-system",
					},
					Manifest: `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: osha-test
  namespace: osha-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: osha-test
  template:
    metadata:
      labels:
        app: osha-test
    spec:
      containers:
        - name: manager
          image: example.invalid/osha-test:1.0.0
`,
				},
			},
		},
	}
}
