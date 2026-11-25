package installer

import (
	"context"
	"testing"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/policy"

	appsv1 "k8s.io/api/apps/v1"
	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestExecuteCreate(t *testing.T) {
	scheme := runtime.NewScheme()

	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme() error = %v", err)
	}

	k8sClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()

	policyEngine := policy.New(
		policy.DefaultConfig(),
	)

	executor, err := New(
		k8sClient,
		policyEngine,
	)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	plan := &lifecyclev1alpha1.OperatorInstallPlan{
		Spec: lifecyclev1alpha1.OperatorInstallPlanSpec{
			Package:         "osha-operator",
			Action:          lifecyclev1alpha1.InstallPlanActionInstall,
			TargetVersion:   "1.0.0",
			TargetNamespace: "osha-system",
			Approval:        lifecyclev1alpha1.ApprovalModeManual,
			Approved:        true,
			ResolutionHash:  "test",

			Steps: []lifecyclev1alpha1.InstallPlanStep{
				{
					ID:     "deployment",
					Action: lifecyclev1alpha1.InstallStepCreate,

					Resource: lifecyclev1alpha1.ResourceReference{
						Group:     "apps",
						Version:   "v1",
						Kind:      "Deployment",
						Name:      "osha",
						Namespace: "osha-system",
					},

					Manifest: `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: osha
  namespace: osha-system
spec:
  selector:
    matchLabels:
      app: osha
  template:
    metadata:
      labels:
        app: osha
    spec:
      containers:
        - name: osha
          image: example/osha:1.0.0
`,
				},
			},
		},
	}

	result, err := executor.Execute(
		context.Background(),
		plan,
	)
	if err != nil {
		t.Fatalf("Execute() error = %v", err)
	}

	if len(result.Steps) != 1 {
		t.Fatalf(
			"expected 1 step result, got %d",
			len(result.Steps),
		)
	}

	var deployment appsv1.Deployment

	if err := k8sClient.Get(
		context.Background(),
		client.ObjectKey{
			Name:      "osha",
			Namespace: "osha-system",
		},
		&deployment,
	); err != nil {
		t.Fatalf(
			"expected Deployment to exist: %v",
			err,
		)
	}

	if deployment.Spec.Template.Spec.Containers[0].Image !=
		"example/osha:1.0.0" {

		t.Fatalf(
			"unexpected image %q",
			deployment.Spec.Template.Spec.Containers[0].Image,
		)
	}
}
