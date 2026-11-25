package policy

import (
	"testing"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
)

func TestManualPlanRequiresApproval(t *testing.T) {
	engine := New(DefaultConfig())

	plan := validPlan()
	plan.Spec.Approval = lifecyclev1alpha1.ApprovalModeManual
	plan.Spec.Approved = false

	result := engine.Evaluate(plan)

	if result.Decision != DecisionRequireApproval {
		t.Fatalf(
			"expected RequireApproval, got %q",
			result.Decision,
		)
	}
}

func TestApprovedPlanAllowed(t *testing.T) {
	engine := New(DefaultConfig())

	plan := validPlan()
	plan.Spec.Approval = lifecyclev1alpha1.ApprovalModeManual
	plan.Spec.Approved = true

	result := engine.Evaluate(plan)

	if result.Decision != DecisionAllow {
		t.Fatalf(
			"expected Allow, got %q: %s",
			result.Decision,
			result.Message,
		)
	}
}

func TestDeleteDeniedByDefault(t *testing.T) {
	engine := New(DefaultConfig())

	plan := validPlan()
	plan.Spec.Approved = true
	plan.Spec.Steps[0].Action =
		lifecyclev1alpha1.InstallStepDelete
	plan.Spec.Steps[0].Manifest = ""

	result := engine.Evaluate(plan)

	if result.Decision != DecisionDeny {
		t.Fatalf(
			"expected Deny, got %q",
			result.Decision,
		)
	}
}

func validPlan() *lifecyclev1alpha1.OperatorInstallPlan {
	return &lifecyclev1alpha1.OperatorInstallPlan{
		Spec: lifecyclev1alpha1.OperatorInstallPlanSpec{
			Package:         "osha-operator",
			Action:          lifecyclev1alpha1.InstallPlanActionInstall,
			TargetVersion:   "1.0.0",
			TargetNamespace: "osha-system",
			Approval:        lifecyclev1alpha1.ApprovalModeManual,
			Approved:        true,
			ResolutionHash:  "test-hash",

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
`,
				},
			},
		},
	}
}
