package upgrade

import (
	"testing"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestUpgradePath(t *testing.T) {
	engine := New()

	pkg := upgradePackage()

	plan, err := engine.Plan(
		pkg,
		"stable",
		"1.0.0",
		"1.2.0",
	)
	if err != nil {
		t.Fatalf("Plan() error = %v", err)
	}

	if len(plan.Transitions) != 2 {
		t.Fatalf(
			"expected 2 transitions, got %d",
			len(plan.Transitions),
		)
	}

	if plan.Transitions[0].From != "1.0.0" ||
		plan.Transitions[0].To != "1.1.0" {

		t.Fatalf(
			"unexpected first transition: %#v",
			plan.Transitions[0],
		)
	}

	if plan.Transitions[1].From != "1.1.0" ||
		plan.Transitions[1].To != "1.2.0" {

		t.Fatalf(
			"unexpected second transition: %#v",
			plan.Transitions[1],
		)
	}
}

func TestDowngradeRejected(t *testing.T) {
	engine := New()

	_, err := engine.Plan(
		upgradePackage(),
		"stable",
		"1.2.0",
		"1.0.0",
	)

	if err == nil {
		t.Fatal("expected downgrade error")
	}
}

func upgradePackage() *lifecyclev1alpha1.OperatorPackage {
	return &lifecyclev1alpha1.OperatorPackage{
		ObjectMeta: metav1.ObjectMeta{
			Name: "osha-operator",
		},

		Spec: lifecyclev1alpha1.OperatorPackageSpec{
			DisplayName:    "OSHA Operator",
			Provider:       "OSHA",
			DefaultChannel: "stable",

			Channels: []lifecyclev1alpha1.OperatorChannel{
				{
					Name: "stable",

					Entries: []lifecyclev1alpha1.ChannelEntry{
						{
							Version: "1.0.0",
						},
						{
							Version:  "1.1.0",
							Replaces: "1.0.0",
						},
						{
							Version:  "1.2.0",
							Replaces: "1.1.0",
						},
					},
				},
			},

			Releases: []lifecyclev1alpha1.OperatorRelease{
				{
					Version: "1.0.0",
					Artifacts: lifecyclev1alpha1.OperatorArtifact{
						Image: "registry.example/osha:1.0.0",
					},
				},
				{
					Version: "1.1.0",
					Artifacts: lifecyclev1alpha1.OperatorArtifact{
						Image: "registry.example/osha:1.1.0",
					},
				},
				{
					Version: "1.2.0",
					Artifacts: lifecyclev1alpha1.OperatorArtifact{
						Image: "registry.example/osha:1.2.0",
					},
				},
			},
		},
	}
}
