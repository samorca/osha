package catalog

import (
	"errors"
	"testing"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestMemoryStorePutAndGet(t *testing.T) {
	store := NewMemoryStore()

	pkg := testPackage()

	if err := store.Put(pkg); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	got, err := store.Get("osha-system", "osha-operator")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	if got.Name != pkg.Name {
		t.Fatalf("expected package %q, got %q", pkg.Name, got.Name)
	}

	if got.Spec.DefaultChannel != "stable" {
		t.Fatalf(
			"expected default channel stable, got %q",
			got.Spec.DefaultChannel,
		)
	}
}

func TestMemoryStoreReturnsDeepCopy(t *testing.T) {
	store := NewMemoryStore()
	pkg := testPackage()

	if err := store.Put(pkg); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	got, err := store.Get("osha-system", "osha-operator")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	got.Spec.Provider = "modified"

	again, err := store.Get("osha-system", "osha-operator")
	if err != nil {
		t.Fatalf("Get() error = %v", err)
	}

	if again.Spec.Provider == "modified" {
		t.Fatal("store returned mutable internal package instance")
	}
}

func TestFindChannelUsesDefault(t *testing.T) {
	store := NewMemoryStore()

	if err := store.Put(testPackage()); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	channel, err := store.FindChannel(
		"osha-system",
		"osha-operator",
		"",
	)
	if err != nil {
		t.Fatalf("FindChannel() error = %v", err)
	}

	if channel.Name != "stable" {
		t.Fatalf("expected stable, got %q", channel.Name)
	}
}

func TestFindRelease(t *testing.T) {
	store := NewMemoryStore()

	if err := store.Put(testPackage()); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	release, err := store.FindRelease(
		"osha-system",
		"osha-operator",
		"1.1.0",
	)
	if err != nil {
		t.Fatalf("FindRelease() error = %v", err)
	}

	if release.Version != "1.1.0" {
		t.Fatalf("expected 1.1.0, got %q", release.Version)
	}
}

func TestValidatePackageRejectsMissingDefaultChannel(t *testing.T) {
	pkg := testPackage()
	pkg.Spec.DefaultChannel = "missing"

	err := ValidatePackage(pkg)

	if !errors.Is(err, ErrInvalidPackage) {
		t.Fatalf(
			"expected ErrInvalidPackage, got %v",
			err,
		)
	}
}

func testPackage() *lifecyclev1alpha1.OperatorPackage {
	return &lifecyclev1alpha1.OperatorPackage{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "osha-operator",
			Namespace: "osha-system",
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
					},
				},
			},

			Releases: []lifecyclev1alpha1.OperatorRelease{
				{
					Version:  "1.0.0",
					Maturity: lifecyclev1alpha1.ReleaseMaturityStable,
					Artifacts: lifecyclev1alpha1.OperatorArtifact{
						Image: "registry.example/osha:v1.0.0",
					},
				},
				{
					Version:  "1.1.0",
					Maturity: lifecyclev1alpha1.ReleaseMaturityStable,
					Artifacts: lifecyclev1alpha1.OperatorArtifact{
						Image: "registry.example/osha:v1.1.0",
					},
				},
			},
		},
	}
}
