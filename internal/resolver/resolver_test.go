package resolver

import (
	"testing"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/catalog"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestResolveHighestCompatibleRelease(t *testing.T) {
	store := catalog.NewMemoryStore()

	pkg := packageFixture(
		"osha-operator",
		[]string{"1.0.0", "1.1.0", "2.0.0"},
	)

	if err := store.Put(pkg); err != nil {
		t.Fatalf("Put() error = %v", err)
	}

	engine, err := New(store)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := engine.Resolve(Request{
		Namespace:         "osha-system",
		Package:           "osha-operator",
		Channel:           "stable",
		VersionConstraint: ">=1.0.0, <2.0.0",
	})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}

	if result.Primary.Release.Version != "1.1.0" {
		t.Fatalf(
			"expected 1.1.0, got %q",
			result.Primary.Release.Version,
		)
	}
}

func TestResolveDependency(t *testing.T) {
	store := catalog.NewMemoryStore()

	dependency := packageFixture(
		"database-operator",
		[]string{"1.0.0"},
	)

	primary := packageFixture(
		"osha-operator",
		[]string{"1.0.0"},
	)

	primary.Spec.Releases[0].Dependencies =
		[]lifecyclev1alpha1.OperatorDependency{
			{
				Package:           "database-operator",
				VersionConstraint: ">=1.0.0",
			},
		}

	if err := store.Put(dependency); err != nil {
		t.Fatalf("Put dependency error = %v", err)
	}

	if err := store.Put(primary); err != nil {
		t.Fatalf("Put primary error = %v", err)
	}

	engine, err := New(store)
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := engine.Resolve(Request{
		Namespace: "osha-system",
		Package:   "osha-operator",
		Channel:   "stable",
	})
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}

	if len(result.Dependencies) != 1 {
		t.Fatalf(
			"expected 1 dependency, got %d",
			len(result.Dependencies),
		)
	}

	if result.Dependencies[0].Package != "database-operator" {
		t.Fatalf(
			"unexpected dependency %q",
			result.Dependencies[0].Package,
		)
	}
}

func packageFixture(
	name string,
	versions []string,
) *lifecyclev1alpha1.OperatorPackage {
	pkg := &lifecyclev1alpha1.OperatorPackage{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: "osha-system",
		},
		Spec: lifecyclev1alpha1.OperatorPackageSpec{
			DisplayName:    name,
			Provider:       "OSHA",
			DefaultChannel: "stable",
		},
	}

	channel := lifecyclev1alpha1.OperatorChannel{
		Name: "stable",
	}

	for index, version := range versions {
		release := lifecyclev1alpha1.OperatorRelease{
			Version:  version,
			Maturity: lifecyclev1alpha1.ReleaseMaturityStable,
			Artifacts: lifecyclev1alpha1.OperatorArtifact{
				Image: "registry.example/" + name + ":" + version,
			},
		}

		entry := lifecyclev1alpha1.ChannelEntry{
			Version: version,
		}

		if index > 0 {
			entry.Replaces = versions[index-1]
		}

		pkg.Spec.Releases = append(
			pkg.Spec.Releases,
			release,
		)

		channel.Entries = append(
			channel.Entries,
			entry,
		)
	}

	pkg.Spec.Channels = []lifecyclev1alpha1.OperatorChannel{
		channel,
	}

	return pkg
}
