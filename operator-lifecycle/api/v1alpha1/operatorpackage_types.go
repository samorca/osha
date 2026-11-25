package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +kubebuilder:validation:Enum=Alpha;Beta;Stable;Deprecated
type ReleaseMaturity string

const (
	ReleaseMaturityAlpha      ReleaseMaturity = "Alpha"
	ReleaseMaturityBeta       ReleaseMaturity = "Beta"
	ReleaseMaturityStable     ReleaseMaturity = "Stable"
	ReleaseMaturityDeprecated ReleaseMaturity = "Deprecated"
)

// APIReference identifies a Kubernetes API provided or required by an Operator.
type APIReference struct {
	// +kubebuilder:validation:MinLength=1
	Group string `json:"group"`

	// +kubebuilder:validation:MinLength=1
	Version string `json:"version"`

	// +kubebuilder:validation:MinLength=1
	Kind string `json:"kind"`
}

// OperatorDependency declares another Operator package requirement.
type OperatorDependency struct {
	// +kubebuilder:validation:MinLength=1
	Package string `json:"package"`

	// VersionConstraint is a semantic-version constraint interpreted by
	// the lifecycle resolver.
	// +kubebuilder:validation:MinLength=1
	VersionConstraint string `json:"versionConstraint"`
}

// OperatorArtifact identifies the deployable artifacts for one release.
type OperatorArtifact struct {
	// Image is the controller image for this Operator release.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// BundleImage optionally identifies an OLM-compatible bundle image.
	// +optional
	BundleImage string `json:"bundleImage,omitempty"`

	// Digest optionally pins the controller image to an immutable digest.
	// +optional
	Digest string `json:"digest,omitempty"`
}

// OperatorRelease describes one available Operator version.
type OperatorRelease struct {
	// Version must contain the release version understood by the resolver.
	// +kubebuilder:validation:MinLength=1
	Version string `json:"version"`

	// +kubebuilder:validation:Enum=Alpha;Beta;Stable;Deprecated
	Maturity ReleaseMaturity `json:"maturity"`

	Artifacts OperatorArtifact `json:"artifacts"`

	// +optional
	ProvidedAPIs []APIReference `json:"providedAPIs,omitempty"`

	// +optional
	RequiredAPIs []APIReference `json:"requiredAPIs,omitempty"`

	// +optional
	Dependencies []OperatorDependency `json:"dependencies,omitempty"`

	// Replaces identifies the previous release when the package declares
	// an explicit sequential upgrade.
	// +optional
	Replaces string `json:"replaces,omitempty"`

	// +optional
	ReleaseNotes string `json:"releaseNotes,omitempty"`

	// +optional
	PublishedAt *metav1.Time `json:"publishedAt,omitempty"`
}

// ChannelEntry places a release into an update stream.
type ChannelEntry struct {
	// Version references an entry from spec.releases.
	// +kubebuilder:validation:MinLength=1
	Version string `json:"version"`

	// Replaces optionally defines the previous version in this channel.
	// +optional
	Replaces string `json:"replaces,omitempty"`

	// Skips allows explicitly bypassing listed versions during resolution.
	// +optional
	Skips []string `json:"skips,omitempty"`
}

// OperatorChannel defines one package update stream.
type OperatorChannel struct {
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Entries contains releases belonging to this channel.
	// +kubebuilder:validation:MinItems=1
	Entries []ChannelEntry `json:"entries"`
}

// OperatorPackageSpec defines an installable Operator package.
type OperatorPackageSpec struct {
	// DisplayName is the human-readable package name.
	// +kubebuilder:validation:MinLength=1
	DisplayName string `json:"displayName"`

	// Provider identifies the package publisher or owner.
	// +kubebuilder:validation:MinLength=1
	Provider string `json:"provider"`

	// Description provides a human-readable package description.
	// +optional
	Description string `json:"description,omitempty"`

	// DefaultChannel identifies the channel used when a subscription does not
	// explicitly select one.
	// +kubebuilder:validation:MinLength=1
	DefaultChannel string `json:"defaultChannel"`

	// +kubebuilder:validation:MinItems=1
	Channels []OperatorChannel `json:"channels"`

	// +kubebuilder:validation:MinItems=1
	Releases []OperatorRelease `json:"releases"`
}

// OperatorPackageStatus contains lifecycle-manager observations about the package.
type OperatorPackageStatus struct {
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// +optional
	ResolvedDefaultVersion string `json:"resolvedDefaultVersion,omitempty"`

	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=operatorpackages,scope=Namespaced,shortName=opkg,categories=osha
// +kubebuilder:printcolumn:name="Provider",type="string",JSONPath=".spec.provider"
// +kubebuilder:printcolumn:name="Default Channel",type="string",JSONPath=".spec.defaultChannel"
// +kubebuilder:printcolumn:name="Default Version",type="string",JSONPath=".status.resolvedDefaultVersion"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type OperatorPackage struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec OperatorPackageSpec `json:"spec"`

	// +optional
	Status OperatorPackageStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type OperatorPackageList struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ListMeta `json:"metadata,omitempty"`

	Items []OperatorPackage `json:"items"`
}

func init() {
	SchemeBuilder.Register(
		&OperatorPackage{},
		&OperatorPackageList{},
	)
}
