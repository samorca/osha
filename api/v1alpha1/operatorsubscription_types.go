package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +kubebuilder:validation:Enum=Automatic;Manual
type ApprovalMode string

const (
	ApprovalModeAutomatic ApprovalMode = "Automatic"
	ApprovalModeManual    ApprovalMode = "Manual"
)

// +kubebuilder:validation:Enum=Pending;Resolving;InstallPlanPending;Installing;Ready;Upgrading;Degraded;Paused
type OperatorSubscriptionPhase string

const (
	SubscriptionPhasePending            OperatorSubscriptionPhase = "Pending"
	SubscriptionPhaseResolving          OperatorSubscriptionPhase = "Resolving"
	SubscriptionPhaseInstallPlanPending OperatorSubscriptionPhase = "InstallPlanPending"
	SubscriptionPhaseInstalling         OperatorSubscriptionPhase = "Installing"
	SubscriptionPhaseReady              OperatorSubscriptionPhase = "Ready"
	SubscriptionPhaseUpgrading          OperatorSubscriptionPhase = "Upgrading"
	SubscriptionPhaseDegraded           OperatorSubscriptionPhase = "Degraded"
	SubscriptionPhasePaused             OperatorSubscriptionPhase = "Paused"
)

// NamespacedObjectReference identifies another OSHA lifecycle resource.
type NamespacedObjectReference struct {
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Namespace defaults to the subscription namespace.
	// +optional
	Namespace string `json:"namespace,omitempty"`
}

// OperatorSubscriptionSpec declares the desired Operator installation.
type OperatorSubscriptionSpec struct {
	// Package identifies the OperatorPackage to install.
	// +kubebuilder:validation:MinLength=1
	Package string `json:"package"`

	// Catalog identifies the OperatorCatalog used for resolution.
	Catalog NamespacedObjectReference `json:"catalog"`

	// Channel selects an update stream.
	// When omitted, the package default channel is used.
	// +optional
	Channel string `json:"channel,omitempty"`

	// VersionConstraint optionally restricts acceptable versions.
	//
	// Examples may include exact versions or semantic-version ranges
	// supported by the OSHA resolver.
	// +optional
	VersionConstraint string `json:"versionConstraint,omitempty"`

	// Approval controls whether generated install plans may run
	// automatically or require explicit approval.
	// +kubebuilder:default=Manual
	Approval ApprovalMode `json:"approval,omitempty"`

	// TargetNamespace is where the Operator should be installed.
	//
	// When omitted, the subscription namespace is used.
	// +optional
	TargetNamespace string `json:"targetNamespace,omitempty"`

	// StartingVersion optionally requests an initial version.
	//
	// It does not disable future upgrades unless the version constraint
	// also prevents them.
	// +optional
	StartingVersion string `json:"startingVersion,omitempty"`

	// Paused prevents new install or upgrade plans from being executed.
	// +kubebuilder:default=false
	Paused bool `json:"paused,omitempty"`
}

// OperatorSubscriptionStatus describes the resolved and installed state.
type OperatorSubscriptionStatus struct {
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	Phase OperatorSubscriptionPhase `json:"phase,omitempty"`

	// ResolvedChannel is the effective channel selected by the resolver.
	// +optional
	ResolvedChannel string `json:"resolvedChannel,omitempty"`

	// ResolvedVersion is the version currently selected by dependency resolution.
	// +optional
	ResolvedVersion string `json:"resolvedVersion,omitempty"`

	// InstalledVersion is the version currently installed in the target namespace.
	// +optional
	InstalledVersion string `json:"installedVersion,omitempty"`

	// InstallPlanRef references the current install or upgrade plan.
	// +optional
	InstallPlanRef *NamespacedObjectReference `json:"installPlanRef,omitempty"`

	// LastResolutionTime records the most recently completed resolution.
	// +optional
	LastResolutionTime *metav1.Time `json:"lastResolutionTime,omitempty"`

	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=operatorsubscriptions,scope=Namespaced,shortName=osub,categories=osha
// +kubebuilder:printcolumn:name="Package",type="string",JSONPath=".spec.package"
// +kubebuilder:printcolumn:name="Channel",type="string",JSONPath=".status.resolvedChannel"
// +kubebuilder:printcolumn:name="Installed",type="string",JSONPath=".status.installedVersion"
// +kubebuilder:printcolumn:name="Available",type="string",JSONPath=".status.resolvedVersion"
// +kubebuilder:printcolumn:name="Approval",type="string",JSONPath=".spec.approval"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type OperatorSubscription struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec OperatorSubscriptionSpec `json:"spec"`

	// +optional
	Status OperatorSubscriptionStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type OperatorSubscriptionList struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ListMeta `json:"metadata,omitempty"`

	Items []OperatorSubscription `json:"items"`
}

func init() {
	SchemeBuilder.Register(
		&OperatorSubscription{},
		&OperatorSubscriptionList{},
	)
}
