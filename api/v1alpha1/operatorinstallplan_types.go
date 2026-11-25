package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +kubebuilder:validation:Enum=Install;Upgrade;Rollback
type InstallPlanAction string

const (
	InstallPlanActionInstall  InstallPlanAction = "Install"
	InstallPlanActionUpgrade  InstallPlanAction = "Upgrade"
	InstallPlanActionRollback InstallPlanAction = "Rollback"
)

// +kubebuilder:validation:Enum=Create;Update;Patch;Delete
type InstallStepAction string

const (
	InstallStepCreate InstallStepAction = "Create"
	InstallStepUpdate InstallStepAction = "Update"
	InstallStepPatch  InstallStepAction = "Patch"
	InstallStepDelete InstallStepAction = "Delete"
)

// +kubebuilder:validation:Enum=Pending;WaitingForApproval;Approved;Installing;Verifying;Complete;Failed;Cancelled
type InstallPlanPhase string

const (
	InstallPlanPhasePending            InstallPlanPhase = "Pending"
	InstallPlanPhaseWaitingForApproval InstallPlanPhase = "WaitingForApproval"
	InstallPlanPhaseApproved           InstallPlanPhase = "Approved"
	InstallPlanPhaseInstalling         InstallPlanPhase = "Installing"
	InstallPlanPhaseVerifying          InstallPlanPhase = "Verifying"
	InstallPlanPhaseComplete           InstallPlanPhase = "Complete"
	InstallPlanPhaseFailed             InstallPlanPhase = "Failed"
	InstallPlanPhaseCancelled          InstallPlanPhase = "Cancelled"
)

// +kubebuilder:validation:Enum=Pending;Running;Complete;Failed;Skipped
type InstallStepPhase string

const (
	InstallStepPhasePending  InstallStepPhase = "Pending"
	InstallStepPhaseRunning  InstallStepPhase = "Running"
	InstallStepPhaseComplete InstallStepPhase = "Complete"
	InstallStepPhaseFailed   InstallStepPhase = "Failed"
	InstallStepPhaseSkipped  InstallStepPhase = "Skipped"
)

// ResourceReference identifies a Kubernetes resource affected by a plan step.
type ResourceReference struct {
	// +kubebuilder:validation:MinLength=1
	Group string `json:"group"`

	// +kubebuilder:validation:MinLength=1
	Version string `json:"version"`

	// +kubebuilder:validation:MinLength=1
	Kind string `json:"kind"`

	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Namespace is empty for cluster-scoped resources.
	// +optional
	Namespace string `json:"namespace,omitempty"`
}

// InstallPlanStep describes one deterministic Kubernetes operation.
type InstallPlanStep struct {
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	Action InstallStepAction `json:"action"`

	Resource ResourceReference `json:"resource"`

	// Manifest contains the desired Kubernetes resource for Create or Update
	// operations. The installer will validate and decode it before execution.
	// +optional
	Manifest string `json:"manifest,omitempty"`

	// DependsOn contains IDs of steps that must complete first.
	// +optional
	DependsOn []string `json:"dependsOn,omitempty"`

	// Destructive marks operations requiring elevated policy evaluation.
	// +kubebuilder:default=false
	Destructive bool `json:"destructive,omitempty"`
}

// ResolvedDependency records a package selected by dependency resolution.
type ResolvedDependency struct {
	// +kubebuilder:validation:MinLength=1
	Package string `json:"package"`

	// +kubebuilder:validation:MinLength=1
	Version string `json:"version"`

	// +optional
	Channel string `json:"channel,omitempty"`

	// +optional
	Catalog NamespacedObjectReference `json:"catalog,omitempty"`
}

// OperatorInstallPlanSpec is an immutable execution proposal produced
// by lifecycle resolution.
type OperatorInstallPlanSpec struct {
	// Subscription references the request that generated this plan.
	Subscription NamespacedObjectReference `json:"subscription"`

	// Package identifies the primary Operator package.
	// +kubebuilder:validation:MinLength=1
	Package string `json:"package"`

	Action InstallPlanAction `json:"action"`

	// CurrentVersion is empty for a first installation.
	// +optional
	CurrentVersion string `json:"currentVersion,omitempty"`

	// TargetVersion is the release selected by dependency resolution.
	// +kubebuilder:validation:MinLength=1
	TargetVersion string `json:"targetVersion"`

	// TargetNamespace is the namespace where the primary Operator will run.
	// +kubebuilder:validation:MinLength=1
	TargetNamespace string `json:"targetNamespace"`

	Approval ApprovalMode `json:"approval"`

	// Approved is set explicitly for Manual plans.
	//
	// Automatic plans are evaluated by policy before execution and do not
	// treat this field as a policy bypass.
	// +kubebuilder:default=false
	Approved bool `json:"approved,omitempty"`

	// Dependencies records the resolver's selected dependency set.
	// +optional
	Dependencies []ResolvedDependency `json:"dependencies,omitempty"`

	// Steps is the ordered execution proposal.
	// +kubebuilder:validation:MinItems=1
	Steps []InstallPlanStep `json:"steps"`

	// ResolutionHash protects against executing a plan whose resolution
	// inputs have changed since it was created.
	// +kubebuilder:validation:MinLength=1
	ResolutionHash string `json:"resolutionHash"`
}

// InstallStepStatus reports execution state for a plan step.
type InstallStepStatus struct {
	// +kubebuilder:validation:MinLength=1
	ID string `json:"id"`

	Phase InstallStepPhase `json:"phase"`

	// +optional
	Message string `json:"message,omitempty"`

	// +optional
	StartedAt *metav1.Time `json:"startedAt,omitempty"`

	// +optional
	CompletedAt *metav1.Time `json:"completedAt,omitempty"`
}

// OperatorInstallPlanStatus describes plan execution.
type OperatorInstallPlanStatus struct {
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	Phase InstallPlanPhase `json:"phase,omitempty"`

	// +optional
	ResolvedAt *metav1.Time `json:"resolvedAt,omitempty"`

	// +optional
	ApprovedAt *metav1.Time `json:"approvedAt,omitempty"`

	// +optional
	StartedAt *metav1.Time `json:"startedAt,omitempty"`

	// +optional
	CompletedAt *metav1.Time `json:"completedAt,omitempty"`

	// +optional
	// +listType=map
	// +listMapKey=id
	StepStatuses []InstallStepStatus `json:"stepStatuses,omitempty"`

	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=operatorinstallplans,scope=Namespaced,shortName=oip,categories=osha
// +kubebuilder:printcolumn:name="Package",type="string",JSONPath=".spec.package"
// +kubebuilder:printcolumn:name="Action",type="string",JSONPath=".spec.action"
// +kubebuilder:printcolumn:name="Current",type="string",JSONPath=".spec.currentVersion"
// +kubebuilder:printcolumn:name="Target",type="string",JSONPath=".spec.targetVersion"
// +kubebuilder:printcolumn:name="Approval",type="string",JSONPath=".spec.approval"
// +kubebuilder:printcolumn:name="Approved",type="boolean",JSONPath=".spec.approved"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type OperatorInstallPlan struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec OperatorInstallPlanSpec `json:"spec"`

	// +optional
	Status OperatorInstallPlanStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type OperatorInstallPlanList struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ListMeta `json:"metadata,omitempty"`

	Items []OperatorInstallPlan `json:"items"`
}

func init() {
	SchemeBuilder.Register(
		&OperatorInstallPlan{},
		&OperatorInstallPlanList{},
	)
}
