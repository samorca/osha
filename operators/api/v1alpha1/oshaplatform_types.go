package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	ConditionTypeAvailable   = "Available"
	ConditionTypeProgressing = "Progressing"
	ConditionTypeDegraded    = "Degraded"
	ConditionTypeReady       = "Ready"
)

// +kubebuilder:validation:Enum=ObserveOnly;Reconcile
type ManagementPolicy string

const (
	ManagementPolicyObserveOnly ManagementPolicy = "ObserveOnly"
	ManagementPolicyReconcile   ManagementPolicy = "Reconcile"
)

// +kubebuilder:validation:Enum=Deployment;StatefulSet;DaemonSet
type WorkloadKind string

const (
	WorkloadKindDeployment  WorkloadKind = "Deployment"
	WorkloadKindStatefulSet WorkloadKind = "StatefulSet"
	WorkloadKindDaemonSet   WorkloadKind = "DaemonSet"
)

// +kubebuilder:validation:Enum=Pending;Discovering;Progressing;Ready;Degraded;Terminating
type OSHAPlatformPhase string

const (
	OSHAPlatformPhasePending     OSHAPlatformPhase = "Pending"
	OSHAPlatformPhaseDiscovering OSHAPlatformPhase = "Discovering"
	OSHAPlatformPhaseProgressing OSHAPlatformPhase = "Progressing"
	OSHAPlatformPhaseReady       OSHAPlatformPhase = "Ready"
	OSHAPlatformPhaseDegraded    OSHAPlatformPhase = "Degraded"
	OSHAPlatformPhaseTerminating OSHAPlatformPhase = "Terminating"
)

// +kubebuilder:validation:Enum=Unknown;NotFound;Progressing;Ready;Degraded
type ComponentPhase string

const (
	ComponentPhaseUnknown     ComponentPhase = "Unknown"
	ComponentPhaseNotFound    ComponentPhase = "NotFound"
	ComponentPhaseProgressing ComponentPhase = "Progressing"
	ComponentPhaseReady       ComponentPhase = "Ready"
	ComponentPhaseDegraded    ComponentPhase = "Degraded"
)

type WorkloadReference struct {
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=253
	Name string `json:"name"`

	// Defaults to the OSHAPlatform namespace when empty.
	// +optional
	// +kubebuilder:validation:MaxLength=63
	Namespace string `json:"namespace,omitempty"`

	// +kubebuilder:default=Deployment
	Kind WorkloadKind `json:"kind,omitempty"`

	// +kubebuilder:default=true
	Required bool `json:"required,omitempty"`

	// Used as an expectation in ObserveOnly and may be reconciled in Reconcile mode.
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1000
	DesiredReplicas *int32 `json:"desiredReplicas,omitempty"`
}

type ApprovalPolicy struct {
	// +kubebuilder:default=true
	RequireApproval bool `json:"requireApproval,omitempty"`

	// +kubebuilder:default=false
	AllowScaling bool `json:"allowScaling,omitempty"`

	// +kubebuilder:default=false
	AllowRestart bool `json:"allowRestart,omitempty"`

	// +kubebuilder:default=false
	AllowDeletion bool `json:"allowDeletion,omitempty"`
}

type OSHAPlatformSpec struct {
	// +kubebuilder:default=ObserveOnly
	ManagementPolicy ManagementPolicy `json:"managementPolicy,omitempty"`

	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=100
	Components []WorkloadReference `json:"components"`

	// +kubebuilder:default=30
	// +kubebuilder:validation:Minimum=5
	// +kubebuilder:validation:Maximum=3600
	ObservationIntervalSeconds int32 `json:"observationIntervalSeconds,omitempty"`

	// +optional
	Approval ApprovalPolicy `json:"approval,omitempty"`

	// +kubebuilder:default=false
	Paused bool `json:"paused,omitempty"`
}

type OSHAComponentStatus struct {
	Name      string         `json:"name"`
	Namespace string         `json:"namespace"`
	Kind      WorkloadKind   `json:"kind"`
	Phase     ComponentPhase `json:"phase"`
	Required  bool           `json:"required"`

	// +optional
	DesiredReplicas *int32 `json:"desiredReplicas,omitempty"`

	CurrentReplicas   int32 `json:"currentReplicas,omitempty"`
	ReadyReplicas     int32 `json:"readyReplicas,omitempty"`
	AvailableReplicas int32 `json:"availableReplicas,omitempty"`
	UpdatedReplicas   int32 `json:"updatedReplicas,omitempty"`

	ObservedGeneration int64  `json:"observedGeneration,omitempty"`
	ResourceVersion    string `json:"resourceVersion,omitempty"`
	Message            string `json:"message,omitempty"`

	LastObservedTime metav1.Time `json:"lastObservedTime"`
}

type OSHAPlatformStatus struct {
	ObservedGeneration int64             `json:"observedGeneration,omitempty"`
	Phase              OSHAPlatformPhase `json:"phase,omitempty"`
	ReadyComponents    int32             `json:"readyComponents,omitempty"`
	TotalComponents    int32             `json:"totalComponents,omitempty"`

	// +optional
	// +listType=map
	// +listMapKey=kind
	// +listMapKey=namespace
	// +listMapKey=name
	Components []OSHAComponentStatus `json:"components,omitempty"`

	// +optional
	// +listType=map
	// +listMapKey=type
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// +optional
	LastReconcileTime *metav1.Time `json:"lastReconcileTime,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=oshaplatforms,scope=Namespaced,shortName=osha,categories=all
// +kubebuilder:storageversion
// +kubebuilder:printcolumn:name="Policy",type="string",JSONPath=".spec.managementPolicy"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Ready",type="integer",JSONPath=".status.readyComponents"
// +kubebuilder:printcolumn:name="Total",type="integer",JSONPath=".status.totalComponents"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type OSHAPlatform struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec OSHAPlatformSpec `json:"spec"`

	// +optional
	Status OSHAPlatformStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type OSHAPlatformList struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ListMeta `json:"metadata,omitempty"`

	Items []OSHAPlatform `json:"items"`
}

func init() {
	SchemeBuilder.Register(
		&OSHAPlatform{},
		&OSHAPlatformList{},
	)
}
