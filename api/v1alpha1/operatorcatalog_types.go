package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +kubebuilder:validation:Enum=OCI;HTTP;ConfigMap
type CatalogSourceType string

const (
	CatalogSourceOCI       CatalogSourceType = "OCI"
	CatalogSourceHTTP      CatalogSourceType = "HTTP"
	CatalogSourceConfigMap CatalogSourceType = "ConfigMap"
)

// +kubebuilder:validation:Enum=Ready;Syncing;Degraded;Unavailable
type OperatorCatalogPhase string

const (
	OperatorCatalogPhaseReady       OperatorCatalogPhase = "Ready"
	OperatorCatalogPhaseSyncing     OperatorCatalogPhase = "Syncing"
	OperatorCatalogPhaseDegraded    OperatorCatalogPhase = "Degraded"
	OperatorCatalogPhaseUnavailable OperatorCatalogPhase = "Unavailable"
)

type OCIRepositorySource struct {
	// Image identifies the OCI catalog image.
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// Digest optionally pins the catalog to an immutable image digest.
	// +optional
	Digest string `json:"digest,omitempty"`

	// PullSecretRef references an image-pull Secret in the catalog namespace.
	// +optional
	PullSecretRef string `json:"pullSecretRef,omitempty"`
}

type HTTPRepositorySource struct {
	// URL identifies the remote catalog endpoint.
	// +kubebuilder:validation:Pattern=`^https://`
	URL string `json:"url"`

	// AuthSecretRef optionally references authentication material.
	// +optional
	AuthSecretRef string `json:"authSecretRef,omitempty"`

	// CABundleSecretRef optionally references custom CA material.
	// +optional
	CABundleSecretRef string `json:"caBundleSecretRef,omitempty"`
}

type ConfigMapRepositorySource struct {
	// Name identifies the ConfigMap containing catalog metadata.
	// +kubebuilder:validation:MinLength=1
	Name string `json:"name"`

	// Namespace defaults to the OperatorCatalog namespace.
	// +optional
	Namespace string `json:"namespace,omitempty"`

	// Key identifies the ConfigMap data key containing catalog metadata.
	// +kubebuilder:default=catalog.yaml
	Key string `json:"key,omitempty"`
}

type CatalogSource struct {
	Type CatalogSourceType `json:"type"`

	// +optional
	OCI *OCIRepositorySource `json:"oci,omitempty"`

	// +optional
	HTTP *HTTPRepositorySource `json:"http,omitempty"`

	// +optional
	ConfigMap *ConfigMapRepositorySource `json:"configMap,omitempty"`
}

type CatalogSyncPolicy struct {
	// Interval specifies how frequently the catalog should be refreshed.
	// Examples: 5m, 30m, 1h.
	// +kubebuilder:default="10m"
	// +kubebuilder:validation:Pattern=`^[1-9][0-9]*(s|m|h)$`
	Interval string `json:"interval,omitempty"`

	// Suspend disables automatic synchronization without deleting the catalog.
	// +kubebuilder:default=false
	Suspend bool `json:"suspend,omitempty"`
}

type OperatorCatalogSpec struct {
	// DisplayName is the human-readable catalog name.
	// +kubebuilder:validation:MinLength=1
	DisplayName string `json:"displayName"`

	// Publisher identifies the organization responsible for this catalog.
	// +kubebuilder:validation:MinLength=1
	Publisher string `json:"publisher"`

	Source CatalogSource `json:"source"`

	// +optional
	SyncPolicy CatalogSyncPolicy `json:"syncPolicy,omitempty"`

	// Priority allows the resolver to prefer one catalog over another.
	// Higher values represent stronger preference.
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=-1000
	// +kubebuilder:validation:Maximum=1000
	Priority int32 `json:"priority,omitempty"`

	// Enabled controls whether the resolver may select packages from this catalog.
	// +kubebuilder:default=true
	Enabled bool `json:"enabled,omitempty"`
}

type OperatorCatalogStatus struct {
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	Phase OperatorCatalogPhase `json:"phase,omitempty"`

	// ResolvedReference records the exact catalog reference observed during
	// the latest successful synchronization.
	// +optional
	ResolvedReference string `json:"resolvedReference,omitempty"`

	// PackageCount reports the number of packages indexed from the latest
	// successful catalog synchronization.
	// +optional
	PackageCount int32 `json:"packageCount,omitempty"`

	// LastSyncTime records the latest completed synchronization.
	// +optional
	LastSyncTime *metav1.Time `json:"lastSyncTime,omitempty"`

	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:path=operatorcatalogs,scope=Namespaced,shortName=ocat,categories=osha
// +kubebuilder:printcolumn:name="Type",type="string",JSONPath=".spec.source.type"
// +kubebuilder:printcolumn:name="Publisher",type="string",JSONPath=".spec.publisher"
// +kubebuilder:printcolumn:name="Phase",type="string",JSONPath=".status.phase"
// +kubebuilder:printcolumn:name="Packages",type="integer",JSONPath=".status.packageCount"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type OperatorCatalog struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec OperatorCatalogSpec `json:"spec"`

	// +optional
	Status OperatorCatalogStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true
type OperatorCatalogList struct {
	metav1.TypeMeta `json:",inline"`

	metav1.ListMeta `json:"metadata,omitempty"`

	Items []OperatorCatalog `json:"items"`
}

func init() {
	SchemeBuilder.Register(
		&OperatorCatalog{},
		&OperatorCatalogList{},
	)
}
