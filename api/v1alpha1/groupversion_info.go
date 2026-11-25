// +kubebuilder:object:generate=true
// +groupName=lifecycle.osha.io
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"

	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

const (
	GroupName   = "lifecycle.osha.io"
	VersionName = "v1alpha1"
)

var (
	SchemeGroupVersion = schema.GroupVersion{
		Group:   GroupName,
		Version: VersionName,
	}

	GroupVersion = SchemeGroupVersion

	SchemeBuilder = &scheme.Builder{
		GroupVersion: SchemeGroupVersion,
	}

	AddToScheme = SchemeBuilder.AddToScheme
)
