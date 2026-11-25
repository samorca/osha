// +kubebuilder:object:generate=true
// +groupName=platform.osha.io
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"

	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

const (
	GroupName        = "platform.osha.io"
	GroupVersionName = "v1alpha1"
)

var (
	SchemeGroupVersion = schema.GroupVersion{
		Group:   GroupName,
		Version: GroupVersionName,
	}

	GroupVersion = SchemeGroupVersion

	SchemeBuilder = &scheme.Builder{
		GroupVersion: SchemeGroupVersion,
	}

	AddToScheme = SchemeBuilder.AddToScheme
)
