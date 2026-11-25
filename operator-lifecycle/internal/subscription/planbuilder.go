package subscription

import (
	"context"
	"fmt"
	"strings"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/resolver"

	"sigs.k8s.io/yaml"
)

type DeploymentPlanBuilder struct{}

func NewDeploymentPlanBuilder() *DeploymentPlanBuilder {
	return &DeploymentPlanBuilder{}
}

func (b *DeploymentPlanBuilder) Build(
	ctx context.Context,
	subscription *lifecyclev1alpha1.OperatorSubscription,
	resolution *resolver.Result,
) (*PlanBuildResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if subscription == nil {
		return nil, fmt.Errorf(
			"subscription is required",
		)
	}

	if resolution == nil {
		return nil, fmt.Errorf(
			"resolution result is required",
		)
	}

	image := strings.TrimSpace(
		resolution.Primary.Release.Artifacts.Image,
	)

	if image == "" {
		return nil, fmt.Errorf(
			"resolved release %q has no controller image",
			resolution.Primary.Release.Version,
		)
	}

	namespace := targetNamespace(subscription)

	deploymentName := sanitizeName(
		subscription.Spec.Package,
	)

	manifest, err := deploymentManifest(
		deploymentName,
		namespace,
		image,
	)
	if err != nil {
		return nil, err
	}

	dependencies := make(
		[]lifecyclev1alpha1.ResolvedDependency,
		0,
		len(resolution.Dependencies),
	)

	for _, dependency := range resolution.Dependencies {
		dependencies = append(
			dependencies,
			lifecyclev1alpha1.ResolvedDependency{
				Package: dependency.Package,
				Version: dependency.Release.Version,
				Channel: dependency.Channel,

				Catalog: lifecyclev1alpha1.NamespacedObjectReference{
					Name:      subscription.Spec.Catalog.Name,
					Namespace: subscription.Spec.Catalog.Namespace,
				},
			},
		)
	}

	return &PlanBuildResult{
		Dependencies: dependencies,

		Steps: []lifecyclev1alpha1.InstallPlanStep{
			{
				ID: "operator-deployment",

				Action: lifecyclev1alpha1.InstallStepCreate,

				Resource: lifecyclev1alpha1.ResourceReference{
					Group:     "apps",
					Version:   "v1",
					Kind:      "Deployment",
					Name:      deploymentName,
					Namespace: namespace,
				},

				Manifest: manifest,
			},
		},
	}, nil
}

func deploymentManifest(
	name string,
	namespace string,
	image string,
) (string, error) {
	object := map[string]interface{}{
		"apiVersion": "apps/v1",
		"kind":       "Deployment",

		"metadata": map[string]interface{}{
			"name":      name,
			"namespace": namespace,

			"labels": map[string]interface{}{
				"app.kubernetes.io/name":       name,
				"app.kubernetes.io/managed-by": "osha-lifecycle",
			},
		},

		"spec": map[string]interface{}{
			"replicas": 1,

			"selector": map[string]interface{}{
				"matchLabels": map[string]interface{}{
					"app.kubernetes.io/name": name,
				},
			},

			"template": map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": map[string]interface{}{
						"app.kubernetes.io/name": name,
					},
				},

				"spec": map[string]interface{}{
					"containers": []interface{}{
						map[string]interface{}{
							"name":  "manager",
							"image": image,

							"securityContext": map[string]interface{}{
								"allowPrivilegeEscalation": false,
								"runAsNonRoot":             true,

								"capabilities": map[string]interface{}{
									"drop": []interface{}{
										"ALL",
									},
								},
							},
						},
					},

					"securityContext": map[string]interface{}{
						"runAsNonRoot": true,

						"seccompProfile": map[string]interface{}{
							"type": "RuntimeDefault",
						},
					},
				},
			},
		},
	}

	data, err := yaml.Marshal(object)
	if err != nil {
		return "", fmt.Errorf(
			"marshal Operator Deployment: %w",
			err,
		)
	}

	return string(data), nil
}
