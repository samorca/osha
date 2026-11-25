package installer

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strings"
	"time"

	lifecyclev1alpha1 "github.com/samorca/osha/operator-lifecycle/api/v1alpha1"
	"github.com/samorca/osha/operator-lifecycle/internal/policy"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

var (
	ErrClientRequired       = errors.New("Kubernetes client is required")
	ErrPolicyRequired       = errors.New("policy engine is required")
	ErrPlanRequired         = errors.New("install plan is required")
	ErrPlanDenied           = errors.New("install plan denied")
	ErrApprovalRequired     = errors.New("install plan requires approval")
	ErrDependencyCycle      = errors.New("install step dependency cycle")
	ErrManifestRequired     = errors.New("resource manifest is required")
	ErrManifestInvalid      = errors.New("resource manifest is invalid")
	ErrManifestMismatch     = errors.New("manifest does not match resource reference")
	ErrUnsupportedOperation = errors.New("unsupported install operation")
)

type StepResult struct {
	ID        string
	Phase     lifecyclev1alpha1.InstallStepPhase
	Message   string
	StartedAt time.Time
	EndedAt   time.Time
}

type Result struct {
	StartedAt time.Time
	EndedAt   time.Time
	Steps     []StepResult
}

type Executor struct {
	client client.Client
	policy *policy.Engine
}

func New(
	k8sClient client.Client,
	policyEngine *policy.Engine,
) (*Executor, error) {
	if k8sClient == nil {
		return nil, ErrClientRequired
	}

	if policyEngine == nil {
		return nil, ErrPolicyRequired
	}

	return &Executor{
		client: k8sClient,
		policy: policyEngine,
	}, nil
}

func (e *Executor) Execute(
	ctx context.Context,
	plan *lifecyclev1alpha1.OperatorInstallPlan,
) (*Result, error) {
	if plan == nil {
		return nil, ErrPlanRequired
	}

	evaluation := e.policy.Evaluate(plan)

	switch evaluation.Decision {
	case policy.DecisionRequireApproval:
		return nil, fmt.Errorf(
			"%w: %s",
			ErrApprovalRequired,
			evaluation.Message,
		)

	case policy.DecisionDeny:
		return nil, fmt.Errorf(
			"%w: %s",
			ErrPlanDenied,
			evaluation.Message,
		)

	case policy.DecisionAllow:
		// Continue.

	default:
		return nil, fmt.Errorf(
			"%w: unknown policy decision %q",
			ErrPlanDenied,
			evaluation.Decision,
		)
	}

	orderedSteps, err := orderSteps(plan.Spec.Steps)
	if err != nil {
		return nil, err
	}

	result := &Result{
		StartedAt: time.Now().UTC(),
		Steps: make(
			[]StepResult,
			0,
			len(orderedSteps),
		),
	}

	for _, step := range orderedSteps {
		if err := ctx.Err(); err != nil {
			result.EndedAt = time.Now().UTC()
			return result, err
		}

		stepResult := StepResult{
			ID:        step.ID,
			Phase:     lifecyclev1alpha1.InstallStepPhaseRunning,
			StartedAt: time.Now().UTC(),
		}

		err := e.executeStep(ctx, step)

		stepResult.EndedAt = time.Now().UTC()

		if err != nil {
			stepResult.Phase =
				lifecyclev1alpha1.InstallStepPhaseFailed
			stepResult.Message = err.Error()

			result.Steps = append(result.Steps, stepResult)
			result.EndedAt = time.Now().UTC()

			return result, fmt.Errorf(
				"execute step %q: %w",
				step.ID,
				err,
			)
		}

		stepResult.Phase =
			lifecyclev1alpha1.InstallStepPhaseComplete
		stepResult.Message = "step completed"

		result.Steps = append(result.Steps, stepResult)
	}

	result.EndedAt = time.Now().UTC()

	return result, nil
}

func (e *Executor) executeStep(
	ctx context.Context,
	step lifecyclev1alpha1.InstallPlanStep,
) error {
	switch step.Action {
	case lifecyclev1alpha1.InstallStepCreate:
		return e.create(ctx, step)

	case lifecyclev1alpha1.InstallStepUpdate:
		return e.update(ctx, step)

	case lifecyclev1alpha1.InstallStepPatch:
		return e.patch(ctx, step)

	case lifecyclev1alpha1.InstallStepDelete:
		return e.delete(ctx, step)

	default:
		return fmt.Errorf(
			"%w: %q",
			ErrUnsupportedOperation,
			step.Action,
		)
	}
}

func (e *Executor) create(
	ctx context.Context,
	step lifecyclev1alpha1.InstallPlanStep,
) error {
	obj, err := decodeManifest(step)
	if err != nil {
		return err
	}

	err = e.client.Create(ctx, obj)
	if err == nil {
		return nil
	}

	if !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf(
			"create %s: %w",
			resourceDescription(step.Resource),
			err,
		)
	}

	var existing unstructured.Unstructured
	setReference(&existing, step.Resource)

	if err := e.client.Get(
		ctx,
		client.ObjectKeyFromObject(obj),
		&existing,
	); err != nil {
		return fmt.Errorf(
			"get existing %s: %w",
			resourceDescription(step.Resource),
			err,
		)
	}

	if existing.GetGeneration() > 0 {
		return nil
	}

	return nil
}

func (e *Executor) update(
	ctx context.Context,
	step lifecyclev1alpha1.InstallPlanStep,
) error {
	desired, err := decodeManifest(step)
	if err != nil {
		return err
	}

	var current unstructured.Unstructured
	setReference(&current, step.Resource)

	key := client.ObjectKey{
		Name:      step.Resource.Name,
		Namespace: step.Resource.Namespace,
	}

	if err := e.client.Get(ctx, key, &current); err != nil {
		return fmt.Errorf(
			"get current %s: %w",
			resourceDescription(step.Resource),
			err,
		)
	}

	desired.SetResourceVersion(current.GetResourceVersion())

	if err := e.client.Update(ctx, desired); err != nil {
		return fmt.Errorf(
			"update %s: %w",
			resourceDescription(step.Resource),
			err,
		)
	}

	return nil
}

func (e *Executor) patch(
	ctx context.Context,
	step lifecyclev1alpha1.InstallPlanStep,
) error {
	patchData, err := manifestJSON(step.Manifest)
	if err != nil {
		return err
	}

	var obj unstructured.Unstructured
	setReference(&obj, step.Resource)

	key := client.ObjectKey{
		Name:      step.Resource.Name,
		Namespace: step.Resource.Namespace,
	}

	if err := e.client.Get(ctx, key, &obj); err != nil {
		return fmt.Errorf(
			"get patch target %s: %w",
			resourceDescription(step.Resource),
			err,
		)
	}

	if err := e.client.Patch(
		ctx,
		&obj,
		client.RawPatch(
			k8stypes.MergePatchType,
			patchData,
		),
	); err != nil {
		return fmt.Errorf(
			"patch %s: %w",
			resourceDescription(step.Resource),
			err,
		)
	}

	return nil
}

func (e *Executor) delete(
	ctx context.Context,
	step lifecyclev1alpha1.InstallPlanStep,
) error {
	var obj unstructured.Unstructured
	setReference(&obj, step.Resource)

	err := e.client.Delete(ctx, &obj)

	if err == nil || apierrors.IsNotFound(err) {
		return nil
	}

	return fmt.Errorf(
		"delete %s: %w",
		resourceDescription(step.Resource),
		err,
	)
}

func decodeManifest(
	step lifecyclev1alpha1.InstallPlanStep,
) (*unstructured.Unstructured, error) {
	if strings.TrimSpace(step.Manifest) == "" {
		return nil, ErrManifestRequired
	}

	jsonData, err := manifestJSON(step.Manifest)
	if err != nil {
		return nil, err
	}

	var object map[string]interface{}

	if err := yaml.Unmarshal(jsonData, &object); err != nil {
		return nil, fmt.Errorf(
			"%w: %v",
			ErrManifestInvalid,
			err,
		)
	}

	result := &unstructured.Unstructured{
		Object: object,
	}

	if err := validateManifestReference(
		result,
		step.Resource,
	); err != nil {
		return nil, err
	}

	return result, nil
}

func manifestJSON(manifest string) ([]byte, error) {
	if strings.TrimSpace(manifest) == "" {
		return nil, ErrManifestRequired
	}

	jsonData, err := yaml.YAMLToJSON([]byte(manifest))
	if err != nil {
		return nil, fmt.Errorf(
			"%w: %v",
			ErrManifestInvalid,
			err,
		)
	}

	return jsonData, nil
}

func validateManifestReference(
	obj *unstructured.Unstructured,
	ref lifecyclev1alpha1.ResourceReference,
) error {
	gvk := obj.GroupVersionKind()

	if gvk.Group != ref.Group ||
		gvk.Version != ref.Version ||
		gvk.Kind != ref.Kind ||
		obj.GetName() != ref.Name ||
		obj.GetNamespace() != ref.Namespace {

		return fmt.Errorf(
			"%w: manifest=%s/%s/%s %s/%s reference=%s/%s/%s %s/%s",
			ErrManifestMismatch,
			gvk.Group,
			gvk.Version,
			gvk.Kind,
			obj.GetNamespace(),
			obj.GetName(),
			ref.Group,
			ref.Version,
			ref.Kind,
			ref.Namespace,
			ref.Name,
		)
	}

	return nil
}

func setReference(
	obj *unstructured.Unstructured,
	ref lifecyclev1alpha1.ResourceReference,
) {
	obj.SetGroupVersionKind(
		schema.GroupVersionKind{
			Group:   ref.Group,
			Version: ref.Version,
			Kind:    ref.Kind,
		},
	)

	obj.SetName(ref.Name)
	obj.SetNamespace(ref.Namespace)
}

func orderSteps(
	steps []lifecyclev1alpha1.InstallPlanStep,
) ([]lifecyclev1alpha1.InstallPlanStep, error) {
	stepsByID := make(
		map[string]lifecyclev1alpha1.InstallPlanStep,
		len(steps),
	)

	indegree := make(map[string]int, len(steps))
	dependents := make(map[string][]string, len(steps))

	for _, step := range steps {
		stepsByID[step.ID] = step

		if _, exists := indegree[step.ID]; !exists {
			indegree[step.ID] = 0
		}
	}

	for _, step := range steps {
		for _, dependencyID := range step.DependsOn {
			if _, exists := stepsByID[dependencyID]; !exists {
				return nil, fmt.Errorf(
					"step %q depends on unknown step %q",
					step.ID,
					dependencyID,
				)
			}

			indegree[step.ID]++
			dependents[dependencyID] = append(
				dependents[dependencyID],
				step.ID,
			)
		}
	}

	ready := make([]string, 0)

	for id, degree := range indegree {
		if degree == 0 {
			ready = append(ready, id)
		}
	}

	sort.Strings(ready)

	ordered := make(
		[]lifecyclev1alpha1.InstallPlanStep,
		0,
		len(steps),
	)

	for len(ready) > 0 {
		id := ready[0]
		ready = ready[1:]

		ordered = append(
			ordered,
			stepsByID[id],
		)

		for _, dependentID := range dependents[id] {
			indegree[dependentID]--

			if indegree[dependentID] == 0 {
				ready = append(ready, dependentID)
				sort.Strings(ready)
			}
		}
	}

	if len(ordered) != len(steps) {
		return nil, ErrDependencyCycle
	}

	return ordered, nil
}

func resourceDescription(
	ref lifecyclev1alpha1.ResourceReference,
) string {
	name := ref.Name

	if ref.Namespace != "" {
		name = ref.Namespace + "/" + name
	}

	return fmt.Sprintf(
		"%s/%s/%s %s",
		ref.Group,
		ref.Version,
		ref.Kind,
		name,
	)
}
