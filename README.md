# OSHA Operator Lifecycle Manager

This Go module contains OSHA's Kubernetes-native Operator lifecycle manager.
It defines namespaced custom resources for catalogs, packages, subscriptions,
and install plans, resolves a requested package release, and executes an
approved install plan through the Kubernetes API.

It is a separate controller from the platform observer in
[`../operators`](../operators/README.md). The platform observer watches
OSHA workloads; this lifecycle manager manages Operator package installation
plans.

## Current Implementation and Limits

The API group/version is `lifecycle.osha.io/v1alpha1`. The module defines four
custom resources:

| Resource | Purpose |
| --- | --- |
| `OperatorCatalog` | Names a catalog and describes its source and sync policy. |
| `OperatorPackage` | Declares releases, channels, artifacts, API references, and dependencies. |
| `OperatorSubscription` | Requests a package/channel/version in a target namespace. |
| `OperatorInstallPlan` | Records a resolved install or upgrade proposal and per-step status. |

The current implementation is narrower than the API schema:

- A subscription checks that its referenced catalog exists and is enabled,
  then reads the `OperatorPackage` custom resource directly from the
  subscription namespace. It does not currently synchronize catalog data
  from OCI, HTTP, or ConfigMap sources. Catalog source and sync fields are
  declarative API fields today.
- The sample `OperatorCatalog` references a ConfigMap that is not included in
  the sample kustomization. That reference is not fetched by the current
  reconciler; the package custom resource is the data used for resolution.
- The current `DeploymentPlanBuilder` creates one Deployment from
  `spec.releases[].artifacts.image`. It does not install a bundle, CRDs,
  ServiceAccounts, Roles, or other resources from `bundleImage` or
  `providedAPIs`.
- The generated lifecycle ClusterRole currently grants install-time
  permissions for Deployments only. The generic plan schema and executor
  support more step types, but RBAC and the production plan builder do not
  make those operations generally available.
- The E2E script validates generated APIs, manager deployment, and fixture
  storage. Its own summary marks subscription reconciliation, production plan
  building, install-plan execution, and upgrade/rollback execution as not yet
  validated end-to-end.

Treat this as an actively developed lifecycle controller, not a drop-in
replacement for a mature Operator Lifecycle Manager. Review every install
plan, image, target namespace, and permission before approval.

## Lifecycle Flow

1. Create an enabled `OperatorCatalog` and an `OperatorPackage` in Kubernetes.
2. Create an `OperatorSubscription` referring to the catalog and package.
3. The subscription reconciler resolves the requested channel and version,
   then creates an `OperatorInstallPlan`.
4. A `Manual` plan remains in `WaitingForApproval` until
   `spec.approved: true`. An `Automatic` plan is still evaluated by the
   lifecycle policy; automatic approval does not bypass policy.
5. The install-plan reconciler executes permitted steps and records
   `Complete` or `Failed` plus individual step statuses.
6. The subscription observes the plan and updates its installed version and
   phase.

Subscriptions default to `Manual` approval. A paused subscription does not
create or execute new plans. Target namespace defaults to the subscription
namespace unless `spec.targetNamespace` is set.

## Policy and Safety

The manager's default policy allows Install and Upgrade actions, but denies
Rollback, resource Delete, destructive steps, and cluster-scoped resources.
It allows up to 100 steps per plan. Every step must target the plan's target
namespace; an empty step namespace is treated as cluster-scoped and denied by
default. The policy's namespace allowlist is empty by default, which means it
does not impose an additional fixed namespace list.

Manual approval is explicit on each plan. To approve a plan, inspect it first
and patch only the intended resource. The manager's service account should be
kept scoped to the permissions required by the deployed feature set.

## Prerequisites

- Go version from `go.mod` (currently Go 1.27.3)
- `kubectl` configured for the target Kubernetes cluster
- `controller-gen` on `PATH` for code/manifests generation
- Docker or another OCI-compatible container builder for images
- `ctr` and `sudo` for `make docker-build-k8s`
- `nerdctl`, `sudo`, and a reachable cluster for `make test-e2e`

`make verify` checks Go, kubectl, cluster access, controller-gen, Docker, and
containerd tooling.

## Develop and Test

Run commands from this directory:

```bash
cd operator-lifecycle
go test ./...
go vet ./...
make generate
make manifests
make build
```

The Makefile also provides `make test`, `make test-verbose`, `make fmt`, and
`make tidy`. `make build` generates DeepCopy code, formats Go files, runs vet,
and builds `bin/manager`. `make run` runs the manager against the current
kubeconfig.

`make manifests` regenerates CRDs and RBAC under `config/crd` and
`config/rbac`; review generated changes with API or controller changes.

The optional cluster E2E target is:

```bash
make test-e2e IMG=osha-operator-lifecycle:e2e
```

It uses `sudo nerdctl --namespace k8s.io`, applies resources to the current
cluster, and removes its fixtures and development manager on exit unless
`KEEP_RESOURCES=true` is set. It is not a read-only test.

## Build and Deploy

Build the development manager image:

```bash
make docker-build IMG=osha-operator-lifecycle:dev
```

To import it into Kubernetes containerd:

```bash
make docker-build-k8s IMG=osha-operator-lifecycle:dev
```

The development overlay uses `imagePullPolicy: Never`; import the image on
the cluster node before deployment. From this directory, install CRDs, RBAC,
and the manager with:

```bash
make deploy-dev
```

The deployment uses namespace `osha-lifecycle-system` and one replica in the
development overlay. Verify it with:

```bash
make status
make logs
kubectl get crd | grep lifecycle.osha.io
```

`make render-dev` and `make render-prod` render manifests without applying
them. The production overlay uses
`ghcr.io/samorca/osha-operator-lifecycle:v0.1.0`; publish the corresponding
image before using `make deploy-prod`.

## Install the Sample

Ensure the target namespace exists. The sample targets `osha-system`:

```bash
kubectl create namespace osha-system \
  --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -k config/samples
```

The sample package is `osha-operator` version `0.1.0`, on channel `stable`,
and references image `osha-operator:dev`. Make sure that image is available to
the cluster or edit `config/samples/package.yaml` to use a reachable image.
The sample catalog's ConfigMap source is descriptive only in the current
implementation; no ConfigMap is required for subscription resolution.

Inspect the resources and status:

```bash
kubectl get operatorcatalog,operatorpackage,operatorsubscription \
  -n osha-lifecycle-system
kubectl get operatorinstallplans -n osha-lifecycle-system
kubectl describe operatorsubscription osha-operator \
  -n osha-lifecycle-system
```

For a manual plan, inspect the proposed target and steps, then approve the
specific plan:

```bash
kubectl get operatorinstallplans -n osha-lifecycle-system -o yaml
kubectl patch operatorinstallplan <plan-name> \
  -n osha-lifecycle-system \
  --type=merge \
  -p '{"spec":{"approved":true}}'
```

Watch plan and subscription status:

```bash
kubectl get operatorinstallplans,operatorsubscriptions \
  -n osha-lifecycle-system -w
```

## Custom Resources

### OperatorCatalog

`spec.displayName`, `publisher`, `source`, `enabled`, optional `priority`, and
`syncPolicy` describe a catalog. The API accepts `OCI`, `HTTP`, and `ConfigMap`
source shapes. Source retrieval and periodic synchronization are not
implemented yet; current subscription resolution only checks that the
catalog exists and is enabled.

### OperatorPackage

`spec.releases` holds versions and artifacts. `spec.channels` maps channels to
ordered release entries, with optional `replaces` and `skips` upgrade edges.
`defaultChannel` is used when a subscription omits `channel`. Releases may
declare maturity, provided/required APIs, dependencies, release notes, and
published time. The current plan builder uses the release image and does not
consume the optional bundle image or API declarations.

### OperatorSubscription

Selects a package and catalog, optional channel and version constraint,
approval mode, target namespace, starting version, and pause state. Status
reports the selected channel/version, installed version, current plan, phase,
and conditions.

### OperatorInstallPlan

Contains the resolved action/version, target namespace, approval state,
dependencies, resolution hash, and ordered steps. Step actions are Create,
Update, Patch, and Delete. The default policy denies Delete and destructive
steps. Status records plan phase, timestamps, conditions, and each step's
result.

## Operations and Cleanup

```bash
kubectl logs -n osha-lifecycle-system deployment/controller-manager -c manager -f
make undeploy-dev
```

Delete sample resources explicitly if they should not remain:

```bash
kubectl delete -k config/samples --ignore-not-found=true
```

`make uninstall` removes the lifecycle CRDs. Kubernetes deletes the custom
resources when their CRDs are removed; export anything needed before
uninstalling. The target namespace `osha-system` is separate and is not
created by the lifecycle sample.

## Troubleshooting

- **Subscription is `Degraded`:** inspect its conditions and confirm the
  referenced catalog is enabled and the `OperatorPackage` exists in the
  subscription namespace.
- **No install plan appears:** check package channel entries, release
  versions, version constraints, and manager logs.
- **Plan waits for approval:** this is expected for `approval: Manual`; inspect
  the plan before setting `spec.approved: true`.
- **Plan is denied:** check action, step count, target namespace, cluster-scope,
  Delete, and `destructive` policy restrictions.
- **Create/update fails:** confirm the manager RBAC allows the resource and
  the rendered manifest's group, version, kind, name, and namespace match the
  plan step.
- **Image cannot start:** verify the referenced image exists in the cluster's
  registry/runtime and is compatible with the target platform.