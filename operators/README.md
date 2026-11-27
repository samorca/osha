# OSHA Kubernetes Operator

The OSHA Operator is a Kubernetes controller for the namespaced
`OSHAPlatform` custom resource. It observes selected Deployments, StatefulSets,
and DaemonSets, then writes their readiness and replica information to the
`OSHAPlatform` status.

This module is separate from the OSHA application and from the Operator
Lifecycle Manager in [`../operator-lifecycle`](../operator-lifecycle/README.md).

## Current Scope

The controller is currently **observe-only**. It reads workload status and
updates `OSHAPlatform.status`; it does not create, scale, restart, patch, or
delete workloads.

The API already defines `managementPolicy`, `approval`, `paused`, and
`desiredReplicas` fields. At present, `managementPolicy`, `approval`, and
`paused` do not enable or gate workload mutations. `desiredReplicas` is used as
the expected replica count for readiness reporting only. Use this controller
for monitoring and status reporting, not automated remediation.

## Architecture

- Go module: `github.com/samorca/osha/operators`
- API group/version: `platform.osha.io/v1alpha1`
- Custom resource: `OSHAPlatform` (`oshaplatforms`, short name `osha`)
- Controller entry point: `cmd/main.go`
- Reconciler: `internal/controller/oshaplatform_controller.go`
- CRDs and RBAC are generated from API and controller markers with
	`controller-gen`.

The manager watches `OSHAPlatform` resources and reads apps/v1 Deployments,
StatefulSets, and DaemonSets. A namespaced `OSHAPlatform` can reference
workloads in its own namespace or explicitly name another namespace. Its
ClusterRole grants read-only access to those workloads and status updates to
`OSHAPlatform` resources.

## `OSHAPlatform` Resource

The API accepts these spec fields:

| Field | Meaning |
| --- | --- |
| `managementPolicy` | `ObserveOnly` (default) or `Reconcile`. Reconciliation currently observes in either mode. |
| `components` | One to 100 workload references. Each has a name, kind, optional namespace, `required`, and optional `desiredReplicas`. |
| `observationIntervalSeconds` | Requeue interval in seconds; defaults to 30 and is limited to 5-3600. |
| `approval` | Approval-related policy fields reserved for future mutation behavior; they do not currently gate reconciliation. |
| `paused` | Present in the API, but the controller currently continues observations when it is true. |

Supported component kinds are `Deployment`, `StatefulSet`, and `DaemonSet`.
An omitted component namespace defaults to the `OSHAPlatform` namespace. A
component with `required: true` contributes to the overall platform phase.

The controller reports the platform phase (`Pending`, `Discovering`,
`Progressing`, `Ready`, `Degraded`, or `Terminating`), ready/total component
counts, observed replica counts, resource versions, messages, conditions, and
observation times in `.status`.

The repository sample is
[`config/samples/platform_v1alpha1_oshaplatform.yaml`](config/samples/platform_v1alpha1_oshaplatform.yaml).
It expects workloads named `osha-api`, `osha-runtime`, and `osha-worker` in
namespace `osha`; edit the component references to match your cluster.

## Prerequisites

- Go version from `go.mod` (currently Go 1.27.3)
- `kubectl` configured for the target cluster
- `controller-gen` on `PATH` for generation, tests that depend on generated
	files, and deployment targets
- A container build tool for image builds; `ctr` and `sudo` are additionally
	used by `make docker-build-k8s`

`make verify` checks the local Go, Kubernetes, controller-gen, and container
tooling. It also checks `ctr`.

## Develop and Test

Run module commands from this directory:

```bash
cd operators
go test ./...
go vet ./...
make generate
make manifests
make build
```

Equivalent Make targets include `make test`, `make vet`, and `make build`.
`make build` generates code, formats Go sources, runs vet, and writes
`bin/manager`. `make run` performs generation, formatting, and vet before
running the manager against the current kubeconfig.

`make manifests` regenerates CRDs and RBAC under `config/crd` and
`config/rbac`. Review those generated changes along with API changes.

## Build an Image

```bash
make docker-build IMG=osha-operator:dev
```

To make the image available in a Kubernetes node's containerd image store:

```bash
make docker-build-k8s IMG=osha-operator:dev
```

That target builds with the configured container tool, saves the image, and
imports it into the `k8s.io` containerd namespace using `ctr` and `sudo`. The
development overlay uses `imagePullPolicy: Never`, so the image must already
be present on the node that runs the manager.

For a registry workflow, tag and push the image with `make docker-push IMG=...`
after building it. The production overlay currently points to
`ghcr.io/samorca/osha-operator:latest`; update the overlay to a release tag or
digest before a production rollout.

## Install and Run

From this directory, the development deployment generates manifests, renders
the development overlay, and applies it to the current cluster:

```bash
make deploy-dev
```

The overlay installs the CRD, service account, RBAC, and manager in namespace
`osha-system`. To check the deployment:

```bash
make status
make logs
kubectl get oshaplatforms -A
```

Apply the sample after editing its workload references:

```bash
kubectl apply -k config/samples
kubectl get oshaplatform osha -n osha-system -o yaml
kubectl describe oshaplatform osha -n osha-system
```

`status.phase` and `status.components` show the latest observation. The
controller requeues at the configured interval; a `NotFound` component or a
required workload with insufficient ready replicas prevents the platform from
reporting ready.

For a registry-backed production image, render and apply the production
overlay with:

```bash
make deploy-production
```

`make render-dev` and `make render-production` print the rendered manifests
without applying them. `make undeploy-dev` removes the development overlay.
`make uninstall` deletes the CRDs; deleting a CRD also removes its custom
resources, so export any resources you need first.

To install only the CRDs, use `make install`; this still runs manifest
generation. The manager must be deployed separately.

## Troubleshooting

- **A component is `NotFound`:** check its name, kind, and namespace. An empty
	component namespace means the `OSHAPlatform` namespace, not the sample's
	`osha` application namespace.
- **A workload remains `Progressing`:** inspect its observed generation,
	desired replicas, ready replicas, available replicas, and updated replicas.
- **No status changes:** check `make logs`, manager Pod readiness, CRD
	installation, and the controller service account's read/status RBAC.
- **Image pull or startup failure:** ensure the image name in the selected
	overlay is available to the cluster and matches the built or pushed image.

## Security Notes

The current operator is read-only with respect to workloads. Its generated
ClusterRole grants `get`, `list`, and `watch` on Deployments, StatefulSets, and
DaemonSets; it can update the status subresource of `OSHAPlatform`. Do not
interpret the future-facing approval fields as an authorization control for
mutations that this controller does not currently implement.
