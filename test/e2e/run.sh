#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

NAMESPACE="${NAMESPACE:-osha-lifecycle-system}"
IMAGE="${IMAGE:-osha-operator-lifecycle:e2e}"
TIMEOUT="${TIMEOUT:-180s}"
KEEP_RESOURCES="${KEEP_RESOURCES:-false}"

cd "$ROOT"

log() {
    printf '\n== %s ==\n' "$1"
}

fail() {
    echo "E2E FAILURE: $*" >&2
    exit 1
}

cleanup() {
    if [[ "$KEEP_RESOURCES" == "true" ]]; then
        echo "KEEP_RESOURCES=true; leaving E2E resources installed."
        return
    fi

    log "Cleaning E2E fixtures"

    kubectl delete \
        -k test/e2e/fixtures \
        --ignore-not-found=true \
        >/dev/null 2>&1 || true

    log "Removing lifecycle manager"

    kubectl kustomize config/overlays/development 2>/dev/null |
        kubectl delete \
            -f - \
            --ignore-not-found=true \
            >/dev/null 2>&1 || true
}

trap cleanup EXIT

for command_name in go kubectl controller-gen nerdctl; do
    command -v "$command_name" >/dev/null 2>&1 ||
        fail "required command not found: $command_name"
done

log "Checking Kubernetes API"

kubectl cluster-info >/dev/null

log "Generating code and manifests"

make generate
make manifests

log "Running unit tests"

go test ./...

log "Rendering development manifests"

kubectl kustomize \
    config/overlays/development \
    >/tmp/osha-lifecycle-e2e.yaml

test -s /tmp/osha-lifecycle-e2e.yaml ||
    fail "rendered manifest is empty"

log "Building manager image in containerd k8s.io namespace"

sudo nerdctl \
    --namespace k8s.io \
    build \
    -t "$IMAGE" \
    .

log "Deploying lifecycle manager"

sed \
    "s|image: osha-operator-lifecycle:dev|image: ${IMAGE}|g" \
    /tmp/osha-lifecycle-e2e.yaml |
    kubectl apply -f -

log "Waiting for CRDs"

for crd in \
    operatorpackages.lifecycle.osha.io \
    operatorcatalogs.lifecycle.osha.io \
    operatorsubscriptions.lifecycle.osha.io \
    operatorinstallplans.lifecycle.osha.io
do
    kubectl wait \
        --for=condition=Established \
        "crd/${crd}" \
        --timeout="$TIMEOUT"
done

log "Waiting for manager deployment"

kubectl rollout status \
    deployment/controller-manager \
    -n "$NAMESPACE" \
    --timeout="$TIMEOUT"

log "Checking manager Pod readiness"

kubectl wait \
    --for=condition=Ready \
    pod \
    -l app.kubernetes.io/component=lifecycle-manager \
    -n "$NAMESPACE" \
    --timeout="$TIMEOUT"

log "Applying lifecycle fixtures"

kubectl apply -k test/e2e/fixtures

log "Checking stored lifecycle resources"

kubectl get operatorcatalog e2e-catalog \
    -n "$NAMESPACE"

kubectl get operatorpackage e2e-operator \
    -n "$NAMESPACE"

kubectl get operatorsubscription e2e-subscription \
    -n "$NAMESPACE"

kubectl get operatorinstallplan e2e-install-plan \
    -n "$NAMESPACE"

log "Checking CRD status subresources"

for resource in \
    operatorpackages \
    operatorcatalogs \
    operatorsubscriptions \
    operatorinstallplans
do
    kubectl get \
        --raw="/apis/lifecycle.osha.io/v1alpha1/namespaces/${NAMESPACE}/${resource}" \
        >/dev/null
done

log "Checking lifecycle discovery"

kubectl api-resources \
    --api-group=lifecycle.osha.io

log "Checking manager logs"

kubectl logs \
    -n "$NAMESPACE" \
    deployment/controller-manager \
    -c manager \
    --tail=100

log "E2E framework checks passed"

echo
echo "Validated:"
echo "  - generated CRDs and RBAC"
echo "  - Kustomize rendering"
echo "  - manager image build"
echo "  - manager rollout and readiness"
echo "  - lifecycle API discovery"
echo "  - catalog/package/subscription/install-plan storage"
echo
echo "Not yet validated:"
echo "  - Subscription reconciliation"
echo "  - production PlanBuilder"
echo "  - InstallPlan execution by a controller"
echo "  - upgrade execution and rollback"
