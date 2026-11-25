#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

IMAGE="${IMAGE:-osha-operator:dev}"
NAMESPACE="${NAMESPACE:-osha-system}"

cd "$ROOT"

echo "======================================"
echo " OSHA Operator development deployment "
echo "======================================"
echo

required_commands=(
    go
    kubectl
    controller-gen
    docker
    ctr
    make
    sudo
)

for command_name in "${required_commands[@]}"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name" >&2
        exit 1
    fi
done

echo "[1/8] Checking Kubernetes..."
kubectl cluster-info >/dev/null

echo "[2/8] Generating DeepCopy code..."
make generate

echo "[3/8] Generating CRD and RBAC..."
make manifests

echo "[4/8] Running unit tests..."
go test ./...

echo "[5/8] Building Operator image..."
make docker-build-k8s \
    IMG="$IMAGE" \
    SUDO="${SUDO:-sudo}"

echo "[6/8] Deploying Operator..."
kubectl kustomize config/overlays/development | \
    sed "s|image: osha-operator:dev|image: ${IMAGE}|g" | \
    kubectl apply -f -

echo "[7/8] Waiting for controller manager..."
kubectl rollout status \
    deployment/controller-manager \
    -n "$NAMESPACE" \
    --timeout=180s

echo "[8/8] Applying OSHAPlatform sample..."
kubectl apply \
    -f config/samples/platform_v1alpha1_oshaplatform.yaml

echo
echo "OSHA Operator deployment completed."
echo

"$ROOT/scripts/verify.sh"
