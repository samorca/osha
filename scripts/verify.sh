#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${NAMESPACE:-osha-system}"

cd "$ROOT"

required_commands=(
    go
    kubectl
    controller-gen
)

for command_name in "${required_commands[@]}"; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "ERROR: required command not found: $command_name" >&2
        exit 1
    fi
done

echo "== Go =="
go version

echo
echo "== kubectl =="
kubectl version --client

echo
echo "== controller-gen =="
controller-gen --version

echo
echo "== Kubernetes cluster =="
kubectl cluster-info >/dev/null
echo "Cluster API reachable"

echo
echo "== Current context =="
kubectl config current-context

echo
echo "== Operator CRD =="

if kubectl get crd oshaplatforms.platform.osha.io >/dev/null 2>&1; then
    echo "OSHAPlatform CRD installed"
else
    echo "OSHAPlatform CRD not installed"
fi

echo
echo "== Operator namespace =="

if kubectl get namespace "$NAMESPACE" >/dev/null 2>&1; then
    kubectl get namespace "$NAMESPACE"
else
    echo "Namespace $NAMESPACE not installed"
fi

echo
echo "== Operator workload =="

if kubectl get deployment controller-manager \
    -n "$NAMESPACE" >/dev/null 2>&1; then

    kubectl get deployment controller-manager \
        -n "$NAMESPACE" \
        -o wide

    echo
    kubectl get pods \
        -n "$NAMESPACE" \
        -o wide
else
    echo "Operator Deployment not installed"
fi

echo
echo "== OSHAPlatform =="

if kubectl get oshaplatform osha \
    -n "$NAMESPACE" >/dev/null 2>&1; then

    kubectl get oshaplatform osha \
        -n "$NAMESPACE"

    echo
    kubectl get oshaplatform osha \
        -n "$NAMESPACE" \
        -o jsonpath='Phase: {.status.phase}{"\n"}Ready: {.status.readyComponents}/{.status.totalComponents}{"\n"}'
else
    echo "OSHAPlatform sample not installed"
fi

echo
echo "Verification complete."
