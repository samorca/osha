#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"

echo "Removing OSHAPlatform sample..."

kubectl delete \
    -f config/samples/platform_v1alpha1_oshaplatform.yaml \
    --ignore-not-found=true || true

echo "Removing OSHA Operator..."

if kubectl kustomize config/overlays/development >/dev/null 2>&1; then
    kubectl kustomize config/overlays/development | \
        kubectl delete \
        -f - \
        --ignore-not-found=true || true
fi

echo
echo "OSHA Operator removed."
