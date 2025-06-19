#!/bin/bash

set -e

DIR=$(dirname "$0")

if [ "$1" == "prod" ]; then
    TAG="latest"
else
    TAG="dev"
fi

echo "Deploying environment with tag: $TAG"

# Build and push orchestrator image
docker build --tag ghcr.io/martienskienum1/orchestrator:$TAG ${DIR}
docker push ghcr.io/martienskienum1/orchestrator:$TAG

# Recreate orchestrator daemonset
kubectl delete daemonset orchestrator-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/orchestrator-daemonset.yaml
else
    kubectl apply -f ${DIR}/orchestrator-daemonset-dev.yaml
fi