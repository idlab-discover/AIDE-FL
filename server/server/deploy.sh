#!/bin/bash

set -e

DIR=$(dirname "$0")

if [ "$1" == "prod" ]; then
    TAG="latest"
else
    TAG="dev"
fi

echo "Deploying environment with tag: $TAG"

# Build and push flower-serverapp image
docker build -f ${DIR}/../../flower/serverapp.Dockerfile --tag ghcr.io/martienskienum1/flower-serverapp:$TAG ${DIR}/../../flower
docker push ghcr.io/martienskienum1/flower-serverapp:$TAG

# Build and push server-logic image
docker build --tag ghcr.io/martienskienum1/server-logic:$TAG ${DIR}/server-logic
docker push ghcr.io/martienskienum1/server-logic:$TAG

# Recreate flower-superlink + flower-serverapp + server-logic deployment
kubectl delete deployment server-deployment --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/server-deployment.yaml
else
    kubectl apply -f ${DIR}/server-deployment-dev.yaml
fi