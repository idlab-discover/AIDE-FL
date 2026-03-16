#!/bin/bash

set -e

DIR=$(dirname "$0")
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/str-gen}"

if [ "$1" == "prod" ]; then
    TAG="latest"
else
    TAG="dev"
fi

echo "Deploying environment with tag: $TAG"

# Build and push flower-serverapp image
docker build -f ${DIR}/../../flower/serverapp.Dockerfile --tag ${IMAGE_REPO}/flower-serverapp:$TAG ${DIR}/../../flower
docker push ${IMAGE_REPO}/flower-serverapp:$TAG

# Build and push server-logic image
docker build --tag ${IMAGE_REPO}/server-logic:$TAG ${DIR}/server-logic
docker push ${IMAGE_REPO}/server-logic:$TAG

# Recreate flower-superlink + flower-serverapp + server-logic deployment
kubectl delete deployment server-deployment --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/server-deployment.yaml
else
    kubectl apply -f ${DIR}/server-deployment-dev.yaml
fi
