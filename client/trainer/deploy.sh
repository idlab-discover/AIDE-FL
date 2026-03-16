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

# Build and push flower-clientapp image
docker build -f ${DIR}/../../flower/clientapp.Dockerfile --tag ${IMAGE_REPO}/flower-clientapp:$TAG ${DIR}/../../flower/.
docker push ${IMAGE_REPO}/flower-clientapp:$TAG

# Build and push trainer-logic image
docker build --tag ${IMAGE_REPO}/trainer-logic:$TAG ${DIR}/trainer-logic
docker push ${IMAGE_REPO}/trainer-logic:$TAG

# Recreate flower-superlink + flower-clientapp + trainer-logic daemonset
kubectl delete daemonset trainer-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/trainer-daemonset.yaml
else
    kubectl apply -f ${DIR}/trainer-daemonset-dev.yaml
fi
