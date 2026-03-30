#!/bin/bash

set -e

DIR=$(dirname "$0")
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/str-gen}"
IMAGE_TAG="${IMAGE_TAG:-}"

if [[ -z "$IMAGE_TAG" && "$1" == "prod" ]]; then
    TAG="latest"
elif [[ -z "$IMAGE_TAG" ]]; then
    TAG="dev"
else
    TAG="$IMAGE_TAG"
fi

echo "Deploying environment with tag: $TAG"

# Build and push orchestrator image
docker build --tag ${IMAGE_REPO}/orchestrator:$TAG ${DIR}
docker push ${IMAGE_REPO}/orchestrator:$TAG

# Recreate orchestrator daemonset
kubectl delete daemonset orchestrator-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/orchestrator-daemonset.yaml
else
    kubectl apply -f ${DIR}/orchestrator-daemonset-dev.yaml
fi

${DIR}/../../scripts/set-custom-images.sh --component orchestrator --image-repo "${IMAGE_REPO}" --image-tag "${TAG}"
