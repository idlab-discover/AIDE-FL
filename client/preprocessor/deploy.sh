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

# Build and push preprocessor-logic image
docker build --tag ${IMAGE_REPO}/preprocessor-logic:$TAG ${DIR}/preprocessor-logic
docker push ${IMAGE_REPO}/preprocessor-logic:$TAG

# Recreate redis configmap for Redis config
kubectl delete configmap redis-config -n aide-fl
kubectl apply -f ${DIR}/redis-configmap.yaml

# Recreate preprocessor daemonset
kubectl delete daemonset preprocessor-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
   kubectl apply -f ${DIR}/preprocessor-daemonset.yaml
else
    kubectl apply -f ${DIR}/preprocessor-daemonset-dev.yaml
fi

${DIR}/../../scripts/set-custom-images.sh --component preprocessor --image-repo "${IMAGE_REPO}" --image-tag "${TAG}"
