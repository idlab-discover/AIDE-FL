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

${DIR}/../../scripts/set-custom-images.sh --component trainer --image-repo "${IMAGE_REPO}" --image-tag "${TAG}"
