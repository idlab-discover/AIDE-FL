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

# Build and push datastore-logic image
docker build --tag ${IMAGE_REPO}/datastore-logic:$TAG ${DIR}/datastore-logic
docker push ${IMAGE_REPO}/datastore-logic:$TAG

# Recreate Minio + datastore-logic daemonset
kubectl delete daemonset datastore-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/datastore-daemonset.yaml
else
    kubectl apply -f ${DIR}/datastore-daemonset-dev.yaml
fi
