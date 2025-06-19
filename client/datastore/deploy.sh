#!/bin/bash

set -e

DIR=$(dirname "$0")

if [ "$1" == "prod" ]; then
    TAG="latest"
else
    TAG="dev"
fi

echo "Deploying environment with tag: $TAG"

# Build and push datastore-logic image
docker build --tag ghcr.io/martienskienum1/datastore-logic:$TAG ${DIR}/datastore-logic
docker push ghcr.io/martienskienum1/datastore-logic:$TAG

# Recreate Minio + datastore-logic daemonset
kubectl delete daemonset datastore-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/datastore-daemonset.yaml
else
    kubectl apply -f ${DIR}/datastore-daemonset-dev.yaml
fi