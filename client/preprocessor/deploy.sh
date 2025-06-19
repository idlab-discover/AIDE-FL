#!/bin/bash

set -e

DIR=$(dirname "$0")

if [ "$1" == "prod" ]; then
    TAG="latest"
else
    TAG="dev"
fi

echo "Deploying environment with tag: $TAG"

# Build and push preprocessor-logic image
docker build --tag ghcr.io/martienskienum1/preprocessor-logic:$TAG ${DIR}/preprocessor-logic
docker push ghcr.io/martienskienum1/preprocessor-logic:$TAG

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