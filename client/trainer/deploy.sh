#!/bin/bash

set -e

DIR=$(dirname "$0")

if [ "$1" == "prod" ]; then
    TAG="latest"
else
    TAG="dev"
fi

echo "Deploying environment with tag: $TAG"

# Build and push flower-clientapp image
docker build -f ${DIR}/../../flower/clientapp.Dockerfile --tag ghcr.io/martienskienum1/flower-clientapp:$TAG ${DIR}/../../flower/.
docker push ghcr.io/martienskienum1/flower-clientapp:$TAG

# Build and push trainer-logic image
docker build --tag ghcr.io/martienskienum1/trainer-logic:$TAG ${DIR}/trainer-logic
docker push ghcr.io/martienskienum1/trainer-logic:$TAG

# Recreate flower-superlink + flower-clientapp + trainer-logic daemonset
kubectl delete daemonset trainer-daemonset --namespace aide-fl || true

if [ "$1" == "prod" ]; then
    kubectl apply -f ${DIR}/trainer-daemonset.yaml
else
    kubectl apply -f ${DIR}/trainer-daemonset-dev.yaml
fi