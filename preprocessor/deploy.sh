#!/bin/bash

DIR=$(dirname "$0")

# Build and push preprocessor image
docker build --tag ghcr.io/martienskienum1/preprocessor ${DIR}
docker push ghcr.io/martienskienum1/preprocessor:latest

# Recreate preprocessor daemonset
kubectl delete daemonset preprocessor-daemonset --namespace aide-fl
kubectl apply -f ${DIR}/preprocessor-daemonset.yaml