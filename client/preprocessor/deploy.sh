#!/bin/bash

DIR=$(dirname "$0")

# Build and push preprocessor-logic image
docker build --tag ghcr.io/martienskienum1/preprocessor-logic ${DIR}/preprocessor-logic
docker push ghcr.io/martienskienum1/preprocessor-logic:latest

# Recreate redis configmap for Redis config
kubectl delete configmap redis-config -n aide-fl
kubectl apply -f ${DIR}/redis-configmap.yaml

# Recreate preprocessor daemonset
kubectl delete daemonset preprocessor-daemonset --namespace aide-fl
kubectl apply -f ${DIR}/preprocessor-daemonset.yaml