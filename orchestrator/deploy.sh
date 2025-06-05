#!/bin/bash

DIR=$(dirname "$0")

# Build and push orchestrator image
docker build --tag ghcr.io/martienskienum1/orchestrator ${DIR}
docker push ghcr.io/martienskienum1/orchestrator:latest

# Recreate orchestrator daemonset
kubectl delete daemonset orchestrator-daemonset --namespace aide-fl
kubectl apply -f ${DIR}/orchestrator-daemonset.yaml