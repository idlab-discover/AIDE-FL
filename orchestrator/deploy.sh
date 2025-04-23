#!/bin/bash

docker build --tag ghcr.io/martienskienum1/orchestrator $(dirname "$0")
docker push ghcr.io/martienskienum1/orchestrator:latest
kubectl delete daemonset orchestrator-daemonset --namespace aide-fl
kubectl apply -f orchestrator-daemonset.yaml