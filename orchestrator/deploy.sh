#!/bin/bash

docker build --tag ghcr.io/martienskienum1/orchestrator $(dirname "$0")
docker push ghcr.io/martienskienum1/orchestrator:latest
microk8s kubectl delete daemonset orchestrator-daemonset --namespace aide-fl
microk8s kubectl apply -f orchestrator-daemonset.yaml