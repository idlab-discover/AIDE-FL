#!/bin/bash

docker build --tag ghcr.io/martienskienum1/trainer $(dirname "$0")
docker push ghcr.io/martienskienum1/trainer:latest
kubectl delete daemonset trainer-daemonset --namespace aide-fl
kubectl apply -f trainer-daemonset.yaml