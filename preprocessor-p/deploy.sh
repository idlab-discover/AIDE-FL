#!/bin/bash

docker build --tag ghcr.io/martienskienum1/preprocessor $(dirname "$0")
docker push ghcr.io/martienskienum1/preprocessor:latest
kubectl delete daemonset preprocessor-daemonset --namespace aide-fl
kubectl apply -f preprocessor-daemonset.yaml