#!/bin/bash

docker build --tag ghcr.io/martienskienum1/preprocessor $(dirname "$0")
docker push ghcr.io/martienskienum1/preprocessor:latest
microk8s kubectl delete daemonset preprocessor-daemonset --namespace aide-fl
microk8s kubectl apply -f preprocessor-daemonset.yaml