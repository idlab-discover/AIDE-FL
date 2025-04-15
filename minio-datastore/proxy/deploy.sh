#!/bin/bash

docker build --tag ghcr.io/martienskienum1/minio-proxy $(dirname "$0")
docker push ghcr.io/martienskienum1/minio-proxy:latest
microk8s kubectl delete daemonset minio-daemonset --namespace aide-fl
microk8s kubectl apply -f ../minio-daemonset.yaml