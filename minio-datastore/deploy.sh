#!/bin/bash

docker build --tag ghcr.io/martienskienum1/minio-proxy $(dirname "$0")/proxy
docker push ghcr.io/martienskienum1/minio-proxy:latest

kubectl delete daemonset minio-daemonset --namespace aide-fl
kubectl apply -f ../minio-daemonset.yaml