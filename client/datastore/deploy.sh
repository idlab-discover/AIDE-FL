#!/bin/bash

DIR=$(dirname "$0")

# Build and push minio-proxy image
docker build --tag ghcr.io/martienskienum1/minio-proxy ${DIR}/proxy
docker push ghcr.io/martienskienum1/minio-proxy:latest

# Recreate Minio + minio-proxy daemonset
kubectl delete daemonset minio-daemonset --namespace aide-fl
kubectl apply -f ${DIR}/minio-daemonset.yaml