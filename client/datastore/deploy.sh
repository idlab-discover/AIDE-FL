#!/bin/bash

DIR=$(dirname "$0")

# Build and push datastore-logic image
docker build --tag ghcr.io/martienskienum1/datastore-logic ${DIR}/datastore-logic
docker push ghcr.io/martienskienum1/datastore-logic:latest

# Recreate Minio + datastore-logic daemonset
kubectl delete daemonset datastore-daemonset --namespace aide-fl
kubectl apply -f ${DIR}/datastore-daemonset.yaml