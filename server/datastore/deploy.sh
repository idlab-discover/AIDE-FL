#!/bin/bash

DIR=$(dirname "$0")

# Recreate server datastore deployment
kubectl delete deployment minio-server-deployment --namespace aide-fl
kubectl apply -f ${DIR}/minio-server-deployment.yaml