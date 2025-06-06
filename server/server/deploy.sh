#!/bin/bash

DIR=$(dirname "$0")

# Build and push flower-serverapp image
docker build -f ../flower/serverapp.Dockerfile --tag ghcr.io/martienskienum1/flower-serverapp ${DIR}/../../flower/.
docker push ghcr.io/martienskienum1/flower-serverapp:latest

# Build and push server-logic image
docker build --tag ghcr.io/martienskienum1/server-logic ${DIR}/server-logic
docker push ghcr.io/martienskienum1/server-logic:latest

# Recreate flower-superlink + flower-serverapp + server-logic deployment
kubectl delete deployment server-deployment --namespace aide-fl
kubectl apply -f ${DIR}/server-deployment.yaml