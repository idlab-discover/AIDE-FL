#!/bin/bash

DIR=$(dirname "$0")

# Build and push flower-clientapp image
docker build -f ${DIR}/../../flower/clientapp.Dockerfile --tag ghcr.io/martienskienum1/flower-clientapp ${DIR}/../../flower/.
docker push ghcr.io/martienskienum1/flower-clientapp:latest

# Build and push trainer-logic image
docker build --tag ghcr.io/martienskienum1/trainer-logic ${DIR}/trainer-logic
docker push ghcr.io/martienskienum1/trainer-logic:latest

# Recreate flower-superlink + flower-clientapp + trainer-logic daemonset
kubectl delete daemonset trainer-daemonset --namespace aide-fl
kubectl apply -f ${DIR}/trainer-daemonset.yaml