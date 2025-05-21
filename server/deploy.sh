#!/bin/bash

docker build --tag ghcr.io/martienskienum1/flower-serverapp flower-serverapp
docker push ghcr.io/martienskienum1/flower-serverapp:latest

docker build --tag ghcr.io/martienskienum1/server-logic server-logic
docker push ghcr.io/martienskienum1/server-logic:latest

kubectl delete deployment server-deployment --namespace aide-fl
kubectl apply -f server-deployment.yaml