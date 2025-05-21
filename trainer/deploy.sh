#!/bin/bash

docker build --tag ghcr.io/martienskienum1/flower-clientapp flower-clientapp
docker push ghcr.io/martienskienum1/flower-clientapp:latest

docker build --tag ghcr.io/martienskienum1/trainer-logic trainer-logic
docker push ghcr.io/martienskienum1/trainer-logic:latest

kubectl delete daemonset trainer-daemonset --namespace aide-fl
kubectl apply -f trainer-daemonset.yaml