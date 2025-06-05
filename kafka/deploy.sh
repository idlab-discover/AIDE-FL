#!/bin/bash

DIR=$(dirname "$0")

# Configmap sets topics to be created
kubectl apply -f ${DIR}/kafka-configmap.yaml

# Recreate Kafka daemonset
kubectl delete daemonset kafka-daemonset -n aide-fl
kubectl apply -f ${DIR}/kafka-daemonset.yaml