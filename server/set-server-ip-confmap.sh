#!/bin/bash

# Get the IP address of the node with the label role=web
NODE_IP=$(kubectl get nodes -l fl-role=server -o json | jq -r '.items[0].status.addresses[] | select(.type=="InternalIP") | .address')

# Create a ConfigMap with the node IP
kubectl create configmap node-ip --from-literal=NODE_IP=$NODE_IP -n aide-fl