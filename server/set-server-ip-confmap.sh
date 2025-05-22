#!/bin/bash

# Get the IP address of the node with the label role=server
SERVER_IP=$(kubectl get nodes -l fl-role=server -o json | jq -r '.items[0].status.addresses[] | select(.type=="InternalIP") | .address')

# Create a ConfigMap with the node IP
kubectl create configmap server-ip --from-literal=SERVER_IP=$SERVER_IP -n aide-fl