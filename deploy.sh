#!/bin/bash

set -e # Exit if a command exits with non-zero status
set -o pipefail # Exit if part of pipeline fails

DIR=$(dirname "$0")

# Check for dependencies
echo "Checking dependencies..."
required_commands=(kubectl jq)
for cmd in "${required_commands[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Error: Required command '$cmd' is not installed or not in PATH."
    exit 1
  fi
done
echo "All required commands are available."

# Check whether a kubeconfig context is set
current_context=$(kubectl config current-context 2>/dev/null || true)
if [[ -z "$current_context" ]]; then
  echo "Error: No current Kubernetes context is set in kubeconfig."
  exit 1
else
  echo "Current kubeconfig context: $current_context"
fi

# Ask whether kubeconfig points to right k8s instance
server=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')
echo "Current Kubernetes server: ${server}"
read -p "Is above server the right k8s instance? (y/N)" ans
if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
    echo "Exiting."
    exit 1
fi

# Check whether there is exactly one node with label fl-role=server and at least one node with label fl-role=client
echo "Checking node labels..."

server_count=$(kubectl get nodes -l fl-role=server --no-headers 2>/dev/null | wc -l)
client_count=$(kubectl get nodes -l fl-role=client --no-headers 2>/dev/null | wc -l)

if [[ "$server_count" -ne 1 ]]; then
  echo "Error: Expected exactly 1 node with label 'fl-role=server', but found $server_count."
  echo "Use    'kubectl label nodes <node-name> fl-role=server'    to add labels"
  exit 1
fi

if [[ "$client_count" -lt 1 ]]; then
  echo "Error: Expected at least 1 node with label 'fl-role=client', but found $client_count."
  echo "Use    'kubectl label nodes <node-name> fl-role=client'    to add labels"
  exit 1
fi

echo "Valid setup: $server_count server and $client_count client node(s)."


# Create .env files
echo "Creating .env files..."
find ${DIR} -type f -name ".env.example" | while read example_file; do
  env_file="${example_file%.example}"  # Replace .env.example with .env
  if [ -f "$env_file" ]; then
    echo "Skipping $env_file (already exists)"
  else
    cp "$example_file" "$env_file"
    echo "Created $env_file from $example_file"
  fi
done

# Set wandb API key
read -p "Enter your Weights & Biases API key: " wandb_key

flower_env="${DIR}/flower/.env"

sed -i.bak "s/^WANDB_API_KEY.*/WANDB_API_KEY = $wandb_key/" "$flower_env"
echo "Updated WANDB_API_KEY in $flower_env"

# Create aide-fl namespace
echo "Creating aide-fl namespace..."
kubectl create namespace aide-fl

# Configmap sets Kafka topics to be created
echo "Creating Kafka topics configmap..."
kubectl apply -f ${DIR}/client/kafka/kafka-configmap.yaml

# Create Kafka daemonset
echo "Creating Kafka daemonset..."
kubectl apply -f ${DIR}/client/kafka/kafka-daemonset.yaml

# Create flower-superlink + flower-serverapp + server-logic deployment
echo "Creating server deployment..."
kubectl apply -f ${DIR}/server/server/server-deployment.yaml

# Create configmap with server's IP
echo "Creating server's IP configmap..."
SERVER_IP=$(kubectl get nodes -l fl-role=server -o json | jq -r '.items[0].status.addresses[] | select(.type=="InternalIP") | .address')
kubectl create configmap server-ip --from-literal=SERVER_IP=$SERVER_IP -n aide-fl

# Create Minio + minio-proxy daemonset
echo "Creating minio daemonset..."
kubectl apply -f ${DIR}/client/datastore/minio-daemonset.yaml

# Create orchestrator daemonset
echo "Creating orchestrator daemonset..."
kubectl apply -f ${DIR}/client/orchestrator/orchestrator-daemonset.yaml

# Create preprocessor daemonset
echo "Creating preprocessor daemonset..."
kubectl apply -f ${DIR}/client/preprocessor/preprocessor-daemonset.yaml

# Create flower-superlink + flower-clientapp + trainer-logic daemonset
echo "Creating trainer daemonset..."
kubectl apply -f ${DIR}/client/trainer/trainer-daemonset.yaml

echo "Done"