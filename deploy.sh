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

sed -i "s/^WANDB_API_KEY.*/WANDB_API_KEY = $wandb_key/" "$flower_env"
echo "Updated WANDB_API_KEY in $flower_env"

# Check if namespace exists
if kubectl get namespace aide-fl >/dev/null 2>&1; then
  echo "Namespace 'aide-fl' exists."
else
  echo "Namespace 'aide-fl' does not exist. Creating..."
  kubectl create namespace aide-fl
fi

# Configmap sets Kafka topics to be created
echo "Creating Kafka topics configmap..."
kubectl apply -f ${DIR}/client/kafka/kafka-configmap.yaml

# Check if Kafka daemonset exists
if kubectl get daemonset kafka-daemonset -n aide-fl >/dev/null 2>&1; then
  echo "'kafka-daemonset' exists. Deleting..."
  kubectl delete daemonset kafka-daemonset -n aide-fl
fi

# Create Kafka daemonset
echo "Creating Kafka daemonset..."
kubectl apply -f ${DIR}/client/kafka/kafka-daemonset.yaml

# Wait for Kafka daemonset to be ready
while true; do
  daemonset_status=$(kubectl get daemonset kafka-daemonset -n aide-fl -o json 2>/dev/null)
  desired=$(echo "$daemonset_status" | jq -r '.status.desiredNumberScheduled // 0')
  ready=$(echo "$daemonset_status" | jq -r '.status.numberReady // 0')

  if [[ "$desired" -eq "$ready" ]]; then
    echo "Kafka daemonset is ready"
    break
  fi

  echo "Waiting for Kafka daemonset to be ready..."
  sleep 5
done

# Check if Minio server datastore deployment exists
if kubectl get deployment minio-server-deployment -n aide-fl >/dev/null 2>&1; then
  echo "'minio-server-deployment' exists. Deleting..."
  kubectl delete deployment minio-server-deployment -n aide-fl
fi

# Create Minio server datastore deployment
echo "Creating server datastore deployment..."
kubectl apply -f ${DIR}/server/datastore/minio-server-deployment.yaml

# Check if flower-superlink + flower-serverapp + server-logic deployment exists
if kubectl get deployment server-deployment -n aide-fl >/dev/null 2>&1; then
  echo "'server-deployment' exists. Deleting..."
  kubectl delete deployment server-deployment -n aide-fl
fi

# Create flower-superlink + flower-serverapp + server-logic deployment
echo "Creating server deployment..."
kubectl apply -f ${DIR}/server/server/server-deployment.yaml

# Check if configmap with server's IP exists
if kubectl get configmap server-ip -n aide-fl >/dev/null 2>&1; then
  echo "'server-deployment' exists. Deleting..."
  kubectl delete configmap server-ip -n aide-fl
fi

# Create configmap with server's IP
echo "Creating server's IP configmap..."
SERVER_IP=$(kubectl get nodes -l fl-role=server -o json | jq -r '.items[0].status.addresses[] | select(.type=="InternalIP") | .address')
kubectl create configmap server-ip --from-literal=SERVER_IP=$SERVER_IP -n aide-fl

# Check if Minio + minio-proxy daemonset exists
if kubectl get daemonset minio-daemonset -n aide-fl >/dev/null 2>&1; then
  echo "'minio-daemonset' exists. Deleting..."
  kubectl delete daemonset minio-daemonset -n aide-fl
fi

# Create Minio + minio-proxy daemonset
echo "Creating client datastore daemonset..."
kubectl apply -f ${DIR}/client/datastore/minio-daemonset.yaml

# Check if orchestrator daemonset exists
if kubectl get daemonset orchestrator-daemonset -n aide-fl >/dev/null 2>&1; then
  echo "'orchestrator-daemonset' exists. Deleting..."
  kubectl delete daemonset orchestrator-daemonset -n aide-fl
fi

# Create orchestrator daemonset
echo "Creating orchestrator daemonset..."
kubectl apply -f ${DIR}/client/orchestrator/orchestrator-daemonset.yaml

# Check if orchestrator daemonset exists
if kubectl get daemonset preprocessor-daemonset -n aide-fl >/dev/null 2>&1; then
  echo "'preprocessor-daemonset' exists. Deleting..."
  kubectl delete daemonset preprocessor-daemonset -n aide-fl
fi

# Create preprocessor daemonset
echo "Creating preprocessor daemonset..."
kubectl apply -f ${DIR}/client/preprocessor/preprocessor-daemonset.yaml

# Check if flower-superlink + flower-clientapp + trainer-logic daemonset exists
if kubectl get daemonset trainer-daemonset -n aide-fl >/dev/null 2>&1; then
  echo "'trainer-daemonset' exists. Deleting..."
  kubectl delete daemonset trainer-daemonset -n aide-fl
fi

# Create flower-superlink + flower-clientapp + trainer-logic daemonset
echo "Creating trainer daemonset..."
kubectl apply -f ${DIR}/client/trainer/trainer-daemonset.yaml

echo "Done"