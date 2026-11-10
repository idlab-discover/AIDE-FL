#!/usr/bin/env bash
set -euo pipefail
DIR=$(cd "$(dirname "$0")" && pwd)
KUBECONFIG_PATH="${KUBECONFIG_PATH:-${KUBECONFIG:-${HOME}/.kube/aide}}"
WANDB_KEY_VALUE="${WANDB_API_KEY:-}"
WANDB_MODE_VALUE="${WANDB_MODE:-}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/idlab-discover/aide-fl}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
API_PORT="${API_PORT:-80}"
ROLLOUT_TIMEOUT="${ROLLOUT_TIMEOUT:-300s}"
ASSUME_YES=false
usage() {
  cat <<USAGE
Usage: $0 [--kubeconfig PATH] [--yes] [--image-repo REPO] [--image-tag TAG]
          [--api-port PORT] [--wandb-mode offline|disabled|online] [--wandb-api-key KEY]
Defaults: API port 80, W&B offline (online when a key is supplied), image tag latest.
Use an immutable published image tag for reproducible runs. ROLLOUT_TIMEOUT defaults to 300s.
USAGE
}
fail() { echo "Error: $*" >&2; exit 1; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) ASSUME_YES=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --kubeconfig|--wandb-api-key|--wandb-mode|--image-repo|--image-tag|--api-port)
      [[ $# -ge 2 && -n "$2" ]] || fail "Missing value for $1"
      case "$1" in
        --kubeconfig) KUBECONFIG_PATH="$2" ;;
        --wandb-api-key) WANDB_KEY_VALUE="$2" ;;
        --wandb-mode) WANDB_MODE_VALUE="$2" ;;
        --image-repo) IMAGE_REPO="$2" ;;
        --image-tag) IMAGE_TAG="$2" ;;
        --api-port) API_PORT="$2" ;;
      esac
      shift 2 ;;
    *) fail "Unknown argument: $1" ;;
  esac
done
[[ "$API_PORT" =~ ^[1-9][0-9]{0,4}$ ]] && (( API_PORT <= 65535 )) || fail "API port must be 1..65535"
[[ "$ROLLOUT_TIMEOUT" =~ ^[1-9][0-9]*[smh]$ ]] || fail "ROLLOUT_TIMEOUT must be a positive duration, e.g. 300s"
if [[ -z "$WANDB_MODE_VALUE" ]]; then
  WANDB_MODE_VALUE=offline
  [[ -z "$WANDB_KEY_VALUE" ]] || WANDB_MODE_VALUE=online
fi
case "$WANDB_MODE_VALUE" in offline|disabled|online) ;; *) fail "Invalid W&B mode" ;; esac
[[ -f "$KUBECONFIG_PATH" ]] || fail "Kubeconfig does not exist: $KUBECONFIG_PATH"
export KUBECONFIG="$KUBECONFIG_PATH"
for cmd in kubectl jq openssl; do command -v "$cmd" >/dev/null || fail "Missing command: $cmd"; done
context=$(kubectl config current-context)
server=$(kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}')
echo "Deploying to $context ($server), images ${IMAGE_REPO}:$IMAGE_TAG, API port $API_PORT, W&B $WANDB_MODE_VALUE"
if ! "$ASSUME_YES"; then
  read -r -p "Deploy to this cluster? (y/N) " answer
  [[ "$answer" == y || "$answer" == Y ]] || exit 1
fi
nodes=$(kubectl get nodes -o json)
[[ $(jq '[.items[] | select(.metadata.labels["fl-role"] == "server")] | length' <<<"$nodes") == 1 ]] || fail "Expected exactly one fl-role=server node"
[[ $(jq '[.items[] | select(.metadata.labels["fl-role"] == "client")] | length' <<<"$nodes") -ge 1 ]] || fail "Expected at least one fl-role=client node"
SERVER_IP=$(jq -r '.items[] | select(.metadata.labels["fl-role"] == "server") | .status.addresses[] | select(.type == "InternalIP") | .address' <<<"$nodes")
[[ -n "$SERVER_IP" ]] || fail "Server node has no InternalIP"
if [[ "$WANDB_MODE_VALUE" == online && -z "$WANDB_KEY_VALUE" ]]; then
  kubectl get secret wandb-api-key -n aide-fl >/dev/null 2>&1 || fail "Online W&B requires WANDB_API_KEY or an existing wandb-api-key secret"
fi
kubectl create namespace aide-fl --dry-run=client -o yaml | kubectl apply -f -
if [[ -n "$WANDB_KEY_VALUE" ]]; then
  kubectl create secret generic wandb-api-key -n aide-fl --from-literal="WANDB_API_KEY=$WANDB_KEY_VALUE" --dry-run=client -o yaml | kubectl apply -f -
fi
ca_exists=false; cert_exists=false
kubectl get configmap ca-cert -n aide-fl >/dev/null 2>&1 && ca_exists=true
kubectl get secret server-cert -n aide-fl >/dev/null 2>&1 && cert_exists=true
[[ "$ca_exists" == "$cert_exists" ]] || fail "Incomplete Flower certificates: restore the missing ca-cert or server-cert before deploying"
if ! "$ca_exists"; then
  "$DIR/gen_cert.sh"
  kubectl create secret tls server-cert -n aide-fl --cert="$DIR/certs/server.pem" --key="$DIR/certs/server.key" --dry-run=client -o yaml | kubectl apply -f -
  kubectl create configmap ca-cert -n aide-fl --from-file="$DIR/certs/ca.crt" --dry-run=client -o yaml | kubectl apply -f -
fi
kubectl create configmap server-ip -n aide-fl --from-literal="SERVER_IP=$SERVER_IP" --from-literal="SERVER_PORT=:$API_PORT" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "$DIR/client/kafka/kafka-configmap.yaml" -f "$DIR/client/preprocessor/redis-configmap.yaml"
# Render images and runtime settings before creating or updating any workload.
apply_workload() {
  kubectl create --dry-run=client -o json -f "$DIR/$1" |
    jq --arg repo "$IMAGE_REPO" --arg tag "$IMAGE_TAG" --arg port ":$API_PORT" --arg mode "$WANDB_MODE_VALUE" '
      .spec.template.spec.containers |= map(
        if (.name | IN("flower-serverapp", "server-logic", "orchestrator", "datastore-logic", "preprocessor-logic", "flower-clientapp", "trainer-logic"))
        then .image = ($repo + "/" + .name + ":" + $tag) else . end |
        if .name == "flower-serverapp" then .env = ((.env // [] | map(select(.name != "WANDB_MODE"))) + [{name:"WANDB_MODE",value:$mode}])
        elif .name == "server-logic" then .env = ((.env // [] | map(select(.name != "LISTEN_PORT"))) + [{name:"LISTEN_PORT",value:$port}])
        elif .name == "orchestrator" then .env = ((.env // [] | map(select(.name != "SERVER_PORT"))) + [{name:"SERVER_PORT",value:$port}])
        else . end)' | kubectl apply -f -
}
for manifest in client/kafka/kafka-daemonset.yaml server/datastore/minio-server-deployment.yaml server/server/server-deployment.yaml client/datastore/datastore-daemonset.yaml client/orchestrator/orchestrator-daemonset.yaml client/preprocessor/preprocessor-daemonset.yaml client/trainer/trainer-daemonset.yaml; do
  apply_workload "$manifest"
done
for resource in daemonset/kafka-daemonset deployment/minio-server-deployment deployment/server-deployment daemonset/datastore-daemonset daemonset/orchestrator-daemonset daemonset/preprocessor-daemonset daemonset/trainer-daemonset; do
  kubectl rollout status -n aide-fl "$resource" --timeout="$ROLLOUT_TIMEOUT"
done
echo "Deployment ready. API: http://${SERVER_IP}:${API_PORT}"
