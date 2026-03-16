#!/usr/bin/env bash

set -euo pipefail

DEFAULT_KUBECONFIG="${HOME}/.kube/aide"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-${KUBECONFIG:-$DEFAULT_KUBECONFIG}}"
NAMESPACE="${NAMESPACE:-aide-fl}"
EXPECTED_CLIENTS="${EXPECTED_CLIENTS:-2}"
SERVER_URL="${SERVER_URL:-}"
MINIO_URL="${MINIO_URL:-}"

usage() {
  cat <<EOF
Usage: $0 [--kubeconfig PATH] [--namespace NAME] [--expected-clients N] [--server-url URL] [--minio-url URL]

Checks the current AIDE-FL deployment for a demo:
- node labels
- namespace objects
- pod readiness
- server HTTP endpoints
- MinIO reachability
- connected client count
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --kubeconfig)
      KUBECONFIG_PATH="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --expected-clients)
      EXPECTED_CLIENTS="$2"
      shift 2
      ;;
    --server-url)
      SERVER_URL="$2"
      shift 2
      ;;
    --minio-url)
      MINIO_URL="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

required_commands=(kubectl jq curl)
for cmd in "${required_commands[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

if [[ ! -f "$KUBECONFIG_PATH" ]]; then
  echo "Kubeconfig not found: $KUBECONFIG_PATH" >&2
  exit 1
fi

export KUBECONFIG="$KUBECONFIG_PATH"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

pass() {
  echo "PASS: $*"
}

server_node_count="$(kubectl get nodes -l fl-role=server --no-headers 2>/dev/null | wc -l | tr -d ' ')"
client_node_count="$(kubectl get nodes -l fl-role=client --no-headers 2>/dev/null | wc -l | tr -d ' ')"

[[ "$server_node_count" == "1" ]] || fail "expected exactly 1 server-labeled node, found $server_node_count"
[[ "$client_node_count" -ge 1 ]] || fail "expected at least 1 client-labeled node, found $client_node_count"
pass "node labels look valid ($server_node_count server, $client_node_count client)"

server_node="$(kubectl get nodes -l fl-role=server -o jsonpath='{.items[0].metadata.name}')"
server_ip="$(kubectl get nodes -l fl-role=server -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')"
control_plane_labeled="$(kubectl get nodes -l node-role.kubernetes.io/control-plane,fl-role --no-headers 2>/dev/null | wc -l | tr -d ' ')"
[[ "$control_plane_labeled" == "0" ]] || fail "control-plane node has an fl-role label"
pass "control plane is not labeled for FL scheduling"

[[ -n "$SERVER_URL" ]] || SERVER_URL="http://${server_ip}"
[[ -n "$MINIO_URL" ]] || MINIO_URL="http://${server_ip}:9001"

kubectl get namespace "$NAMESPACE" >/dev/null 2>&1 || fail "namespace '$NAMESPACE' does not exist"
pass "namespace '$NAMESPACE' exists"

for obj in \
  "secret/wandb-api-key" \
  "secret/server-cert" \
  "configmap/ca-cert" \
  "configmap/server-ip" \
  "configmap/redis-config" \
  "configmap/kafka-config" \
  "deployment/server-deployment" \
  "deployment/minio-server-deployment" \
  "daemonset/kafka-daemonset" \
  "daemonset/datastore-daemonset" \
  "daemonset/orchestrator-daemonset" \
  "daemonset/preprocessor-daemonset" \
  "daemonset/trainer-daemonset"
do
  kubectl get -n "$NAMESPACE" "$obj" >/dev/null 2>&1 || fail "missing required object: $obj"
done
pass "required Kubernetes objects exist"

pod_readiness_failures="$(
  kubectl get pods -n "$NAMESPACE" -o json |
    jq -r '
      .items[]
      | select(
          (.status.phase != "Running") or
          ((.status.containerStatuses // []) | any(.ready != true))
        )
      | .metadata.name
    '
)"
if [[ -n "$pod_readiness_failures" ]]; then
  echo "$pod_readiness_failures" | sed 's/^/  - /' >&2
  fail "one or more pods are not fully ready"
fi
pass "all pods in '$NAMESPACE' are ready"

clients_json="$(curl -fsS "${SERVER_URL}/clients")" || fail "unable to reach ${SERVER_URL}/clients"
status_json="$(curl -fsS "${SERVER_URL}/status")" || fail "unable to reach ${SERVER_URL}/status"
curl -fsS "${MINIO_URL}" >/dev/null || fail "unable to reach MinIO console at ${MINIO_URL}"
pass "server API and MinIO are reachable"

connected_clients="$(jq '.data | length' <<<"$clients_json")"
[[ "$connected_clients" =~ ^[0-9]+$ ]] || fail "could not parse /clients response"
[[ "$connected_clients" -ge "$EXPECTED_CLIENTS" ]] || fail "expected at least $EXPECTED_CLIENTS connected clients, found $connected_clients"
pass "server reports $connected_clients connected clients"

server_ip_from_configmap="$(kubectl get configmap server-ip -n "$NAMESPACE" -o jsonpath='{.data.SERVER_IP}')"
[[ "$server_ip_from_configmap" == "$server_ip" ]] || fail "server-ip configmap points to $server_ip_from_configmap, expected $server_ip"
pass "server-ip configmap matches server node IP ($server_ip)"

status_success="$(jq -r '.data.success // empty' <<<"$status_json")"
[[ "$status_success" == "true" || "$status_success" == "false" ]] || fail "unexpected /status response shape"
pass "/status response shape looks valid"

echo
echo "Summary"
echo "  Kubeconfig:     $KUBECONFIG_PATH"
echo "  Namespace:      $NAMESPACE"
echo "  Server node:    $server_node"
echo "  Server URL:     $SERVER_URL"
echo "  MinIO URL:      $MINIO_URL"
echo "  Connected:      $connected_clients client(s)"
