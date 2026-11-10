#!/usr/bin/env bash

set -euo pipefail

DEFAULT_KUBECONFIG="${HOME}/.kube/aide"
KUBECONFIG_PATH="${KUBECONFIG_PATH:-${KUBECONFIG:-$DEFAULT_KUBECONFIG}}"
SCENARIO_PATH="${SCENARIO_PATH:-examples/nids.json}"
NAMESPACE="${NAMESPACE:-aide-fl}"
SERVER_URL="${SERVER_URL:-}"
POLL_INTERVAL="${POLL_INTERVAL:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1800}"
EXPECTED_CLIENTS="${EXPECTED_CLIENTS:-2}"
RUN_NAME_SUFFIX="${RUN_NAME_SUFFIX:-demo-$(date +%Y%m%d-%H%M%S)}"
SKIP_HEALTHCHECK=0
KEEP_SCENARIO=0
DRY_RUN=0

usage() {
  cat <<EOF
Usage: $0 [--scenario PATH] [--server-url URL] [--kubeconfig PATH] [--poll-interval SEC] [--timeout SEC] [--expected-clients N] [--skip-healthcheck] [--keep-scenario] [--dry-run]

Starts a demo run against the current AIDE-FL deployment.

Defaults:
- scenario: examples/nids.json
- kubeconfig: ~/.kube/aide
- unique name suffix: ${RUN_NAME_SUFFIX}

If the scenario uses placeholder dataset keys such as client1/client2, they are rewritten
to the currently connected client IDs in lexical order.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario)
      SCENARIO_PATH="$2"
      shift 2
      ;;
    --server-url)
      SERVER_URL="$2"
      shift 2
      ;;
    --kubeconfig)
      KUBECONFIG_PATH="$2"
      shift 2
      ;;
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="$2"
      shift 2
      ;;
    --timeout)
      TIMEOUT_SECONDS="$2"
      shift 2
      ;;
    --expected-clients)
      EXPECTED_CLIENTS="$2"
      shift 2
      ;;
    --run-name-suffix)
      RUN_NAME_SUFFIX="$2"
      shift 2
      ;;
    --skip-healthcheck)
      SKIP_HEALTHCHECK=1
      shift
      ;;
    --keep-scenario)
      KEEP_SCENARIO=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

required_commands=(kubectl jq curl mktemp python)
for cmd in "${required_commands[@]}"; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 1
  fi
done

[[ -f "$KUBECONFIG_PATH" ]] || { echo "Kubeconfig not found: $KUBECONFIG_PATH" >&2; exit 1; }
[[ -f "$SCENARIO_PATH" ]] || { echo "Scenario not found: $SCENARIO_PATH" >&2; exit 1; }

export KUBECONFIG="$KUBECONFIG_PATH"

server_ip="$(kubectl get nodes -l fl-role=server -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')"
[[ -n "$SERVER_URL" ]] || SERVER_URL="http://${server_ip}"

clients_json="$(curl -fsS "${SERVER_URL}/clients")"
mapfile -t connected_client_ids < <(jq -r '.data[].id' <<<"$clients_json" | sort)
connected_count="${#connected_client_ids[@]}"

if [[ "$connected_count" -lt "$EXPECTED_CLIENTS" ]]; then
  echo "Expected at least $EXPECTED_CLIENTS connected clients, found $connected_count" >&2
  exit 1
fi

if [[ "$SKIP_HEALTHCHECK" -eq 0 ]]; then
  ALLOW_CONTROL_PLANE="${ALLOW_CONTROL_PLANE:-false}" ./health-check.sh \
    --kubeconfig "$KUBECONFIG_PATH" \
    --namespace "$NAMESPACE" \
    --expected-clients "$EXPECTED_CLIENTS" \
    --server-url "$SERVER_URL"
  echo
fi

scenario_tmp="$(mktemp /tmp/aide-fl-demo-scenario.XXXXXX.json)"
rendered_scenario_tmp="$(mktemp /tmp/aide-fl-rendered-scenario.XXXXXX.json)"
cleanup() {
  if [[ "$KEEP_SCENARIO" -eq 0 ]]; then
    rm -f "$scenario_tmp"
    rm -f "$rendered_scenario_tmp"
  fi
}
trap cleanup EXIT

python render_scenario.py "$SCENARIO_PATH" -o "$rendered_scenario_tmp"

clients_json_array="$(printf '%s\n' "${connected_client_ids[@]}" | jq -R . | jq -s .)"

jq \
  --arg suffix "$RUN_NAME_SUFFIX" \
  --argjson client_ids "$clients_json_array" \
  '
  .name = (.name + "-" + $suffix)
  | if (.datasets | type) == "object" then
      .datasets =
        (
          .datasets
          | to_entries
          | sort_by(.key)
          | [range(0; length) as $i | {key: $client_ids[$i], value: .[$i].value}]
          | from_entries
        )
    else
      .
    end
  ' "$rendered_scenario_tmp" >"$scenario_tmp"

echo "Demo target"
echo "  Server URL:   $SERVER_URL"
echo "  Scenario:     $SCENARIO_PATH"
echo "  Temp file:    $scenario_tmp"
echo "  Run suffix:   $RUN_NAME_SUFFIX"
echo "  Clients:"
for client_id in "${connected_client_ids[@]}"; do
  echo "    - $client_id"
done
echo

echo "Materialized scenario summary"
jq '{name, dataset, datasets, hold_out_dataset, centralised, hyper, strategy}' "$scenario_tmp"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "Dry run requested. Scenario was materialized but not submitted."
  exit 0
fi

echo "Submitting run via ${SERVER_URL}/start"
start_response="$(curl -fsS -X POST -H 'Content-Type: application/json' --data @"$scenario_tmp" "${SERVER_URL}/start")"
echo "$start_response" | jq .

run_id="$(jq -r '.data["run-id"] // empty' <<<"$start_response")"
if [[ -z "$run_id" || "$run_id" == "null" ]]; then
  echo "Could not extract run-id from /start response" >&2
  exit 1
fi

echo
echo "Polling run status for run-id: $run_id"
start_epoch="$(date +%s)"

while true; do
  now_epoch="$(date +%s)"
  elapsed="$((now_epoch - start_epoch))"
  if [[ "$elapsed" -gt "$TIMEOUT_SECONDS" ]]; then
    echo "Timed out after ${TIMEOUT_SECONDS}s while waiting for run $run_id" >&2
    exit 1
  fi

  status_response="$(curl -fsS -G --data-urlencode "runId=${run_id}" "${SERVER_URL}/status")"
  run_json="$(jq '.data.runs[0]' <<<"$status_response")"
  status="$(jq -r '.status // "UNKNOWN"' <<<"$run_json")"
  created_at="$(jq -r '."created-at" // ""' <<<"$run_json")"
  running_at="$(jq -r '."running-at" // ""' <<<"$run_json")"
  finished_at="$(jq -r '."finished-at" // ""' <<<"$run_json")"
  runtime_elapsed="$(jq -r '.elapsed // ""' <<<"$run_json")"

  printf '[%4ss] status=%s elapsed=%s created=%s running=%s finished=%s\n' \
    "$elapsed" "$status" "${runtime_elapsed:-}" "${created_at:-}" "${running_at:-}" "${finished_at:-}"

  if [[ -n "$finished_at" && "$finished_at" != "null" && "$finished_at" != "N/A" ]]; then
    echo
    echo "Run finished"
    echo "$status_response" | jq .
    break
  fi

  sleep "$POLL_INTERVAL"
done

echo
echo "Next places to show during the demo"
echo "  Clients:  ${SERVER_URL}/clients"
echo "  Status:   ${SERVER_URL}/status?runId=${run_id}"
echo "  MinIO:    http://${server_ip}:9001"
