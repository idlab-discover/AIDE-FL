#!/usr/bin/env bash

set -euo pipefail

NAMESPACE="${NAMESPACE:-aide-fl}"
IMAGE_REPO="${IMAGE_REPO:-ghcr.io/idlab-discover/aide-fl}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
COMPONENT="${COMPONENT:-all}"

usage() {
  cat <<EOF
Usage: $0 [--namespace NAME] [--image-repo REPO] [--image-tag TAG] [--component NAME]

Patches the custom AIDE-FL images on existing Kubernetes resources.

Components:
  all
  server
  orchestrator
  datastore
  preprocessor
  trainer
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --image-repo)
      IMAGE_REPO="$2"
      shift 2
      ;;
    --image-tag)
      IMAGE_TAG="$2"
      shift 2
      ;;
    --component)
      COMPONENT="$2"
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

set_images() {
  local resource="$1"
  shift

  kubectl set image -n "$NAMESPACE" "$resource" "$@"
}

if [[ "$COMPONENT" == "server" || "$COMPONENT" == "all" ]]; then
  set_images deployment/server-deployment \
    flower-serverapp="${IMAGE_REPO}/flower-serverapp:${IMAGE_TAG}" \
    server-logic="${IMAGE_REPO}/server-logic:${IMAGE_TAG}"
fi

if [[ "$COMPONENT" == "orchestrator" || "$COMPONENT" == "all" ]]; then
  set_images daemonset/orchestrator-daemonset \
    orchestrator="${IMAGE_REPO}/orchestrator:${IMAGE_TAG}"
fi

if [[ "$COMPONENT" == "datastore" || "$COMPONENT" == "all" ]]; then
  set_images daemonset/datastore-daemonset \
    datastore-logic="${IMAGE_REPO}/datastore-logic:${IMAGE_TAG}"
fi

if [[ "$COMPONENT" == "preprocessor" || "$COMPONENT" == "all" ]]; then
  set_images daemonset/preprocessor-daemonset \
    preprocessor-logic="${IMAGE_REPO}/preprocessor-logic:${IMAGE_TAG}"
fi

if [[ "$COMPONENT" == "trainer" || "$COMPONENT" == "all" ]]; then
  set_images daemonset/trainer-daemonset \
    flower-clientapp="${IMAGE_REPO}/flower-clientapp:${IMAGE_TAG}" \
    trainer-logic="${IMAGE_REPO}/trainer-logic:${IMAGE_TAG}"
fi
