## Commit Hygiene

- Keep commits clean, bounded, and purpose-specific.
- Prefer one logical change per commit. Do not mix unrelated fixes, refactors, docs updates, and test rewrites unless they are tightly coupled.
- When work spans multiple concerns, split it into a short chain of commits with readable messages.
- Before committing, check that the diff matches the stated purpose of the commit and does not include unrelated workspace noise.
- If a change is exploratory or lower confidence, prefer using a separate branch until it is trusted.

## Cluster Context

- Verified Kubernetes access is available via kubeconfig at `~/.kube/aide`.
- Verified control plane endpoint: `https://10.10.148.109:6443`.
- Active application namespace: `aide-fl`.
- Current node roles:
  - `cluster-laurens-md-0-mjpnq-f9lkf-4j7wr` at `10.10.148.254` with label `fl-role=server`
  - `cluster-laurens-md-0-mjpnq-f9lkf-84lr2` at `10.10.148.246` with label `fl-role=client`
  - `cluster-laurens-md-0-mjpnq-f9lkf-wtt5q` at `10.10.148.115` with label `fl-role=client`
  - `cluster-laurens-r2klm-c5jtw` at `10.10.148.110` is the control-plane node and is not labeled for AIDE-FL workloads
- Verified live AIDE-FL workloads in `aide-fl`:
  - server-side: `server-deployment`, `minio-server-deployment`
  - client-side daemonsets: `kafka-daemonset`, `datastore-daemonset`, `orchestrator-daemonset`, `preprocessor-daemonset`, `trainer-daemonset`
- The deployment uses node-local networking patterns:
  - `hostNetwork: true` is used broadly
  - client orchestrators connect to the server over `ws://10.10.148.254:80/ws`
  - the server HTTP API is reachable at `http://10.10.148.254/`
- Verified current server API state during repository familiarization:
  - `GET /clients` returned 2 connected clients
  - `GET /status` returned no active runs at that time
- When experimenting on-cluster, assume the namespace may already contain a live baseline deployment. Avoid unnecessary redeployments or destructive resets unless explicitly intended.

## Go Guidance

- This repository contains substantial Go code, especially in `server/server/server-logic`, `client/orchestrator`, and `client/datastore/datastore-logic`.
- When writing, reviewing, debugging, or refactoring Go code in this repository, use the `effective-go` skill as the default guidance for idiomatic structure, naming, error handling, simplicity, and code review standards.
- Prefer clear, idiomatic Go over patterns imported from other languages. Keep interfaces minimal, error paths explicit, and concurrency behavior easy to reason about.
