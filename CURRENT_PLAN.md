# Current Plan

## Baseline Status

The baseline is now in good shape.

- The cluster is reachable through `~/.kube/aide`.
- Custom images can be published to GHCR and deployed by immutable SHA tag.
- The stack has been run successfully on updated images on March 30, 2026.
- `examples/nids-demo-live.json` completed successfully on the updated cluster.
- Induced-failure retry behavior has been tested on-cluster:
  - a client-side preparation disruption aborted the first run
  - the retry completed successfully
  - stale Kafka backlog was ignored by `prep_id`
- `server-deployment` now uses `Recreate`, which matches its `hostNetwork` port constraints.
- `server-logic` now handles client disconnect during preparation without panicking.
- `/start` now returns a retryable response for preparation aborts instead of a generic `500`.

## Residual Checklist

### 1. Orchestrator Reconnect Fallback

- After replacing the server pod, verify whether `orchestrator` reconnects automatically.
- If it does not, keep the operational fallback:
  - restart `orchestrator-daemonset`
  - verify `/clients` returns both clients again

### 2. Recovery Procedure

Keep a short, canonical recovery sequence for operators:

- if server replacement succeeds but `/clients` stays empty:
  - restart `orchestrator-daemonset`
- if a run aborts during preparation:
  - wait for client pods to return
  - verify `/clients`
  - retry with a fresh run
- if needed, inspect:
  - `orchestrator` logs
  - `server-logic` logs
  - `/status`

### 3. Optional Follow-Up Work

These are useful, but no longer baseline blockers:

- continue shrinking the heavy Python and Flower images
- document where duplicate side effects are acceptable under retry
- consider longer-term simplification of the client-side Kafka control path
