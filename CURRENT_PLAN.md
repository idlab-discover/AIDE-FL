# Current Plan

## Baseline Status

The platform baseline is now in a solid state.

- The cluster is reachable through `~/.kube/aide`.
- Custom images can be published to GHCR and deployed by immutable SHA tag.
- The stack has been run successfully on updated images on March 30 and March 31, 2026.
- `examples/nids-demo-live.json` completes successfully on the live cluster.
- Preparation abort and retry behavior has been validated on-cluster.
- Stale Kafka backlog is filtered by `prep_id`.
- `server-deployment` uses `Recreate`, which matches its `hostNetwork` port constraints.
- `server-logic` handles client disconnect during preparation without panicking.
- `/start` returns a retryable response for preparation aborts instead of a generic `500`.
- `orchestrator` now reconnects automatically after server replacement without requiring a daemonset restart.

## Next Focus

The next phase is product and architecture work rather than baseline recovery.

- Rework scenario model definition so Keras architectures remain highly flexible while becoming easier to author and review.
- Revisit model serving and model packaging boundaries so future aggregation strategies are easier to introduce.
- Upgrade Flower to the latest supported version in a controlled way.
- Preserve room for more advanced federated strategies beyond FedAvg.
- Design toward a turnkey secure federated learning platform.

## Security Roadmap

Important longer-term capabilities to keep in scope:

- differential privacy
- multi-party computation
- fully homomorphic encryption

These should be evaluated in a way that preserves clean interfaces around scenario definition, aggregation logic, and weight transport.
