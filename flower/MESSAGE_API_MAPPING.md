# Flower Message API Mapping

This note captures how the current AIDE-FL Flower integration should map onto
the Flower `1.27.x` app/runtime model.

It is intentionally specific to this repository, not a generic Flower summary.

## Current State

The current implementation is partly migrated:

- server-side control flow still uses a Flask shim in `flower/main.py`
- the shim still shells out to `flwr run` and `flwr ls`
- `flower/flower/client_app.py` now uses `ClientApp.train` /
  `ClientApp.evaluate` with explicit `Message` handling
- `flower/flower/server_app.py` now uses `ServerApp.main` +
  `serverapp.strategy.FedAvg.start(...)`
- framework-specific behavior is still mixed directly into shared helpers in
  `flower/flower/task.py`
- the TensorFlow path still carries Keras-specific payload fields such as
  `model_data` inside `ConfigRecord`

This is a better Flower `1.27.x` shape for the TensorFlow baseline, but it is
still not the final multi-framework architecture.

## Target Flower Concepts

The Flower `1.27.x` concepts that matter for AIDE-FL are:

- `Context`
- `Message`
- `RecordDict`
- `ArrayRecord`
- `ConfigRecord`
- `MetricRecord`
- `ClientApp`
- `ServerApp`
- `serverapp.strategy.FedAvg`

## AIDE-FL To Flower Mapping

### Context

Flower `Context` should carry:

- `run_config`
  - static Flower app config such as `verbose`
  - framework-level runtime defaults that are not scenario-specific
- `state`
  - per-client cached training state that survives across rounds
  - candidate contents:
    - current framework identifier
    - cached model definition
    - cached local train/eval/test load state
    - most recent scenario-derived client config

What should *not* live primarily in `Context.run_config`:

- scenario payload from `server-logic`
- dataset split definitions
- model architecture JSON as an ad hoc transport channel

Those are run payload concerns and should travel in records/messages.

### Message

Flower `Message` should be the transport envelope for train/evaluate/query
interactions between `ServerApp` and `ClientApp`.

For AIDE-FL:

- `TRAIN`
  - carries global model state plus client training config
- `EVALUATE`
  - carries global model state plus evaluation config
- `QUERY`
  - can later support capabilities discovery or framework metadata

The Flask shim is still currently responsible for `/run` and `/ls`, but inside
the Flower app itself the server/client boundary should be described in terms of
`Message`, not TensorFlow helper callbacks.

### RecordDict

`RecordDict` should be the standard payload container in both directions.

Recommended contract for AIDE-FL:

- `"arrays"`
  - federated parameter/state payload
- `"config"`
  - train/evaluate configuration payload
- `"metrics"`
  - numeric metrics returned by clients or server-side evaluation

Additional records should be introduced only when there is a clear need, such
as non-array framework state or richer debugging output.

### ArrayRecord

`ArrayRecord` should represent the federated model/state payload.

Per framework:

- TensorFlow
  - Keras weights as numpy ndarray lists
- scikit-learn
  - estimator coefficients/intercepts converted to ndarrays
- XGBoost
  - likely serialized booster state represented explicitly as record payload

Important implication:

- AIDE-FL should stop assuming the federated state is always “Keras weights”
- `scenario.model` should define how to build the local model, not how the
  federated transport is represented

### ConfigRecord

`ConfigRecord` should carry run-specific instructions from server to client.

For the current TensorFlow path this should eventually include:

- `model_data`
- `local_epochs`
- `batch_size`
- `label_column`
- framework identifier once scenarios become explicit about framework choice

This is the correct place for scenario-derived train/evaluate configuration that
must be sent to each client.

### MetricRecord

`MetricRecord` should carry numeric results.

For the current TensorFlow path this should include:

- `num-examples`
- evaluation loss
- evaluation metrics such as accuracy, precision, recall
- compact train summaries where useful

Important limitation for migration planning:

- `MetricRecord` is for metric-like values, not arbitrary JSON strings
- the current `"history"` JSON payload used in the legacy `NumPyClient` path
  should not be treated as a long-term `MetricRecord` contract

If per-epoch history remains important, it should be handled deliberately, not
smuggled through as a legacy metrics string.

## Runtime Concerns Versus App Concerns

The repo should distinguish clearly between Flower runtime concerns and AIDE-FL
application concerns.

### Flower Runtime Concerns

These are Flower/deployment concerns:

- `SuperLink` and `SuperNode`
- TLS certificates and root trust
- Flower config in `.flwr/config.toml`
- `flower-superexec`, `flwr run`, `flwr ls`
- `ClientApp` / `ServerApp` registration and startup
- Message transport between server and client apps

### AIDE-FL Application Concerns

These are AIDE-FL concerns:

- scenario ingestion in `server-logic`
- dataset preparation and caching
- parquet split loading
- framework-specific model construction
- framework-specific train/evaluate behavior
- artifact upload to MinIO
- Weights & Biases logging policy

## Recommended Record Keys

To keep the migration consistent, use these keys unless there is a strong
reason to diverge:

- `arrays`
- `config`
- `metrics`
- `num-examples`

These align with Flower `1.27.x` strategy defaults, including
`serverapp.strategy.FedAvg`.

## Immediate Migration Implications

The next code steps should now follow this order:

1. Keep removing runtime behavior that depends on legacy Flower CLI/config
   assumptions.
2. Stabilize the current TensorFlow Message API path with live runs on-cluster.
3. Split framework adapters away from the TensorFlow baseline so
   `flower/flower/task.py` stops being the shared framework choke point.
4. Replace TensorFlow-specific config payload assumptions with a framework-aware
   scenario/model contract.
5. Add the next framework on top of the adapter boundary, starting with
   scikit-learn.
