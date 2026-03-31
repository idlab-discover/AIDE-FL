# Current Plan

## Baseline Status

The platform baseline remains strong enough to support a deeper Flower refactor.

- The cluster is reachable through `~/.kube/aide`.
- Custom images can be published to GHCR by immutable SHA tag when using the repo's `buildx --push` flow.
- Preparation abort and retry behavior has been validated on-cluster.
- Stale Kafka backlog is filtered by `prep_id`.
- `server-deployment` uses `Recreate`, which matches its `hostNetwork` port constraints.
- `server-logic` handles client disconnect during preparation without panicking.
- `orchestrator` now reconnects automatically after server replacement.
- Flower has been lifted in-repo to `1.27.x`, but the live path still needs one final compatibility pass and end-to-end validation.

## Primary Objective

Bring AIDE-FL into line with the Flower `1.27.x` application model while preserving current TensorFlow capability and opening a clean path for additional ML frameworks.

This means:

- stop treating Flower as a TensorFlow-only integration
- stop treating scenario `model` as implicitly "Keras JSON"
- separate framework concerns from aggregation concerns
- align explicitly with Flower's public `flwr.app`, `flwr.clientapp`, and `flwr.serverapp` APIs
- use Flower's Message API primitives as the first-class integration boundary
- preserve deployment-runtime support on the live Kubernetes cluster

Reference docs for this phase:

- Flower API root
  - https://flower.ai/docs/framework/ref-api/flwr.html
- `flwr.app`
  - https://flower.ai/docs/framework/ref-api/flwr.app.html#module-flwr.app
- `flwr.clientapp`
  - https://flower.ai/docs/framework/ref-api/flwr.clientapp.html#module-flwr.clientapp
- `flwr.serverapp`
  - https://flower.ai/docs/framework/ref-api/flwr.serverapp.html#module-flwr.serverapp
- runtime comparison
  - https://flower.ai/docs/framework/ref-flower-runtime-comparison.html
- network communication
  - https://flower.ai/docs/framework/ref-flower-network-communication.html
- exit codes
  - https://flower.ai/docs/framework/ref-exit-codes-dir.html
- TensorFlow quickstart
  - https://flower.ai/docs/framework/tutorial-quickstart-tensorflow.html
- scikit-learn quickstart
  - https://flower.ai/docs/framework/tutorial-quickstart-scikitlearn.html
- XGBoost quickstart
  - https://flower.ai/docs/framework/tutorial-quickstart-xgboost.html
- tutorial series index
  - https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html

Relevant Flower `1.27.x` API surfaces to align with:

- `flwr.app`
  - `ArrayRecord`
  - `ConfigRecord`
  - `MetricRecord`
  - `Message`
  - `Metadata`
  - `RecordDict`
  - `Context`
  - `UserConfig`
- `flwr.clientapp`
  - `ClientApp([client_fn, mods])`
- `flwr.serverapp`
  - `ServerApp([server, config, strategy, ...])`
  - `Grid`
  - `serverapp.strategy`

## Target Architecture

The intended end state is:

- Scenario selects a `framework` explicitly.
- Scenario model definition becomes framework-aware instead of Keras-only.
- `ServerApp` selects a framework adapter based on scenario contents.
- `ClientApp` selects the matching framework adapter on each client.
- framework state and training instructions flow through `Message` / `RecordDict` instead of ad hoc JSON contracts wherever practical
- Aggregation strategy selection is independent from framework selection.
- Artifact saving/logging is independent from both framework and aggregation strategy.

## Ordered Work Plan

### 1. Finish Flower `1.27.x` stabilization

Goal:

- reach a clean live-cluster baseline on Flower `1.27.x`

Tasks:

- finish validating the current `flwr 1.27.x` shim behavior on `/run` and `/ls`
- confirm the live cluster is running the fixed `flower-serverapp` and `flower-clientapp` images
- rerun `examples/nids-demo-live.json` successfully on the upgraded stack
- record any remaining Flower `1.27.x` incompatibilities in the server shim or deployment model

Exit criteria:

- `/clients` works
- `/status` works
- `/start` works
- `examples/nids-demo-live.json` completes successfully against Flower `1.27.x`

### 2. Move off legacy Flower patterns

Goal:

- align AIDE-FL with the modern Flower app model instead of continuing to build around older `NumPyClient` assumptions

Tasks:

- review where current code still assumes old Flower CLI/JSON behavior
- replace legacy assumptions in the Python shim with explicit error handling and Flower `1.27.x`-compatible behavior
- migrate from the current `NumPyClient`-centric client path toward the `ClientApp` + Message API pattern used in current Flower quickstarts
- document which parts of the current stack should map to:
  - `Context`
  - `Message`
  - `RecordDict`
  - `ArrayRecord`
  - `ConfigRecord`
  - `MetricRecord`
- document which parts of the current stack are Flower-runtime concerns versus application concerns

Exit criteria:

- no critical control path depends on pre-`1.27` CLI output shape
- the app structure matches Flower `1.27.x` concepts closely enough to add new frameworks cleanly
- the core server/client interface can be described in Flower Message API terms rather than TensorFlow-specific helper behavior

### 3. Introduce a framework-aware scenario schema

Goal:

- make scenario definition explicit and readable for multiple ML frameworks

Tasks:

- add an explicit top-level `framework` field to scenarios
- stop overloading `model` to always mean Keras architecture JSON
- define a cleaner model section, for example:
  - `framework`
  - `model.kind`
  - `model.config`
  - `training.config`
  - `aggregation.strategy`
- preserve support for current TensorFlow/Keras scenarios during migration
- update validation so framework-specific requirements are enforced by schema and service logic

Exit criteria:

- new scenario format can represent TensorFlow, scikit-learn, and XGBoost without awkward Keras-specific placeholders
- current TensorFlow scenarios can be migrated mechanically

### 4. Introduce a framework adapter boundary

Goal:

- isolate framework-specific training/evaluation/state-handling logic behind a stable interface

Tasks:

- define a framework adapter interface for:
  - model loading
  - initial parameter/state creation
  - local train
  - local evaluate
  - serialization/deserialization of federated state
  - artifact export
- define how each adapter maps onto Flower primitives:
  - model/state payload in `ArrayRecord` or another `RecordDict` member
  - training/evaluation config in `ConfigRecord`
  - result metrics in `MetricRecord`
  - run and node metadata through `Context`
- move TensorFlow-specific code out of shared helpers like `flower/task.py`
- split shared dataset loading from framework-specific model/state handling
- ensure server-side hold-out evaluation can be implemented per framework instead of only for Keras models

Exit criteria:

- TensorFlow logic no longer leaks into generic Flower application plumbing
- adding a second framework does not require editing the same TensorFlow-specific code paths everywhere
- framework adapters are defined in terms of Flower `RecordDict`/`Message` exchange, not in terms of one framework's in-memory object model

### 5. Refactor TensorFlow onto the adapter model

Goal:

- make TensorFlow the first framework implemented on the new abstraction

Tasks:

- move current Keras model loading into a TensorFlow adapter
- keep support for flexible Keras architectures, including existing `model_file` flow
- preserve current preprocessing and parquet-based data path
- map TensorFlow parameter exchange to Flower `ArrayRecord` explicitly
- map TensorFlow training configuration to `ConfigRecord`
- map TensorFlow evaluation output to `MetricRecord`
- preserve current output artifacts where practical:
  - per-round checkpoints
  - final model artifact
  - metrics JSON
- decide which TensorFlow artifact formats remain first-class:
  - Keras JSON
  - weights
  - safetensors

Exit criteria:

- TensorFlow scenarios still run end-to-end after the refactor
- TensorFlow is no longer the implicit default baked into shared code

### 6. Separate aggregation strategy from framework implementation

Goal:

- make future strategies possible without entangling them with TensorFlow assumptions

Tasks:

- split the current `SaveModelFedAvg` responsibilities into:
  - aggregation strategy
  - artifact saver
  - metrics/logging sink
- ensure strategy selection is scenario-driven
- keep `FedAvg` as the first stable strategy
- model the strategy layer in terms of Flower `serverapp.strategy` concepts
- identify which strategy hooks need to be generalized for:
  - FedAvg variants
  - framework-specific parameter/state formats
  - future secure aggregation work

Exit criteria:

- changing the aggregation strategy does not require rewriting framework code
- changing the framework does not require rewriting generic strategy plumbing

### 7. Add scikit-learn as the second framework

Goal:

- prove that the new architecture is genuinely multi-framework, not just a TensorFlow cleanup

Tasks:

- implement a scikit-learn adapter for a small initial estimator set
- start with simple tabular estimators that map naturally onto current parquet preprocessing
- decide how model definitions should be represented in scenarios for sklearn
- define the sklearn adapter around Flower record exchange rather than TensorFlow helper reuse
- add at least one example scenario and one end-to-end validation path

Exit criteria:

- at least one scikit-learn scenario runs successfully through the same deployment pipeline
- sklearn support does not require TensorFlow/Keras artifacts or assumptions

### 8. Add XGBoost after sklearn

Goal:

- support a framework whose federated state and training semantics differ more substantially from dense neural-network weights

Tasks:

- design an XGBoost adapter around serialized booster state
- determine whether XGBoost needs a dedicated strategy path rather than reusing the TensorFlow/sklearn parameter flow
- align XGBoost state handling with Flower's XGBoost strategy family where appropriate, especially `FedXgbBagging` and `FedXgbCyclic`
- add scenario representation for XGBoost models and training config
- validate end-to-end on at least one XGBoost scenario

Exit criteria:

- XGBoost support works without pretending boosters are Keras-like weight arrays
- the architecture still reads cleanly after adding a structurally different framework

## Supporting Work

These items support the migration but should not drive the architecture by themselves:

- improve docs around Flower runtime concepts, especially Deployment Runtime vs Simulation Runtime
- add tests around scenario validation and framework selection
- add tests around server shim behavior for Flower CLI failures
- add tests around `Message` / `RecordDict` contract handling for the Flower app boundary
- add image and rollout verification steps to cluster notes once the `1.27.x` path is stable

## Security and Research Roadmap

This migration should preserve room for later work on:

- differential privacy
- multi-party computation
- fully homomorphic encryption
- richer federated aggregation than FedAvg

Flower-native seams that should remain available for later work:

- `clientapp.mod`
  - for example built-in mods such as clipping, message sizing, and local DP support
- `serverapp.strategy`
  - including DP-oriented and XGBoost-oriented strategies already present in Flower `1.27.x`

These concerns should attach to clean seams:

- scenario definition
- framework adapter boundary
- aggregation strategy boundary
- weight/state transport boundary

## Immediate Next Step

The immediate next step is to finish the Flower `1.27.x` live validation and then begin the scenario-schema and framework-adapter design from a stable upgraded baseline.
