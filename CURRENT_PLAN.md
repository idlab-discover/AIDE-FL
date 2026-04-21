# Current Plan

## Baseline Status

The implementation has moved well past the plan that was previously tracked in
this file.

- Flower `1.27.x` app integration is in place in-repo.
- `flower/flower/client_app.py` uses `ClientApp.train` /
  `ClientApp.evaluate` with `Message` and `RecordDict`.
- `flower/flower/server_app.py` uses `ServerApp.main` and starts strategies
  through `flower/flower/strategies.py`.
- The server-side scenario schema is already framework-aware:
  - top-level `framework`
  - top-level `training`
  - top-level `aggregation`
  - framework-specific blocks for `tensorflow`, `scikit_learn`, and `xgboost`
- Framework adapters already exist under `flower/flower/frameworks/` for:
  - TensorFlow
  - scikit-learn logistic regression
  - XGBoost booster models
- Aggregation strategy selection is already separated from framework adapters in
  `flower/flower/strategies.py`.
- Server-side validation already enforces framework/strategy compatibility and
  covers:
  - TensorFlow strategies including `fedprox`
  - scikit-learn strategy restrictions
  - XGBoost strategy restrictions for `fedxgbbagging` and `fedxgbcyclic`
- The trainer-side validation path is also framework-aware.
- Authored examples currently cover:
  - TensorFlow
  - XGBoost
- The repo does not yet include an authored scikit-learn example scenario.

## What Is Already Complete

The following plan items are no longer future work:

1. Flower `1.27.x` stabilization
2. Move off legacy Flower patterns inside the app layer
3. Introduce a framework-aware scenario schema
4. Introduce a framework adapter boundary
5. Refactor TensorFlow onto the adapter model
6. Separate aggregation strategy from framework implementation

Additional implementation has also landed beyond that original plan:

- initial scikit-learn support is implemented
- initial XGBoost support is implemented

## Current Gaps

The remaining work is no longer core architecture design. It is hardening,
validation, and cleanup around the architecture that now exists.

The main gaps visible in the current codebase are:

- `flower/main.py` still exposes `/run` and `/ls` by shelling out to
  `flwr run` and `flwr ls`
- `server-logic` still normalizes and emits deprecated scenario fields for
  compatibility (`hyper`, top-level `strategy`, top-level `model`)
- `flower/MESSAGE_API_MAPPING.md` still describes scikit-learn and XGBoost as
  future follow-on work even though adapter implementations already exist
- there are direct tests for scenario validation and some service behavior, but
  little direct automated coverage for:
  - `flower/flower/client_app.py`
  - `flower/flower/server_app.py`
  - `flower/flower/strategies.py`
  - adapter-specific Message/Record handling
- XGBoost has an example scenario, but the repo does not yet clearly record an
  end-to-end validation path for the current implementation
- scikit-learn support exists in code, but the repo does not yet include a
  first-class example scenario and validation recipe

## Primary Objective

Harden and validate the already multi-framework Flower integration instead of
continuing to plan foundational architecture work that has already landed.

That means:

- remove remaining legacy compatibility layers
- prove the non-TensorFlow paths end to end
- add direct automated coverage around the Flower app boundary
- bring docs and operator guidance in line with the current implementation

## Ordered Work Plan

### 1. Remove remaining legacy compatibility layers

Goal:

- make the runtime consistently speak the current app/schema model without
  unnecessary legacy backfill

Tasks:

- isolate or replace the `flower/main.py` Flask shim behavior that still shells
  out to `flwr run` and `flwr ls`
- remove `runtimeCompatibleScenario` backfill once no downstream consumer still
  needs deprecated fields
- confirm the client-side services only depend on the current scenario shape
- align scenario transport and runtime control flow with the implementation that
  already exists

Exit criteria:

- no internal service requires deprecated scenario fields
- runtime control flow no longer depends on brittle CLI JSON parsing, or that
  dependency is isolated behind explicit tests
- the actual runtime path matches the schema and adapter model described by the
  repo

### 2. Add direct automated coverage for the Flower app boundary

Goal:

- test the multi-framework Flower integration where it currently has the least
  direct coverage

Tasks:

- add unit tests for `flower/flower/client_app.py` request/reply handling
- add unit tests for `flower/flower/server_app.py` framework selection and
  hold-out evaluation wiring
- add unit tests for `flower/flower/strategies.py` strategy construction and
  metrics aggregation
- add adapter-focused tests for TensorFlow, scikit-learn, and XGBoost record
  handling

Exit criteria:

- Message / `RecordDict` contract handling is tested directly
- strategy selection is covered for every supported framework/strategy pairing
- non-TensorFlow adapters have direct tests beyond schema validation

### 3. Add a scikit-learn example and validation path

Goal:

- move scikit-learn support from "implemented" to "demonstrated"

Tasks:

- add an authored scikit-learn scenario under `examples/`
- document the expected dataset shape and class configuration
- validate at least one end-to-end scikit-learn run
- document the resulting artifacts and metrics shape

Exit criteria:

- the repo contains a first-class scikit-learn example scenario
- the scikit-learn path is validated end to end

### 4. Harden XGBoost validation and docs

Goal:

- move XGBoost support from "implemented" to "well validated"

Tasks:

- validate the current XGBoost example against the present runtime
- document supported XGBoost strategy choices and constraints
- document expected artifact outputs and evaluation behavior
- add coverage for booster-state-specific failure modes where practical

Exit criteria:

- the XGBoost path is validated end to end
- the supported XGBoost contract is explicit in repo docs

### 5. Bring docs and operator guidance up to date

Goal:

- make the repository documentation match what the code now does

Tasks:

- update `flower/MESSAGE_API_MAPPING.md` so it no longer describes current
  frameworks as future work
- keep `README.md`, `CURRENT_PLAN.md`, and example/schema docs aligned
- document a repeatable validation recipe for the key Go and Python test paths
- record current image and rollout verification steps for cluster use

Exit criteria:

- the repo's planning and architecture docs describe the current implementation
  accurately
- validation and rollout steps are repeatable by another operator

## Security and Research Roadmap

This refactor should continue to preserve clean seams for later work on:

- differential privacy
- multi-party computation
- fully homomorphic encryption
- richer aggregation strategies

The seams that should stay clean are:

- scenario definition
- framework adapter boundary
- aggregation strategy boundary
- state transport boundary

## Immediate Next Step

Start with item 1: remove or isolate the remaining legacy runtime and scenario
compatibility layers, then add direct Flower app tests before expanding
framework demos and docs.
