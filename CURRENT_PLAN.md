# Current Plan

## Goal

Get AIDE-FL to one reproducible, end-to-end baseline run on the target Kubernetes cluster without changing Flower's core transport model, then use that stable baseline to drive operational hardening and architectural simplification.

This plan is ordered around the current state of the repository and the first deployment findings. The key change is that proven application-level defects should be fixed before spending more cluster time on baseline validation.

## Current State Summary

### Architecture

- Server side:
  - `server-logic` exposes the public HTTP API.
  - Client orchestrators connect to `server-logic` over WebSocket.
  - Flower server components run beside `server-logic`.
  - Server MinIO stores datasets, models, checkpoints, and metrics.
- Client side:
  - `orchestrator` receives scenarios and coordinates the local preparation flow.
  - `datastore-logic` ensures datasets exist in MinIO or downloads them from Kaggle.
  - `preprocessor-logic` preprocesses datasets and caches train/eval/test parquet blobs in Redis.
  - `trainer-logic` pulls cached parquet blobs into the shared Flower volume and validates training compatibility.
  - Flower client components run beside `trainer-logic`.
  - A node-local Kafka broker is used for the internal client control path.

### Deployment findings already verified

- Docker Hub pulls initially failed because of broken worker-side containerd registry configuration.
- That infrastructure issue is documented in [CLUSTER_NOTES.md](/home/strgenix/postdoc/projects/AIDE-FL/CLUSTER_NOTES.md).
- The stack can be scheduled with one worker labeled `fl-role=server` and two workers labeled `fl-role=client`.
- Kafka now starts successfully with the official `apache/kafka` image.
- `server`, `orchestrator`, and `datastore` can come up.
- `preprocessor` and `trainer` are currently blocked by worker disk pressure during image extraction.
- The worker disks are too small in their current state and are expected to be expanded to `30G`.

### Code-level findings already verified

- `server-logic` has a race in `/start` admission:
  - two concurrent `/start` requests can both pass the "not training" check before the training flag is set.
- `server-logic` has broken reconnect accounting for clients:
  - reconnecting clients can inflate `totalClients`
  - stale disconnect handlers can delete the active client entry
  - preparation can then hang or abort incorrectly
- `trainer-logic` makes a brittle schema assumption:
  - feature count validation subtracts an assumed extra parquet index column
  - valid datasets can be rejected
- `preprocessor-logic` uses exact float equality for split validation:
  - semantically valid split configurations can be rejected due to float behavior

## Constraints

- Do not change Flower's core communication semantics for the baseline.
- The cluster has no GPUs.
- Image size and runtime footprint matter immediately because worker disks are tight.
- Differential privacy is a realistic near-term research extension.
- Secure aggregation, MPC, and homomorphic encryption should be treated as later phases.

## Execution Order

1. Fix the proven application-level correctness issues.
2. Re-run the deployment after worker disk expansion.
3. Achieve one clean end-to-end baseline training run.
4. Add deployment hardening and verification tooling.
5. Reduce image footprint.
6. Only then start larger refactors or a Flower upgrade.

## Phase 1: Immediate Code Fixes

### Objective

Remove the defects that are already visible in code and are likely to distort the next cluster baseline.

### Work items

1. Fix `server-logic` training admission.
   - Make `/start` admission atomic.
   - Ensure only one training preparation flow can begin at a time.
   - Prevent concurrent requests from sharing mutable controller state.

2. Fix `server-logic` client lifecycle accounting.
   - Handle reconnects for the same client ID safely.
   - Keep `clients` and `totalClients` consistent.
   - Ensure a stale connection cannot remove a newer active connection.
   - Re-check preparation completion logic against the corrected client set.

3. Fix trainer dataset/model validation.
   - Stop assuming an extra synthetic parquet column always exists.
   - Compute feature count from real schema names.
   - Validate against the declared label column explicitly.

4. Fix preprocessing split validation.
   - Replace exact `train + eval + test == 1.0` logic with tolerant validation.
   - Reject clearly invalid splits while accepting numerically equivalent values.

5. Add or tighten targeted verification.
   - Build and verify the affected Go service.
   - Add targeted Python tests or lightweight validation scripts where practical.
   - Confirm that the fixes match the existing message flow and deployment model.

### Deliverable

A codebase that no longer contains the known request-level and validation-level defects before the next cluster run.

## Phase 2: Baseline Cluster Run

### Precondition

Worker disk expansion to `30G` is complete.

### Objective

Confirm that the current architecture can complete one real training run after the code-level blockers are removed.

### Work items

1. Re-run or resume deployment on the target cluster.
2. Verify pod readiness for:
   - `server`
   - `kafka`
   - `orchestrator`
   - `datastore`
   - `preprocessor`
   - `trainer`
3. Verify client registration through `/clients`.
4. Run one small baseline tabular scenario.
5. Record the actual sequence for:
   - orchestrator WebSocket connection
   - Kafka topic flow
   - dataset lookup and download
   - Redis cache creation
   - trainer preparation
   - Flower training start and completion
6. Check `/start` and `/status` behavior under normal operation.

### Deliverable

One documented, reproducible, end-to-end baseline run.

## Phase 3: Operational Hardening

### Objective

Make deployments easier to repeat, diagnose, and recover.

### Work items

1. Add a root-level teardown script.
2. Add a deployment verification script covering:
   - node labels
   - namespace
   - prerequisites and secrets
   - deployment and daemonset readiness
   - expected client connectivity
3. Improve `deploy.sh` so it:
   - waits for all critical components, not only Kafka
   - surfaces image pull and disk pressure failures clearly
   - can be resumed safely
4. Document expected recovery procedures for:
   - partial deployments
   - stuck preparation
   - pod restarts
   - client reconnects

### Deliverable

A baseline deployment workflow that is inspectable and repeatable without ad hoc manual recovery.

## Phase 4: Image Footprint Reduction

### Objective

Shrink controlled images enough that worker storage stops being a recurring operational risk.

### Priority images

- `preprocessor-logic`
- `trainer-logic`
- `flower-clientapp`
- `flower-serverapp`

### Work items

1. Audit Python dependencies in each image.
2. Remove packages not required for the baseline path.
3. Use slimmer base images where practical.
4. Re-check image pull, extract, and pod startup behavior after size reductions.

### Deliverable

A materially smaller baseline deployment footprint.

## Phase 5: Post-Baseline Refactor Direction

This phase begins only after the baseline system is stable and deployment behavior is understood.

### Preferred direction

Collapse the client-side control path into a single client agent per node that performs:

- dataset acquisition
- preprocessing and caching
- trainer preparation
- server signaling

Keep Flower components separate if necessary, but remove Kafka from the internal client orchestration path.

### Lower-risk intermediate direction

If the current service split is temporarily preserved, replace Kafka with a simpler local communication mechanism.

### Alternative packaging direction

Keep logical separation, but move the client control services into a single pod per FL client with explicit local IPC and shared lifecycle management.

## Data and Identity Improvements

These should follow the stable baseline, not precede it.

### Goals

- Separate scenario identity from run identity.
- Stop using freeform scenario names as the main technical key.
- Standardize dataset access patterns across centralized and distributed modes.

### Proposed direction

- Introduce a deterministic scenario fingerprint derived from:
  - dataset selection
  - preprocessing configuration
  - hyperparameters
  - model definition
- Use that fingerprint for:
  - cache keys
  - artifact grouping
  - reproducibility checks
- Keep the human-readable `name` as a label, not the primary key.

## Security Roadmap

1. Stable baseline training
2. Better scenario and dataset identity
3. Flower upgrade
4. Differential privacy
5. Secure aggregation or MPC-style aggregation
6. Only then revisit homomorphic encryption feasibility

Differential privacy is the most realistic early security enhancement for this codebase and cluster profile.

## Flower Upgrade Position

- The repository is on Flower `1.18.0`.
- The upgrade should be isolated from both:
  - baseline deployment recovery
  - client-architecture refactoring
- Do not mix Flower migration work with initial stability work.

## Immediate Next Moves

1. Implement the Phase 1 code fixes in:
   - `server-logic`
   - `trainer-logic`
   - `preprocessor-logic`
2. Verify those fixes locally as far as the environment allows.
3. Once worker disks are expanded, execute the next baseline deployment pass.
4. Record the resulting runtime behavior before selecting any refactor track.
