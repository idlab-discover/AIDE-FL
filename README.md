# AIDE-FL

Federated Learning on K8s in a modular and reproducable fashion. Every node works independently from another. All pods use the node localhost network to communicate.

## Installation

The `deploy.sh` script in the root of the repo should guide you through the necessary steps to get this framework up and running on a Kubernetes (k8s) cluster.

The only requirement for the k8s cluster is to have at least two worker nodes. Three is recommended.

```sh
git clone https://github.com/idlab-discover/AIDE-FL.git

cd AIDE-FL

./deploy.sh
```

The scripts guides you through the following steps:

- `kubectl`, `jq` and `openssl` should be installed.
- A kubeconfig context must be set which points to the correct k8s instance.
- There should be exactly one k8s node with label `fl-role=server` and at least one node with label `fl-role=client`.
- The `certs` folder is created containing certificates for secure Flower TLS communication.
- W&B runs offline by default. Supplying `WANDB_API_KEY` selects online logging; `--wandb-mode disabled` disables it.
- For an empty cluster, the script creates the `aide-fl` namespace and deploys all pods.

Rerunning the script applies updates and preserves existing Flower certificates and W&B credentials. Each workload rollout has a timeout, controlled by `ROLLOUT_TIMEOUT` with a default of `300s`. Use `--yes` to skip the cluster confirmation.

If you want to deploy a specific published image set instead of whatever `:latest` currently points to, pass an explicit image tag:

```sh
./deploy.sh --image-tag sha-<git-sha>
```

The deploy script sets the custom image tags before applying workloads.

If port 80 is already occupied, for example by k3s Traefik, use `--api-port 8081`. Pass `--server-url http://<server-node-ip>:8081` to `health-check.sh` and `demo-runner.sh`. On an intentionally shared control-plane node, set `ALLOW_CONTROL_PLANE=true` when running these checks. The default rejects application labels on control-plane nodes. For a two-node smoke run, use `--expected-clients 1` and a scenario with `min_available_clients` set to `1`. This checks execution only; it does not establish multi-client learning performance.

## Container packages

The canonical repository is https://github.com/idlab-discover/AIDE-FL. GitHub Actions builds and publishes ten Linux amd64 container packages under `ghcr.io/idlab-discover/aide-fl`:

- `server-logic`, `orchestrator`, `datastore-logic`, `preprocessor-logic`, `trainer-logic`, `minio`
- `flower-serverapp`, `flower-clientapp`
- `flower-serverapp-base`, `flower-clientapp-base`

The workflow runs on `main`, version tags and manual dispatch. Every build publishes `sha-<full-40-character-commit>` tags. Default-branch builds also publish `latest`; version-tag builds publish the version tag. The Flower applications use base images from the same commit. Registry authentication uses the repository's `GITHUB_TOKEN`.

Deploy a retained revision with:

```sh
./deploy.sh --kubeconfig /path/to/config \
  --image-repo ghcr.io/idlab-discover/aide-fl \
  --image-tag sha-<full-40-character-commit>
```

The packages must be public to allow anonymous pulls. Package administrators set this once in GitHub package settings after the initial publication. Verify anonymous access before distributing a release. The deployment also pulls upstream Flower, Kafka and Redis images.

The `minio` package builds official MinIO source release `RELEASE.2025-10-15T17-29-55Z`, pinned and checked at commit `9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a`. Upstream Community Edition container distribution is no longer available; its repository is archived. This package is an AIDE-FL build, not an upstream-maintained image. MinIO remains licensed under GNU AGPLv3, separately from AIDE-FL; the image includes its LICENSE and CREDITS in `/usr/share/licenses/minio`. Corresponding source is available at https://github.com/minio/minio/tree/9e49d5e7a648f00e26f2246f4dc28e6b07f8c84a; the build recipe is `minio/Dockerfile`.

The initial Python implementation remains in Git history. Its final original revision is `11183c52bac6ee9ad55c7e10760a33e24a203221`; the service-based implementation supersedes it on `main`.

## Scenarios

Scenarios are the driving factor behind this framework. They include the required dataset name, preprocessing steps, training settings, aggregation settings, and framework-specific model definition. It is sent from the central server to each client which prepare themselves for training. Once they are all done preparing, the server will start the training process.

A scenario follows a certain JSON template:

```json
{
  "name": "string",
  "framework": "string",
  "dataset": "string",
  "datasets": {
    "string": "string"
  },
  "hold_out_dataset": "string",
  "centralised": bool,
  "label_column": "string",
  "preprocessing": {
    "drop_columns": ["string"],
    "drop_rows": int,
    "shuffle": bool,
    "split": {
        "train": float,
        "eval": float,
        "test": float
    }
  },
  "training": {
    "server_rounds": int,
    "local_epochs": int,
    "batch_size": int
  },
  "aggregation": {
    "strategy": "string",
    "fraction_fit": float,
    "fraction_evaluate": float,
    "min_available_clients": int
  },
  "tensorflow": {
    "model": {
      "config_file": "string",
      "config": dict
    }
  },
  "scikit_learn": {
    "model": {
      "kind": "logistic_regression",
      "n_features": int,
      "classes": ["any"]
    }
  },
  "xgboost": {
    "model": {
      "kind": "booster",
      "n_features": int
    },
    "params": {
      "objective": "binary:logistic"
    }
  }
}
```

- `name`: Name for the scenario. This is the UID for the scenario. Change the name of the scenario if modifications are made. If a training session is started with a different scenario but with the same name as a previous one, the previous scenario will be used.
- `framework`: The ML framework/runtime family. Current in-repo support includes TensorFlow via `framework: "tensorflow"` with a matching `tensorflow` block, scikit-learn logistic regression via `framework: "scikit_learn"` with a matching `scikit_learn` block, and XGBoost bagging via `framework: "xgboost"` with a matching `xgboost` block.
- `dataset` (mutually exclusive with `datasets`): Uses the parquet dataset in each client's Minio datastore with matching name (under `datasets/[datasetname].parquet`), or tries downloading the dataset from Kaggle (e.g. `dhoogla-nfunswnb15v2` matches Kaggle dataset `nfunswnb15v2` of user `dhoogla`, and `samybtt-ermds-x` matches `ermds-x` of user `samybtt`) and uploads it to the Minio datastore.
- `dataset_file`: Optional parquet member name to extract from a downloaded Kaggle zip when the archive contains multiple parquet files.
- `datasets` (mutually exclusive with `dataset`): Similar to `dataset` but with the ability to define a dataset per client (clients are identified by the node's name on which they run). Value must be a dictionary (e.g. `{"client1": "some-dataset"}`). The length of `datasets` must be greater than or equal to `aggregation.min_available_clients`.
- `hold_out_dataset`: This optional value holds the name of a dataset which is used to evaluate the aggregated model at the end of each round at server-side. This dataset must be placed in the server’s datastore or, like before, may refer to a Kaggle dataset which will be downloaded to the server’s datastore.
- `hold_out_dataset_file`: Optional parquet member name to extract for the hold-out dataset when a Kaggle zip contains multiple parquet files.
- `centralised`: If `true`, the framework retrieves the dataset(s) set in `dataset` or `datasets` from the server's Minio datastore. If `false`, the client's own Minio datastores are used.
- `label_column`: The name of the column containing the labels of your dataset. 
- `preprocessing.drop_columns`: Columns to be dropped/ignored from dataset.
- `preprocessing.drop_rows`: Drop amount of random rows ro shrink dataset.
- `preprocessing.shuffle`: Shuffle dataset.
- `preprocessing.split`: Percentage between `train`, `eval` and `test` splits.
- `training.server_rounds`: Amount of aggregating rounds between clients and server.
- `training.local_epochs`: Amount of local epochs to perform by clients between aggregations.
- `training.batch_size`: Batch size for local epochs.
- `aggregation.strategy`: Aggregation strategy. Current supported values are `fedavg`, `fedmedian`, `fedprox`, `fedadam`, `fedadagrad`, `fedyogi`, or `bulyan` for TensorFlow, `fedavg`, `fedmedian`, `fedadam`, `fedadagrad`, `fedyogi`, or `bulyan` for scikit-learn, and `fedxgbbagging` or `fedxgbcyclic` for XGBoost.
- `aggregation.fraction_fit`: Percentage of available clients that are required for fitting.
- `aggregation.fraction_evaluate`: Percentage of available clients that are required for evaluation.
- `aggregation.min_available_clients`: Minimum amount of clients that need to be available/connected for a training sessions.
- `aggregation.num_malicious_nodes`: Assumed number of Byzantine clients for `bulyan`. Runtime validation requires `min_available_clients >= 4 * num_malicious_nodes + 3`.
- `aggregation.proximal_mu`: Optional FedProx proximal strength. This is currently supported on the TensorFlow path.
- `aggregation.eta`, `aggregation.eta_l`, `aggregation.beta_1`, `aggregation.beta_2`, `aggregation.tau`: Optional FedOpt optimizer controls for `fedadam`, `fedadagrad`, and `fedyogi`. If omitted, each Flower strategy keeps its own built-in defaults.
- `tensorflow.model.config_file`: Preferred authored format. Path to a TensorFlow/Keras model config file relative to the scenario file or absolute on disk.
- `tensorflow.model.config`: Inline TensorFlow/Keras model config payload. This remains useful for tests and generated scenarios.
- `scikit_learn.model.kind`: Currently supports `logistic_regression`.
- `scikit_learn.model.n_features`: Feature-column count expected by the model. This lets the server and trainer initialize and validate the federated parameter shape.
- `scikit_learn.model.classes`: Full ordered class list for the classifier, used to initialize a consistent federated parameter shape across clients.
- `scikit_learn.model.penalty`, `scikit_learn.model.solver`, `scikit_learn.model.multi_class`: Optional scikit-learn logistic regression settings.
- `xgboost.model.kind`: Currently supports `booster`.
- `xgboost.model.n_features`: Feature-column count expected by the booster input matrix.
- `xgboost.params`: Raw XGBoost parameter dictionary. Current runtime support is intentionally narrow and expects `objective: "binary:logistic"` together with a `fedxgbbagging` or `fedxgbcyclic` aggregation strategy.

Legacy compatibility:

- top-level `hyper` is still accepted and normalized to `training`.
- top-level legacy `strategy` is still accepted and normalized to `aggregation` with `aggregation.strategy: "fedavg"`.
- top-level `model_file` is still accepted and normalized to `tensorflow.model.config_file`.
- top-level raw `model` payloads are still accepted and normalized to `tensorflow.model.config`.

These legacy fields emit deprecation warnings and are intended as a migration bridge only.

To submit an authored scenario containing `tensorflow.model.config_file`, render it first so the payload includes `tensorflow.model.config`:

```bash
python render_scenario.py examples/nids.json | \
  curl -X POST -H "Content-Type: application/json" --data @- http://[server-address]/start
```

## Usage

### Server

Interacting with the framework happens fully through the central server (`http://[server-address]:80`) with following endpoints:

- POST `/start`: Start a training run with a scenario as body. The example scenario above can be used to try it out. The response of a successfull request will include that run's `run-id` which can be used to get its status.

  ```bash
  # Here, scenario.json may use tensorflow.model.config_file in authored form
  python render_scenario.py scenario.json | \
    curl -X POST -H "Content-Type: application/json" --data @- http://[server-address]/start
  ```

- GET `/status?runId=`: Get the status of a training run. If no `runId` is given, all runs are requested. Set `runId` to request details of a specific run.

  ```bash
  # Get details about all runs
  curl http://[server-address]/status
  # Get details about specific run
  curl -G -d "runId=[run-id]" http://[server-address]/status
  ```

- GET `/clients`: Get details about currently connected clients. Namely their `id`, which is used to target datasets when using `datasets` in the scenario, and connection `address`.

  ```bash
  # Get connected client details
  curl http://[server-address]/clients
  ```

After a training run, the resulting aggregated dataset can be downloaded from the Minio instance by surfing to the web UI on `http://[server-address]:9001` with default credentials `aide-fl1:aide-fl1`. It's also possible to interact with Minio using S3 over port `:9000`.

If `centralised` is set to `true` and the to be used datasets do not originate from Kaggle, upload the datasets manually to the `datasets` bucket of the server side Minio instance.

### Client

If the to be used datasets do not originate from Kaggle, it is the client node maintainer's job to provide their client with the right datasets on which it is supposed to train. Do this by uploading the parquet dataset to the Minio datastore using its web UI on `http://[client-address]:9001` with default credentials `aide-fl1:aide-fl1`. It's also possible to upload the dataset using S3 over port `:9000`.

Be sure to let the server maintainer know what the dataset's filename is so it may be correctly specified in the scenario.

### Manual Testing

Although this project is completely focused on the training side of model development, it is nice to be able to quickly assess the new model's performance in a local inference test, and to get a view on each client's training performance with learning and validation curves.

This is possible with the two Jupyter notebooks contained in the `manual-testing` directory: One for creating learning and validation curves for each client and another to perform local inderence on the aggregated model.

### Examples

The `examples` directory contains authored scenarios and model definitions for the currently maintained demo flows. The NIDS scenarios retrieve their datasets directly from Kaggle. Additional scenarios can point at Kaggle-hosted parquet archives directly, and may set `dataset_file` when the archive contains multiple parquet members and one specific file should be used.

## Server Services

Services residing on the one server node.

### Server

This service is one of the three services the user communicates with directly and is the
entry point to the framework. The user supplies a scenario to `server-logic` over HTTP which is relayed to all client nodes so they can start preparing for the training run. Once all clients are done preparing, `server-logic` triggers the Flower training process which is coordinated by the `flower-serverapp`. `flower-superlink` retains a persistent connection with the clients’ `flower-supernodes`. The use of separate containers to manage this connection abstracts away the underlying transport mechanisms and ensures reliable bidirectional communication necessary for parameter aggregation during a training run. Upon completion of the run, `server-logic` uploads the resulting artifacts, including the aggregated model, to the server’s `datastore` service. Metrics are logged to Weights & Biases for a visual representation.

### Datastore

The second service with which a user can communicate directly. Its purpose is to store aggregated models, metrics, and model checkpoints resulting from running scenarios, and optionally also datasets provided by the user which can be used by clients to train upon for a more centralised experience rather than distributed, or to evaluate the final model at the end of a training run as a holdout dataset. The technology used is a MinIO object store which hosts a web interface for easy user access.

## Client Services

Except for `server` is each service part of the client. These services are deployed as daemonsets so that they may run on each node of the K8s instance. The nodes in your K8s instance each represent an individual Federated Learning client.

### Orchestrator

As its name suggests, the `orchestrator` service orchestrates all other client node services. Its first task is to connect to `server-logic`, waiting to receive a scenario. Upon acquiring one, a succession of messages is sent between the `orchestrator` and other client services, triggering multiple steps such as `datastore-logic` organising files in the `datastore`, preprocessing of the dataset in `preprocessor-logic`, and making `trainer-logic` prepare for the Flower training run in `trainer`. Once these steps have been
completed, `orchestrator` conveys to `server-logic` that the client is ready for the training run. The `orchestrator` is also in charge of relaying messages from other services to the server. Such messages can contain handy information or error messages. The latter will abort further client preparations together with the training run in general.

### Datastore

`datastore-logic` organises the file structure in `minio`, a MinIO object store, and keeps
track of whether the necessary dataset (in parquet format) is present for the scenario. If not, it attempts to download the dataset from Kaggle.

### Preprocessor

This service downloads the dataset for a given scenario from `minio` and performs the necessary preprocessing steps set in the scenario. The dataset may be downloaded from either the client Minio instance or the server’s instance depending on whether the scenario specifies a distributed or centralised approach. To reduce storage usage, the resulting training, validation, and test sets are not uploaded to `minio`, but kept in `redis`, a Redis database, for a certain time period or until it is overwritten by another preprocessing task. This caching enables reusability of the preprocessed datasets would the same scenario be ran again.

### Trainer

`trainer-logic` downloads the preprocessed datasets from `redis` for `flower-clientapp` to
use, and completes the last preparation step. The `flower-supernode` initialises a connection with `flower-superlink`, registering the client node as a potential FL client in the federation. The `flower-clientapp` contains the logic to perform the Flower training steps on the preprocessed datasets.

### Kafka

Communication between `orchestrator` and other services on that same client node moves through a Kafka broker dedicated to that node. A total of seven topics are used to relay information:

- 3 times 2 topics for two-way communication between `orchestrator` and the 3 other services.
- All services may publish messages with info or errors to the 7th topic. These are consumed by `orchestrator` and sent to the server.

## Known Bugs

Currently, there are two known bugs that might impede smooth operation of the framework.

### Initial Deployment

After completion of the deployment script, everything might seem fine at first glance. But, when trying to start a training run using `/start`, the HTTP request will load indefinitely waiting for a response.

Inspecting the server logs reveals that one or multiple clients don't submit a `DONE` message, halting the training process. Inspecting the `orchestrator` and `datastore` services of those clients reveals that either the message from `orchestrator` to `datastore`, telling `datastore` to prepare the dataset, is not received by `datastore`, or the message from `datastore` back to `orchestrator`, telling `orchestrator` that the dataset has been prepared, is not received by `orchestrator`.

I can confirm that those messages are in every case received by the dedicated Kafka instances which makes me think that either `orchestrator` or `datastore` couldn't always make a reliable connection to consume those messages.

The solution is to restart both `orchestrator`, `datastore` and `server` which can be done by simply deleting the services (K8s will start them back up). Doing this results in `orchestrator` and `datastore` reconnecting with `kafka` fixing the issue. Restarting `server` is necessary to cancel the training loop it is stuck in. After that restart, it might take a few seconds to let the `trainer` and `orchestrator` services connect back to the `server`.

### Preprocessed Datasets

When running a scenario where datasets need to be preprocessed (not using cached datasets), an error will be thrown saying that `trainer` couldn't find the preprocessed datasets within `redis`.

This is thrown because `preprocessor-logic` messages `orchestrator` that it is done preprocessing before the datasets are fully uploaded to `redis` which results in `trainer-logic` attempting to get those datasets before they are fully present.

The solution is to simply rerun the scenario at which time the datasets should've gotten ample time to be uploaded.
