# AIDE-FL

Federated Learning on K8s in a modular and reproducable fashion. Every node works independently from another. All pods use the node localhost network to communicate.

## Installation

The `deploy.sh` script in the root of the repo should guide you through the necessary steps to get this framework up and running on a Kubernetes (k8s) cluster.

The only requirement for the k8s cluster is to have at least two worker nodes. Three is recommended.

```sh
git clone https://github.com/MartienskieNum1/AIDE-FL.git

cd AIDE-FL

./deploy.sh
```

The scripts guides you through the following steps:

- `kubectl`, `jq` and `openssl` should be installed.
- A kubeconfig context must be set which points to the correct k8s instance.
- There should be exactly one k8s node with label `fl-role=server` and at least one node with label `fl-role=client`.
- The `certs` folder is created containing certificates for secure Flower TLS communication.
- A Weights & Biases API key is needed to log metrics.
- For an empty cluster, the script creates the `aide-fl` namespace and deploys all pods.

The script can be reused for a full redeployment to apply updated containers.

## Scenarios

Scenarios are the driving factor behind this framework. They include the required dataset name, preprocessing steps, hyperparameters, and model architecture. It is sent from the central server to each client which prepare themselves for training. Once they are all done preparing, the server will start the training process.

A scenario follows a certain JSON template:

```json
{
  "name": "string",
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
  "hyper": {
    "server_rounds": int,
    "local_epochs": int,
    "batch_size": int
  },
  "strategy": {
    "fraction_fit": float,
    "fraction_evaluate": float,
    "min_available_clients": int
  },
  "model": dict
}
```

- `name`: Name for the scenario. This is the UID for the scenario. Change the name of the scenario if modifications are made. If a training session is started with a different scenario but with the same name as a previous one, the previous scenario will be used.
- `dataset` (mutually exclusive with `datasets`): Uses the parquet dataset in each client's Minio datastore with matching name (under `datasets/[datasetname].parquet`), or tries downloading the dataset from Kaggle (e.g. `dhoogla-nfunswnb15v2` matches with dataset `nfunswnb15v2` (case insensitive) of user `dhoogla`) and uploads it to the Minio datastore. NOTE: The downloaded zip from Kaggle must contain a dataset as parquet file and the first one is taken if there would be multiple parquet files.
- `datasets` (mutually exclusive with `dataset`): Similar to `dataset` but with the ability to define a dataset per client (clients are identified by the node's name on which they run). Value must be a dictionary (e.g. `{"client1": "some-dataset"}`). The length of `datasets` must be greater than or equal to `strategy.min_available_clients`.
- `hold_out_dataset`: This optional value holds the name of a dataset which is used to evaluate the aggregated model at the end of each round at server-side. This dataset must be placed in the server’s datastore or, like before, may refer to a Kaggle dataset which will be downloaded to the server’s datastore.
- `centralised`: If `true`, the framework retrieves the dataset(s) set in `dataset` or `datasets` from the server's Minio datastore. If `false`, the client's own Minio datastores are used.
- `label_column`: The name of the column containing the labels of your dataset. 
- `preprocessing.drop_columns`: Columns to be dropped/ignored from dataset.
- `preprocessing.drop_rows`: Drop amount of random rows ro shrink dataset.
- `preprocessing.shuffle`: Shuffle dataset.
- `preprocessing.split`: Percentage between `train`, `eval` and `test` splits.
- `hyper.server_rounds`: Amount of aggregating rounds between clients and server.
- `hyper.local_epochs`: Amount of local epochs to perform by clients between aggregations.
- `hyper.batch_size`: Batch size for local epochs.
- `strategy.fraction_fit`: Percentage of available clients that are required for fitting.
- `strategy.fraction_evaluate`: Percentage of available clients that are required for evaluation.
- `strategy.min_available_clients`: Minimum amount of clients that need to be available/connected for a training sessions.
- `model`: Copy paste this from a Keras Model `model.to_json()`. The example in the scenario above is the result from running the `model.py` file in the repo root.

## Usage

### Server

Interacting with the framework happens fully through the central server (`http://[server-address]:80`) with following endpoints:

- POST `/start`: Start a training run with a scenario as body. The example scenario above can be used to try it out. The response of a successfull request will include that run's `run-id` which can be used to get its status.

  ```bash
  # Here, a scenario is placed in scenario.json
  curl -X POST -H "Content-Type: application/json" -d @scenario.json http://[server-address]/start
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

The `examples` directory contains Python scripts, used to generate datasets, and corresponding scenarios. Each script generates three parquet files: Parts 1 and 2, each containing 3/7 of the original set, are to be uploaded to two separate clients, number 3, containing 1/7 of the original set, needs to be uploaded to the server’s datastore to serve as hold-out dataset. The scenarios need to be completed with the correct client IDs under the datasets key. The directory also contains a scenario for use of a NIDS dataset, which retrieves its datasets directly from Kaggle.

## Client Services

Except for `server` is each service part of the client. These services are deployed as daemonsets so that they may run on each node of the K8s instance. The nodes in your K8s instance each represent an individual Federated Learning client.

### Orchestrator

This service accepts commands from the central server and deligates all steps to be taken to the corresponding service.

### Minio Datastore

This service exists out of two containers: A default Minio image and a `datastore-logic`. The `datastore-logic` accepts scenarios from the `orchestrator`. It makes sure that the required dataset is downloaded from Kaggle (if not already present) and the scenario is saved along with the dataset.

After the dataset has been preprocessed by `preprocessor`, an index file containing train-, test- and validationset indices is sent to `datastore-logic` to be saved.

Upon receiving a scenario, `datastore-logic` responds with one of four statuses:

- `new dataset`: The dataset has been downloaded and the scenario saved.
- `new scenario`: The dataset is already present but the scenario is new and saved.
- `not preprocessed`: Dataset and scenario are both already present but no preprocessing has been executed.
- `preprocessed`: Dataset, scenario and index file are all present.