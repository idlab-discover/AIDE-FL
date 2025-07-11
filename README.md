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

Scenarios are the driving factor behind this framework. They include the required dataset name, preprocessing steps and model architecture. It is sent from the central server to each client from where they each prepare the dataset to then start training.

Following example scenario works out the box:

```json
{
  "name": "test",
  "dataset": "dhoogla-nfunswnb15v2",
  "centralised": false,
  "label_column": "Label",
  "preprocessing": {
    "drop_columns": ["RETRANSMITTED_IN_BYTES","RETRANSMITTED_IN_PKTS","RETRANSMITTED_OUT_BYTES","RETRANSMITTED_OUT_PKTS","SRC_TO_DST_AVG_THROUGHPUT","DST_TO_SRC_AVG_THROUGHPUT","NUM_PKTS_UP_TO_128_BYTES","NUM_PKTS_128_TO_256_BYTES","NUM_PKTS_256_TO_512_BYTES","NUM_PKTS_512_TO_1024_BYTES","NUM_PKTS_1024_TO_1514_BYTES","TCP_WIN_MAX_IN","TCP_WIN_MAX_OUT","ICMP_TYPE","ICMP_IPV4_TYPE","DNS_QUERY_ID","DNS_QUERY_TYPE","DNS_TTL_ANSWER","FTP_COMMAND_RET_CODE","Attack"],
    "drop_rows": 1500000,
    "shuffle": true,
    "split": {
        "train": 0.7,
        "eval": 0.15,
        "test": 0.15
    }
  },
  "hyper": {
    "server_rounds": 1,
    "local_epochs": 1,
    "batch_size": 256
  },
  "strategy": {
    "fraction_fit": 1.0,
    "fraction_evaluate": 1.0,
    "min_available_clients": 2
  },
  "model": {"module": "keras.src.models.functional", "class_name": "Functional", "config": {"name": "functional", "trainable": true, "layers": [{"module": "keras.layers", "class_name": "InputLayer", "config": {"batch_shape": [null, 22, 1], "dtype": "float32", "sparse": false, "ragged": false, "name": "input_layer"}, "registered_name": null, "name": "input_layer", "inbound_nodes": []}, {"module": "keras.layers", "class_name": "Conv1D", "config": {"name": "conv1d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 8, "kernel_size": [3], "strides": [1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1], "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": {"module": "keras.regularizers", "class_name": "L2", "config": {"l2": 0.001}, "registered_name": null}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 22, 1]}, "name": "conv1d", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 22, 1], "dtype": "float32", "keras_history": ["input_layer", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "MaxPooling1D", "config": {"name": "max_pooling1d", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "pool_size": [2], "padding": "valid", "strides": [2], "data_format": "channels_last"}, "registered_name": null, "name": "max_pooling1d", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 22, 8], "dtype": "float32", "keras_history": ["conv1d", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "LayerNormalization", "config": {"name": "layer_normalization", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": [-1], "epsilon": 0.001, "center": true, "scale": true, "rms_scaling": false, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 11, 8]}, "name": "layer_normalization", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 11, 8], "dtype": "float32", "keras_history": ["max_pooling1d", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Conv1D", "config": {"name": "conv1d_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 24, "kernel_size": [2], "strides": [1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1], "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": {"module": "keras.regularizers", "class_name": "L2", "config": {"l2": 0.001}, "registered_name": null}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 11, 8]}, "name": "conv1d_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 11, 8], "dtype": "float32", "keras_history": ["layer_normalization", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Conv1D", "config": {"name": "conv1d_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "filters": 192, "kernel_size": [2], "strides": [1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1], "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": {"module": "keras.regularizers", "class_name": "L2", "config": {"l2": 0.001}, "registered_name": null}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 11, 24]}, "name": "conv1d_2", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 11, 24], "dtype": "float32", "keras_history": ["conv1d_1", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "LayerNormalization", "config": {"name": "layer_normalization_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "axis": [-1], "epsilon": 0.001, "center": true, "scale": true, "rms_scaling": false, "beta_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "gamma_initializer": {"module": "keras.initializers", "class_name": "Ones", "config": {}, "registered_name": null}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 11, 192]}, "name": "layer_normalization_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 11, 192], "dtype": "float32", "keras_history": ["conv1d_2", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "data_format": "channels_last"}, "registered_name": null, "build_config": {"input_shape": [null, 11, 192]}, "name": "flatten", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 11, 192], "dtype": "float32", "keras_history": ["layer_normalization_1", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 160, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": {"module": "keras.regularizers", "class_name": "L2", "config": {"l2": 0.001}, "registered_name": null}, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 2112]}, "name": "dense", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 2112], "dtype": "float32", "keras_history": ["flatten", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "rate": 0.5, "seed": null, "noise_shape": null}, "registered_name": null, "name": "dropout", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 160], "dtype": "float32", "keras_history": ["dense", 0, 0]}}], "kwargs": {"training": false}}]}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 80, "activation": "relu", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": {"module": "keras.regularizers", "class_name": "L2", "config": {"l2": 0.001}, "registered_name": null}, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 160]}, "name": "dense_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 160], "dtype": "float32", "keras_history": ["dropout", 0, 0]}}], "kwargs": {}}]}, {"module": "keras.layers", "class_name": "Dropout", "config": {"name": "dropout_1", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "rate": 0.5, "seed": null, "noise_shape": null}, "registered_name": null, "name": "dropout_1", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 80], "dtype": "float32", "keras_history": ["dense_1", 0, 0]}}], "kwargs": {"training": false}}]}, {"module": "keras.layers", "class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": {"module": "keras", "class_name": "DTypePolicy", "config": {"name": "float32"}, "registered_name": null}, "units": 1, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"module": "keras.initializers", "class_name": "GlorotUniform", "config": {"seed": null}, "registered_name": null}, "bias_initializer": {"module": "keras.initializers", "class_name": "Zeros", "config": {}, "registered_name": null}, "kernel_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "registered_name": null, "build_config": {"input_shape": [null, 80]}, "name": "dense_2", "inbound_nodes": [{"args": [{"class_name": "__keras_tensor__", "config": {"shape": [null, 80], "dtype": "float32", "keras_history": ["dropout_1", 0, 0]}}], "kwargs": {}}]}], "input_layers": [["input_layer", 0, 0]], "output_layers": [["dense_2", 0, 0]]}, "registered_name": "Functional", "build_config": {"input_shape": null}, "compile_config": {"optimizer": {"module": "keras.optimizers", "class_name": "Adam", "config": {"name": "adam", "learning_rate": 9.999999747378752e-05, "weight_decay": null, "clipnorm": null, "global_clipnorm": null, "clipvalue": null, "use_ema": false, "ema_momentum": 0.99, "ema_overwrite_frequency": null, "loss_scale_factor": null, "gradient_accumulation_steps": null, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1e-07, "amsgrad": false}, "registered_name": null}, "loss": "binary_crossentropy", "loss_weights": null, "metrics": ["accuracy", {"module": "keras.metrics", "class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "registered_name": null}, {"module": "keras.metrics", "class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}, "registered_name": null}, {"module": "keras.metrics", "class_name": "AUC", "config": {"name": "auc_roc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "multi_label": false, "num_labels": null, "label_weights": null, "from_logits": false}, "registered_name": null}], "weighted_metrics": null, "run_eagerly": false, "steps_per_execution": 1, "jit_compile": false}}
}
```

- `name`: Name for the scenario. This is the UID for the scenario. Change the name of the scenario if modifications are made. If a training session is started with a different scenario but with the same name as a previous one, the previous scenario will be used.
- `dataset` (mutually exclusive with `datasets`): Uses the parquet dataset in each client's Minio datastore with matching name (under `datasets/[datasetname].parquet`), or tries downloading the dataset from Kaggle (e.g. `dhoogla-nfunswnb15v2` matches with dataset `nfunswnb15v2` (case insensitive) of user `dhoogla`) and uploads it to the Minio datastore. NOTE: The downloaded zip from Kaggle must contain a dataset as parquet file and the first one is taken if there would be multiple parquet files.
- `datasets` (mutually exclusive with `dataset`): Similar to `dataset` but with the ability to define a dataset per client (clients are identified by the node's name on which they run). Value must be a dictionary (e.g. `{"client1": "some-dataset"}`). The length of `datasets` must be greater than or equal to `strategy.min_available_clients`.
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