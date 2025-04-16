# AIDE-FL

Federated Learning on K8s in a modular and reproducable fashion.

## Installation

Below are the commands to build each service, push them to a package repo of choice and apply them as a daemonset on you K8s instance. Be sure to change change the image location in the daemonset file.

```
git clone https://github.com/MartienskieNum1/AIDE-FL.git

docker build --tag [your package repo] ./AIDE-FL/minio-datastore/proxy/
docker push [your package repo]:latest
kubectl apply -f ./AIDE-FL/minio-datastore/proxy/minio-daemonset.yaml

docker build --tag [your package repo] ./AIDE-FL/orchestrator/
docker push [your package repo]:latest
kubectl apply -f ./AIDE-FL/orchestrator/minio-daemonset.yaml
```

## Scenarios

Scenarios are the driving factor behind this framework. They include the required dataset name, preprocessing steps and model architecture. It is sent from the central server to each client from where they each start the training process.

## Client Services

Except for `server` is each service part of the client. These services are deployed as daemonsets so that they may run on each node of the K8s instance. The nodes in your K8s instance each represent an individual Federated Learning client.

### Orchestrator

This service accepts commands from the central server and deligates all steps to be taken to the corresponding service.

### Minio Datastore

This service exists out of two containers: A default Minio image and a `minio-proxy`. The `minio-proxy` accepts scenarios from the `orchestrator`. It makes sure that the required dataset is downloaded from Kaggle (if not already present) and the scenario is saved along with the dataset.

After the dataset has been preprocessed by `preprocessor`, an index file containing train-, test- and validationset indices is sent to `minio-proxy` to be saved.

Upon receiving a scenario, `minio-proxy` responds with one of four statuses:

- `new dataset`: The dataset has been downloaded and the scenario saved.
- `new scenario`: The dataset is already present but the scenario is new and saved.
- `not preprocessed`: Dataset and scenario are both already present but no preprocessing has been executed.
- `preprocessed`: Dataset, scenario and index file are all present.