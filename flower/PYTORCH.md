# PyTorch adapter

The `pytorch` framework uses the same scenario, parquet, ClientApp, FedAvg,
server holdout and checkpoint paths as the other adapters. Supported models are
binary tabular `logistic_regression` and `mlp` with ReLU hidden layers. Labels must
be numeric 0 and 1. CPU float32 execution is supported. GPU execution, arbitrary
Python model classes, CNNs, sequence models, multiclass tasks and regression are
outside this implementation.

Use `examples/smoke-pytorch.json` as the scenario template. `pytorch.model` defines
`kind`, `n_features`, `hidden_sizes` for an MLP, and an optional `seed` defaulting to
42. Optional `feature_mean` and `feature_scale` arrays apply a fixed transform to
raw input. Supply both, in parquet feature-column order, with positive scales.
Fit them on training data only and use identical settings and feature order at
every participant and the server. The adapter never fits normalization during
evaluation. `pytorch.optimizer` accepts `adam` or `sgd`, `learning_rate` and
`weight_decay`. Defaults are Adam, 0.001 and 0.0. Training uses binary cross
entropy with logits, scenario `local_epochs` and `batch_size`, and deterministic
minibatch ordering based on the model seed and Flower round. Optimizer state is
reset each round. Only `fedavg` is supported.

## Parameters and non-trainable state

Flower 1.43 `ArrayRecord` carries float32 trainable parameters under their actual
PyTorch names, in sorted name order. Clients and the production strategy check
names, order, shape, dtype and finiteness before loading or averaging updates.
Sample counts weight FedAvg. Parameters with `requires_grad=False` and all
registered buffers are rejected, whether integer or floating point. Thus
BatchNorm counters and running statistics cannot silently enter an average.
No FedBN or synchronization policy for mutable buffers is implemented. Adding
such layers requires an explicit state policy and separate verification.

## Persistence and inference

Each round writes `<name>_round_<n>.safetensors` and `.model.json`. The final round
also writes `<name>.safetensors` and `.model.json`. Metadata includes architecture,
seed and fixed preprocessing. Safetensors stores named tensors without pickled
Python code. Reload checks the architecture's exact parameter layout.

```python
from flower.frameworks import get_adapter
adapter = get_adapter("pytorch")
model = adapter.load_model("share/smoke-pytorch")
probabilities = adapter.predict_proba(model, raw_features)
predictions = probabilities[:, 1] >= 0.5
```

`raw_features` must have the training feature order. `predict_proba` applies the
saved transform once. Training and evaluation adapter methods receive features
already transformed by the shared data loader.

## Reproduce the sensor experiment

Use Python 3.12 and install `experiments/wp4/requirements.txt`. Direct dependencies
pin Flower 1.43.0 and the official CPU build `torch==2.14.1+cpu`. The application
metadata pins the equivalent public version `torch==2.14.1`. Both base container
requirements install the CPU build. No torchvision or torchaudio is required.

```sh
python experiments/wp4/pytorch_smoke.py --output /tmp/pytorch-smoke --clients 2 --rounds 5 --seed 42
python scripts/check-pytorch-transport.py --evidence /tmp/pytorch-smoke --output /tmp/pytorch-transport
python experiments/wp4/publish_pytorch.py --evidence /tmp/pytorch-smoke --transport /tmp/pytorch-transport --key-file /secure/path/wandb.key --receipt /tmp/pytorch-receipt.json
```

The runner downloads or reuses the existing Kaggle smoke sensor dataset. It drops
duplicate sensor vectors, splits chronologically with 60-second embargoes, and
reserves the latter test period for the server with another embargo. Training
rows alone determine normalization. Simulated participants get disjoint training,
validation and test rows. Saved partition CSVs retain row IDs, timestamps, labels
and ownership. Prepared parquet files exercise the existing data-loader contract;
the runner supplies explicit splits instead of asking the general preprocessor
to reconstruct temporal embargoes. The template's split fractions are examples
for the generic service, not a substitute for this experiment preparation.

The fixed final round is evaluated without checkpoint selection. The runner
retains versions, seeds, configuration, data provenance, source snapshots,
per-round client/validation/server metrics, predictions, a majority baseline,
model files and hashes. It independently checks sample-weighted aggregation and
exact reloaded predictions. Publication reads the credential only into the
process environment and verifies downloaded W&B artifact hashes and remote
metrics. Never add a key to evidence directories.

The isolated TLS check uses one SuperNode and both production apps for a training
and evaluation round. It compares the saved aggregate against an independently
computed local update, checks server holdout evaluation and worker authentication,
and stops its processes. It does not deploy to Kubernetes. The separate local
experiment verifies multiple participants but is not network-distributed training.

## Privacy scope

Central DP remains explicitly restricted to TensorFlow in the existing privacy
validator. PyTorch DP requests are rejected by both scenario validation and the
runtime. No PyTorch DP accounting or noise experiment has been performed.
The existing SecAgg+ helper uses a legacy protocol comparison path; the CKKS
helper encrypts array sums in a local experiment. Neither is integrated or
verified with this adapter. This integration makes no PyTorch SecAgg+, MPC or
FHE support claim. Shared preprocessing moments and ordinary per-client metrics
are not private. TLS transport protection is separate from these mechanisms.

Version and API decisions were checked on 11 November 2026 against official
[Flower 1.43 ArrayRecord documentation](https://flower.ai/docs/framework/ref-api/flwr.app.ArrayRecord.html),
[Flower's PyTorch quickstart](https://flower.ai/docs/framework/tutorial-quickstart-pytorch.html)
and [PyTorch's published CPU installation commands](https://pytorch.org/get-started/previous-versions/).
Go scenario changes follow [Effective Go](https://go.dev/doc/effective_go).
