# PyTorch integration verification, 11 November 2026

PyTorch binary tabular classifiers are implemented on the existing AIDE-FL
scenario and framework-adapter paths. This work was completed after release
`v2026.11.11`, on branch `feat/pytorch-adapter`. It is for review and is not part
of that release tag. No Kubernetes workloads were redeployed or reset.

The verified implementation revision is
`ad7e98b9c7a779694b283062b39e95c822361ae4`. The experiment recorded a clean source
tree and execution time `2026-11-11T16:47:25.507723+00:00`. Later commits retain
this report and verification evidence. Do not describe this adapter as delivered
or tagged by 31 October.

## Verified behavior

- Registry lookup, scenario rendering/schema, Go server validation, Go
  orchestrator/datastore JSON forwarding, and trainer feature-count validation.
- Binary logistic regression and ReLU MLP training, Adam/SGD configuration,
  named float32 Flower 1.43 parameters and sample-weighted FedAvg.
- Rejection of all state buffers, frozen parameters, invalid tensor layouts and
  non-finite updates. Integer counters are never averaged or cast to weights.
- Local validation, client test evaluation, server holdout evaluation, per-round
  and final checkpoints, and exact prediction equality after model reload.
- Two simulated clients, five rounds, two local epochs per round, batch size
  128, seed 42, MLP widths 12/32/16/1, Adam learning rate 0.001.
- One isolated TLS Flower round using the production ClientApp and ServerApp,
  parquet loaders, saving strategy and adapter. Its saved weights matched an
  independently computed one-client update. Run `12796554799196599977` completed;
  unauthenticated worker access returned HTTP 401.

The TLS round and the local multi-client experiment are separate tests. This is
not evidence of multi-node Kubernetes training, independent organisations, GPU
execution or production deployment.

## Sensor results and limits

The existing Kaggle smoke sensor dataset was deduplicated by sensor features,
then split chronologically. Sixty-second embargoes separate training,
validation, client test and server holdout periods. Training has 31,254 rows;
client test has 9,334 rows and the server holdout has 9,395 rows. Participants
have disjoint rows. Normalization uses training rows only. The final fixed round
was used without checkpoint selection or hyperparameter tuning.

| Server holdout metric | Final model | Majority baseline |
| --- | ---: | ---: |
| Accuracy | 0.689303 | 0.385524 |
| Balanced accuracy | 0.689423 | 0.500000 |
| F1 | 0.631300 | 0.556503 |
| ROC AUC | 0.712242 | 0.500000 |
| Probability log loss | 2.398514 | 22.147952 |

The confusion matrix is `[[3977, 1796], [1123, 2499]]`, with true class on rows.
Server binary cross-entropy computed directly from logits rose from 0.681598 to
4.987805; final client training loss was 0.002094 and validation loss was
4.706273. The probability log loss above clips saturated probabilities and is
therefore different from the stable loss computed from logits. These results
show poor generalization/calibration on the later period despite perfect client
test accuracy. They verify an operational integration, not a tuned detector or
a federation advantage. A single seed is not a statistical performance study.

## Tests and builds

- Root Python suite: 74 passed, including 20 PyTorch tests; four additional
  existing subtests passed. Two upstream Click deprecation warnings, no skips.
- Trainer Python suite: 13 passed. Preprocessor Python suite: 8 passed.
- `go test ./...` passed in the server, orchestrator and datastore modules.
- Flower application bundle built successfully on 1.43.0.
- The local environment passed dependency compatibility checks, with 136
  installed packages. Exact versions are retained in the manifest.
- All eight affected images built locally: Flower client/server base images,
  Flower ClientApp/ServerApp images, server, orchestrator, datastore and trainer.
  Image IDs are retained in `pytorch-evidence-20261111/images.json`.
- Both Flower application images passed an in-container PyTorch training,
  evaluation, save/reload and inference-equality check with networking disabled.
  Both also passed `pip check`.

Container builds and adapter checks do not verify the running Kubernetes stack.
Images were not pushed or deployed.

## Privacy assessment

PyTorch central DP is rejected by the Go scenario validator and existing Python
privacy validator. No PyTorch clipping/noise/accounting experiment was run.
SecAgg+ and the experimental CKKS encrypted-sum path were not exercised with
PyTorch or connected to this adapter. There is no PyTorch DP, SecAgg+, MPC or FHE
support claim. TLS worker authentication was tested separately. Shared
normalization moments and ordinary client metrics are not private.

## Retained evidence

[Completed W&B run](https://forge.coreweave.com/wandb/strgen/aide-wp4-smoke/runs/69cfd552815d88a8)
in the existing [aide-wp4-smoke project](https://forge.coreweave.com/wandb/strgen/aide-wp4-smoke).
Publication verified all five rounds of remote metrics, the summary, completion
status, and all file hashes after downloading both committed artifacts:

- `strgen/aide-wp4-smoke/pytorch-training-69cfd552815d88a8:v0`
- `strgen/aide-wp4-smoke/pytorch-transport-69cfd552815d88a8:v0`

The artifacts contain models, configuration, versions, seeds, partition row IDs,
predictions, metrics, source snapshots and hashes. Credentials were read from
`wandb.key` into the publication process environment, never printed or committed.

Compact review evidence is committed under
[`pytorch-evidence-20261111`](pytorch-evidence-20261111). Its training checksum
manifest refers to the full training artifact, including files intentionally
kept outside Git. The committed transport log has terminal color codes and
trailing whitespace removed; the full artifact retains its original bytes.
Full local evidence, including build logs and prepared data,
is retained at `/home/strgenix/postdoc/projects/AIDE-FL-pytorch-evidence-20261111`.
See [the adapter guide](../../flower/PYTORCH.md) for reproduction commands and
supported configuration.

The worktree is `/home/strgenix/postdoc/projects/AIDE-FL-pytorch`. The original
checkout's unrelated changes were preserved. The release tag object remains
`08a9b5006342d7e4f0d869d02fbdb24febad05d5`, pointing to commit `c928440`.
There are no outstanding implementation or verification blockers within the
supported scope. Review and any later cluster rollout remain separate work.
