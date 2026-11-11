# WP4 smoke-sensor experiments

`smoke.py` downloads the real smoke-detection sensor dataset, trains local and centralised baselines and a federated model, validates every round, and evaluates the fixed final model once on a later hold-out period. It uses the repository's scikit-learn adapter and Flower's `Message` records and `FedAvg.aggregate_train`. Participants execute in one Python process. This establishes model training, aggregation, validation and saved-model inference, not Kubernetes transport, cross-site operation or deployment readiness.

## Run

Use the repository's compatible Flower and ML requirements in a Python 3.12 environment. The runner needs Flower, scikit-learn, pandas, NumPy and joblib. Results must use a new output directory.

Install `experiments/wp4/requirements.txt` for the experiments and plots. Add
`flower/fhe-requirements.txt` for the optional CKKS experiment.

```sh
python experiments/wp4/smoke.py \
  --output final-report/working/wp4-validation/sensor-iid \
  --seeds 17 42 73 --clients 2 --rounds 20 --local-epochs 5

python experiments/wp4/smoke.py \
  --output final-report/working/wp4-validation/sensor-chronological \
  --partition chronological --seeds 17 42 73 --clients 2 \
  --rounds 20 --local-epochs 5
```

To study participant count with fixed total training data, repeat the IID command with `--clients 2`, `--clients 4` and `--clients 8`, each with its own output directory. This measures the model's sensitivity to splitting one dataset among more simulated participants. It does not measure system scaling or additional sites. Class labels come from the scenario configuration and are never averaged as model parameters; this avoids rounding errors when participants have unequal numbers of rows.

Each seed trains every client-local baseline, one centralised baseline and FedAvg. Each participating sample receives the same maximum solver iterations, `rounds * local_epochs`. SAGA may not converge within each deliberately bounded local update; the run summary counts convergence warnings. Models use the same logistic-regression configuration and default regularisation. Local baselines see fewer rows, and centralised training has more solver updates per round because it sees all training rows. These are training comparisons, not wall-time benchmarks.

The per-round `train_*` values are sample-weighted client metrics measured before aggregation. The `global_train_*` values evaluate the resulting model on the pooled training rows after aggregation. These answer different questions, particularly when clients have different data distributions. For local-only baselines, `global_train_*` also includes other clients' training rows for diagnosis; those rows do not update the local model. Summaries retain zero-initialisation validation scores, final-round validation scores and coefficient norms so that an executed training run can be distinguished from an untrained baseline. The untouched test period is still evaluated only after the fixed final round.

Use `publish.py` to retain finished results in W&B and generate figures. The runner itself does not read credentials or upload data.

```sh
python experiments/wp4/publish.py \
  final-report/working/wp4-validation/sensor-iid \
  --online --key-file wandb.key --group smoke-sensor-iid
```

Without `--online`, this command only generates PNG and PDF figures. Online publication uses the `aide-wp4-smoke` project by default. It uploads each run's metrics, model, predictions, partitions, source snapshots and figures, then reads the remote run state and history to verify completion. `wandb-receipts.json` records the verified links. The key is read into the process environment and is not included in any artifact.

Publication first checks saved-file hashes and complete, ordered training rounds.
Remote verification compares every retained history measurement and summary value,
allowing only floating-point serialization rounding. New runs also produce
`figures/training-validation.pdf` and PNG, comparing the resulting model's training
and validation balanced accuracy. The [11 November results](RESULTS-20261111.md)
report the participant-count and privacy comparisons with their limitations.

## Dataset and split

The source is [Kaggle's Smoke Detection Dataset](https://www.kaggle.com/datasets/deepcontractor/smoke-detection-dataset). Its originator, Stefan Blattmann, describes the sensors, 1 Hz readings and fire-alarm labels in [Real-time Smoke Detection](https://hackaday.io/project/186779-real-time-smoke-detection-ai-based-sensor-fusion/details). The downloaded archive and CSV hashes identify the exact input bytes; a cached copy is reused. This is a new experiment, not evidence that the same dataset was used in any earlier project run.

The runner retains twelve sensor columns and excludes UTC, CNT and the exported row index from model inputs. It orders readings by UTC and drops exact duplicate sensor vectors before splitting. Row identifiers refer to original CSV row positions. Global chronological cutoffs allocate approximately 50% to training, 20% to validation and 30% to test, with a 60-second embargo before validation and test. Timestamp groups stay together. The 50/20/30 allocation was chosen after checking class coverage; the conventional 60/20/20 split had an all-positive validation period. Cutoffs remain fixed across seeds and methods. This is an exploratory temporal benchmark, not a preregistered study.

Default client ownership uses seeded stratified partitioning of training rows. `--partition chronological` instead divides training rows into consecutive client blocks. Both represent simulated ownership within one source dataset. This dataset does not establish distinct real organisations or sites. Train, validation and test must each contain both labels; every participant must also contain both labels.

One common `StandardScaler` uses training rows only. Shared training moments are pooled in this local experiment and reveal aggregate information; preprocessing is not privacy-preserving. All models use the same transform. No validation or test data fits the scaler. Validation metrics are recorded but do not select checkpoints or tune hyperparameters. The final fixed round is evaluated on the test period. Repeated seeds measure partition and solver randomness on the same hold-out period, not independent dataset repetitions. The model classifies individual readings and makes no temporal-sequence claim.

## Evidence

The output contains dataset provenance, library versions, source revision, dirty-tree flag, tracked diff, exact runner and adapter sources and hashes. Partition CSVs retain original row IDs, timestamps, labels and ownership, with separate class-count tables. Each method directory contains configuration, per-round JSONL metrics, final test metrics, a confusion matrix, majority-class baseline scores, per-row test predictions, and a `model.joblib` bundle containing the model, scaler and feature names. The runner reloads every saved model and checks that predictions match before recording success. Each directory also has a checksum manifest.

A high F1 score can coexist with poor class balance or poor ranking. Report balanced accuracy, ROC AUC, log loss, confusion matrices and the majority baseline alongside F1. Do not use the figures to claim an advantage from federation unless the retained comparisons support it. These runs do not require InfluxDB because the fixed timestamped CSV supplies the necessary time ordering.

## Experimental encrypted aggregation

Install the separately pinned optional `tenseal==0.3.16` and add `--fhe` to include a CKKS FedAvg comparison. This encrypts floating-point coefficient and intercept updates, adds their sample-weighted ciphertexts with a public-only context, and decrypts the aggregate using a trusted key holder. Integer class metadata stays public. Every round checks the decrypted aggregate against Flower's plaintext FedAvg within absolute tolerance `1e-5`, then trains the next round from the decrypted result. Metrics retain the numerical discrepancy and encrypted upload size.

This is a correctness experiment in one process. It deliberately retains a plaintext reference and ordinary training metrics. It provides no deployment privacy guarantee, threshold decryption, malicious-client protection or private preprocessing. CKKS supports the additions needed here; this experiment does not demonstrate arbitrary-depth or bootstrapped FHE. Secret keys and ciphertexts are not persisted in the run artifacts.

Add `--secagg --clients 3` for a separate native Flower SecAgg+ protocol comparison. It executes four protocol stages per round with isolated participant state and checks that replies expose no plaintext model arrays. The runner verifies the quantized result against ordinary Flower FedAvg within `1e-4` before using it for the next round. It records protocol-stage counts, reply counts and the discrepancy. This is in-process protocol validation. The comparison process still holds the plaintext reference and metrics, and the split/scaler retain their stated limitations. This is secure aggregation, not a general-purpose MPC implementation.

## PyTorch integration

The [PyTorch adapter guide](../../flower/PYTORCH.md) describes a separate runnable
sensor experiment and an isolated production-app TLS check. Its
[11 November verification report](RESULTS-PYTORCH-20261111.md) includes W&B
artifacts, test/build results, model-quality limitations and the exact privacy
scope. These results were produced after release `v2026.11.11` and do not modify
that release.
