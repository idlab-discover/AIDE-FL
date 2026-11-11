# Dependency versions and upgrade validation

The 11 November 2026 revision uses Python 3.12, Flower 1.43.0, TensorFlow 2.20.0,
scikit-learn 1.9.1 and XGBoost 3.4.1. Container requirements pin direct dependencies;
`pyproject.toml` permits only the Flower 1.43 series because FAB format 1 requires
an inclusive lower bound. Its target runtime is 1.43.0.

Flower 1.43.0 was the latest stable release checked on
[PyPI](https://pypi.org/project/flwr/) on 11 November 2026. TensorFlow 2.20.0
retains the pin in Flower's current
[TensorFlow quickstart](https://github.com/flwrlabs/flower/blob/main/examples/quickstart-tensorflow/pyproject.toml),
although TensorFlow 2.21.0 is available. Flower's
[XGBoost quickstart](https://github.com/flwrlabs/flower/blob/main/examples/quickstart-xgboost/pyproject.toml)
accepts XGBoost 2.0 or later. The existing scikit-learn 1.9.1 and XGBoost 3.4.1
pins remain current stable releases. These framework pins did not need a further
change from the 10 November upgrade.

## Transport migration and rollout

Flower 1.43 merges the Fleet, Control and Runtime APIs into one HTTPS listener.
The server now listens on port 9092. `flower/config.toml` connects to that port;
ports 9091 and 9093 are no longer Flower listeners. SuperNodes still connect to
server port 9092, using the CA certificate. Server workers also verify that CA.
SuperExec now uses `--runtime-api-address`; the old `--plugin-type` and
`--appio-api-address` options are removed. See Flower's
[TLS guide](https://flower.ai/docs/framework/how-to-enable-tls-connections.html).

Moving the server Runtime API off loopback requires worker authentication.
`deploy.sh` creates a random 32-byte `flower-superexec-auth` Kubernetes Secret
only when absent and retains it on subsequent deployments. Only SuperLink and
the server worker mount this secret. Both use `--superexec-auth-secret-file`.
Flower labels this authentication feature experimental. It authenticates workers;
it does not add account authentication to the Control API or participant identity
checks to the Fleet API. Preserve the deployment's network access restrictions.
The client Runtime API stays on loopback port 9094 with its existing local HTTP
connection. Runtime dependency installation remains disabled on the server.

Rebuild the application images and deploy them together with the 1.43 SuperLink
and SuperNode images and the new secret. Do not mix the 1.35 and 1.43 runtime
components. Existing server certificates can be reused if their SANs cover the
server address and localhost; deployment reruns preserve them. A rollback must
restore all runtime images, manifests and the old CLI configuration together.
No live cluster was upgraded during this validation.

## Verification

Validate an installed environment without changing the cluster:

```sh
python scripts/check-flower-runtime.py --output-dir /tmp/aide-flower-check
```

The check starts isolated SuperLink, SuperNode and two SuperExec processes with
temporary TLS certificates and a random server worker secret. It verifies that a
caller holding only the CA receives HTTP 401 when acquiring a task, runs a
training and evaluation round through the authenticated worker, and asserts the
aggregated weight. It retains submission, status and logs, then stops its
processes and deletes the temporary private key and worker secret. This checks
transport and worker execution, not smoke-detection quality or cluster rollout.

The 1.43 authenticated round completed as run 5126280952805898079. The rejected
unauthenticated request returned `Authentication failed`. All four deployment
manifests parsed with the new CLI arguments and server secret mounts. The 16
deployment and Flask-wrapper tests passed, including preservation of existing
secrets and creation without exposing secret bytes in command arguments.

Evidence is retained under
`final-report/working/wp4-validation/flower-upgrade-20261111`. The complete
application FAB built on 1.43, and `uv pip check` reported all 121 installed
packages compatible in the WP4 verification environment. Its package inventory
is retained alongside the transport logs.
TensorFlow CPU `train_on_batch`, scikit-learn logistic-regression `fit/predict`
and XGBoost `fit/predict` also completed with these installed versions. GPU
execution and rebuilt deployment containers were not tested.
