# Dependency versions and upgrade validation

The November 2026 upgrade uses Python 3.12, Flower 1.35.0, TensorFlow 2.20.0,
scikit-learn 1.9.1 and XGBoost 3.4.1. Container requirements pin direct dependencies;
`pyproject.toml` permits only the Flower 1.35 series because FAB format 1 requires
an inclusive lower bound. Its target runtime remains 1.35.0.

Flower 1.43.0 was the latest release checked on 10 November 2026. It replaces the
external Fleet and Control gRPC connections with HTTP and changes runtime flags.
The 1.35.0 upgrade retains the deployment's external gRPC/TLS connections and
includes native differential-privacy strategies and privacy accounting. Moving
to 1.43 requires a separate transport migration and corresponding TLS validation.
The installed 1.35.0 and 1.43.0 CLI help and package code establish these interface
differences. Release artifacts are available on [PyPI](https://pypi.org/project/flwr/).

TensorFlow 2.20.0 follows the version pinned by Flower's
[TensorFlow quickstart](https://github.com/flwrlabs/flower/blob/main/examples/quickstart-tensorflow/pyproject.toml)
at the inspection date, although TensorFlow 2.21.0 was available. Flower's
[XGBoost quickstart](https://github.com/flwrlabs/flower/blob/main/examples/quickstart-xgboost/pyproject.toml)
accepts XGBoost 2.0 or later. The selected scikit-learn and XGBoost versions were
the current stable releases on [scikit-learn PyPI](https://pypi.org/project/scikit-learn/1.9.1/)
and [XGBoost PyPI](https://pypi.org/project/xgboost/3.4.1/).

The client container now runs a persistent `flower-superexec` worker. The
`flwr-clientapp` entry point requires a token for an individual task. Internal
Runtime HTTP APIs bind to loopback on ports 9091 and 9094; the external Fleet and
Control connections retain TLS on ports 9092 and 9093. Runtime dependency
installation is disabled on the server so workers use the installed container
environment. FAB format 1 also requires the bundled license file.

Validate an installed environment without changing the cluster:

```sh
python scripts/check-flower-runtime.py --output-dir /tmp/aide-flower-check
```

This check starts isolated SuperLink, SuperNode and two SuperExec processes. It
uses temporary TLS certificates, runs one training and evaluation round, and
asserts the aggregated weight. It retains submission, status and logs, then
stops its processes and deletes the temporary private key. This tests transport
and worker execution, not smoke-detection model quality or a cluster deployment.

The upgrade also passed package consistency checks, the repository tests and
the preprocessing/trainer service tests. The application FAB builds with
`flwr build --app flower`. Run evidence is retained under
`final-report/working/wp4-validation/flower-upgrade-20261110`.
