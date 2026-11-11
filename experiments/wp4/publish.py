"""Plot retained WP4 measurements and optionally publish them to W&B."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path


def same_measurement(actual, expected):
    if isinstance(expected, float) and isinstance(actual, (float, int)):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    if isinstance(expected, list) and isinstance(actual, list):
        return len(actual) == len(expected) and all(
            same_measurement(a, e) for a, e in zip(actual, expected))
    return actual == expected


def summary_items(summary, prefix=""):
    # W&B's public API flattens nested summary mappings into dotted keys.
    for key, value in summary.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            yield from summary_items(value, name + ".")
        else:
            yield name, value


def validate_evidence(directory, config, metrics):
    """Refuse incomplete or changed results before plotting or uploading."""
    rounds = [row.get("round") for row in metrics]
    if rounds != list(range(1, config["rounds"] + 1)):
        raise ValueError(f"Incomplete or duplicate training rounds in {directory}")
    checksums = json.loads((directory / "checksums.json").read_text())
    for required in ("config.json", "metrics.jsonl", "summary.json"):
        if required not in checksums:
            raise ValueError(f"Missing checksum for {directory / required}")
    for name, expected in checksums.items():
        path = directory / name
        if path.parent != directory or not path.is_file():
            raise ValueError(f"Invalid evidence file: {path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Evidence checksum mismatch: {path}")


def read_runs(root):
    runs = []
    for path in sorted(root.rglob("metrics.jsonl")):
        directory = path.parent
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        metrics = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not metrics:
            raise ValueError(f"No measurements in {path}")
        validate_evidence(directory, config, metrics)
        runs.append((directory, config, summary, metrics))
    if not runs:
        raise ValueError(f"No completed runs in {root}")
    return runs


def plot_runs(root, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures = root / "figures"
    figures.mkdir(exist_ok=True)
    methods = sorted({config["method"] for _, config, _, _ in runs})
    colors = dict(zip(methods, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for _, config, _, metrics in runs:
        method = config["method"]
        for ax, metric, title in zip(axes, ["validation_logloss", "validation_f1"],
                                     ["Validation log loss", "Validation F1"]):
            if metric in metrics[0]:
                ax.plot([row["round"] for row in metrics], [row[metric] for row in metrics],
                        color=colors[method], alpha=0.65, label=method)
            ax.set(xlabel="Training round", ylabel=title)
            ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes[0].legend(unique.values(), unique.keys(), fontsize=8)
    fig.suptitle("Smoke sensor classification: each line is one seed")
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"validation-curves.{extension}", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, ["test_f1", "test_balanced_accuracy", "test_roc_auc"],
                                 ["F1", "Balanced accuracy", "ROC AUC"]):
        for index, method in enumerate(methods):
            scores = [summary[metric] for _, config, summary, _ in runs if config["method"] == method]
            offsets = np.linspace(-0.08, 0.08, len(scores)) if len(scores) > 1 else [0]
            ax.scatter(index + np.array(offsets), scores, color=colors[method], label=method)
            ax.plot([index - .18, index + .18], [np.mean(scores)] * 2, color=colors[method])
        baseline = runs[0][2].get("majority_baseline_" + metric, None if metric == "test_f1" else .5)
        if baseline is not None:
            ax.axhline(baseline, color="grey", linestyle="--", linewidth=1, label="Majority / chance")
        ax.set(xticks=range(len(methods)), xticklabels=methods, ylabel=title)
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Untouched test period: seeds, mean, and majority/chance reference")
    for extension in ("png", "pdf"):
        fig.savefig(figures / f"heldout-comparison.{extension}", dpi=180)
    plt.close(fig)
    if all("global_train_balanced_accuracy" in rows[0] for _, _, _, rows in runs):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        for _, config, _, rows in runs:
            for ax, metric, title in zip(
                    axes, ["global_train_balanced_accuracy", "validation_balanced_accuracy"],
                    ["Training balanced accuracy", "Validation balanced accuracy"]):
                ax.plot([row["round"] for row in rows], [row[metric] for row in rows],
                        color=colors[config["method"]], alpha=.65, label=config["method"])
                ax.set(xlabel="Training round", ylabel=title, ylim=(0, 1))
                ax.axhline(.5, color="grey", linestyle="--", linewidth=1)
                ax.grid(alpha=.2)
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        axes[0].legend(unique.values(), unique.keys(), fontsize=8)
        fig.suptitle("Global model on training and later validation periods")
        for extension in ("png", "pdf"):
            fig.savefig(figures / f"training-validation.{extension}", dpi=180)
        plt.close(fig)
    return figures


def publish(root, runs, figures, args):
    import wandb

    if args.key_file:
        os.environ["WANDB_API_KEY"] = args.key_file.read_text().strip()
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("Online publication requires WANDB_API_KEY or --key-file")
    receipts = []
    group = args.group or root.name
    for directory, config, summary, metrics in runs:
        digest = hashlib.sha256((directory / "metrics.jsonl").read_bytes()).hexdigest()
        run_id = hashlib.sha256(f"{group}/{directory.name}/{digest}".encode()).hexdigest()[:16]
        with wandb.init(project=args.project, entity=args.entity, group=group,
                        id=run_id, resume="allow", name=directory.name, config=config,
                        mode="online", job_type="wp4-validation", dir=str(root),
                        tags=["wp4", "smoke-detection", "local-simulation"],
                        settings=wandb.Settings(disable_git=True, save_code=False)) as run:
            run.define_metric("round")
            run.define_metric("*", step_metric="round")
            if not run.resumed:
                for row in metrics:
                    run.log(row)
            run.summary.update(summary)
            run.summary["evidence_scope"] = "Local experiment; not distributed deployment evidence"
            artifact = wandb.Artifact(f"wp4-{run_id}", type="experiment-evidence",
                                      metadata={"metrics_sha256": digest})
            for name in ("config.json", "metrics.jsonl", "summary.json", "model.joblib",
                         "checksums.json", "model-scenario.json", "test-predictions.csv"):
                path = directory / name
                if path.is_file():
                    artifact.add_file(str(path), name=name)
            for pattern in ("*.keras", "*.weights.h5", "*.safetensors", "*_privacy.json"):
                for path in sorted(directory.glob(pattern)):
                    artifact.add_file(str(path), name=path.name)
            for name in ("manifest.json", "experiment.json", "provenance.json",
                         f"partitions-seed-{config['seed']}.csv",
                         f"partition-counts-seed-{config['seed']}.csv"):
                if (root / name).is_file():
                    artifact.add_file(str(root / name), name=name)
            for path in sorted((root / "source").glob("*.py")):
                artifact.add_file(str(path), name=f"source/{path.name}")
            for path in sorted(figures.glob("*")):
                artifact.add_file(str(path), name=f"figures/{path.name}")
            run.log_artifact(artifact).wait()
            run.log({"figures/validation": wandb.Image(str(figures / "validation-curves.png")),
                     "figures/heldout": wandb.Image(str(figures / "heldout-comparison.png"))})
            if (figures / "training-validation.png").is_file():
                run.log({"figures/training-validation": wandb.Image(str(figures / "training-validation.png"))})
            receipts.append({"directory": str(directory), "id": run.id, "url": run.url,
                             "path": run.path, "metrics_sha256": digest})
        (root / "wandb-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")
    verify(root, receipts)
    print(json.dumps(receipts, indent=2))


def verify(root, receipts):
    import wandb

    # The sampled-history endpoint returns all rows for these short experiments.
    # scan_history and the summary file endpoint can lag behind committed runs.
    api = wandb.Api()
    for receipt in receipts:
        remote = api.run(receipt["path"])
        for _ in range(5):
            if remote.state == "finished" and remote.summary_metrics.get("test_f1") is not None:
                break
            time.sleep(2)
            api.flush()
            remote = api.run(receipt["path"])
        if remote.state != "finished":
            raise RuntimeError(f"W&B run is not finished: {receipt['url']}: {remote.state}")
        receipt["verified_state"] = remote.state
        receipt["remote_test_f1"] = remote.summary_metrics.get("test_f1")
        history = remote.history(samples=10000, pandas=False)
        receipt["remote_history_rows"] = sum("round" in row for row in history)
        local_summary = json.loads((Path(receipt["directory"]) / "summary.json").read_text())
        rounds = json.loads((Path(receipt["directory"]) / "config.json").read_text())["rounds"]
        if not same_measurement(receipt["remote_test_f1"], local_summary["test_f1"]) or receipt["remote_history_rows"] < rounds:
            raise RuntimeError(f"W&B run is missing retained metrics: {receipt['url']}")
        local_metrics = [json.loads(line) for line in
                         (Path(receipt["directory"]) / "metrics.jsonl").read_text().splitlines()
                         if line.strip()]
        by_round = {}
        for row in history:
            if "round" in row:
                by_round.setdefault(row["round"], {}).update(row)
        for row in local_metrics:
            if any(not same_measurement(by_round.get(row["round"], {}).get(key), value)
                   for key, value in row.items()):
                raise RuntimeError(f"W&B history differs from local evidence: {receipt['url']}")
        for key, value in summary_items(local_summary):
            if not same_measurement(remote.summary_metrics.get(key), value):
                raise RuntimeError(f"W&B summary differs for {key}: {receipt['url']}")
        receipt["committed_artifacts"] = [artifact.name for artifact in remote.logged_artifacts()
                                          if artifact.state == "COMMITTED"]
        if not receipt["committed_artifacts"]:
            raise RuntimeError(f"W&B run has no committed artifact: {receipt['url']}")
    (root / "wandb-receipts.json").write_text(json.dumps(receipts, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--project", default="aide-wp4-smoke")
    parser.add_argument("--entity")
    parser.add_argument("--group")
    args = parser.parse_args()
    if args.verify_only:
        if args.key_file:
            os.environ["WANDB_API_KEY"] = args.key_file.read_text().strip()
        verify(args.root, json.loads((args.root / "wandb-receipts.json").read_text()))
        print("Verified remote W&B metrics and committed artifacts")
        return
    runs = read_runs(args.root)
    figures = plot_runs(args.root, runs)
    if args.online:
        publish(args.root, runs, figures, args)
    print(f"Figures saved in {figures}")


if __name__ == "__main__":
    main()
