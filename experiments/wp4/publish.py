"""Plot retained WP4 measurements and optionally publish them to W&B."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


def read_runs(root):
    runs = []
    for path in sorted(root.rglob("metrics.jsonl")):
        directory = path.parent
        config = json.loads((directory / "config.json").read_text())
        summary = json.loads((directory / "summary.json").read_text())
        metrics = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not metrics:
            raise ValueError(f"No measurements in {path}")
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
        if receipt["remote_test_f1"] != local_summary["test_f1"] or receipt["remote_history_rows"] < rounds:
            raise RuntimeError(f"W&B run is missing retained metrics: {receipt['url']}")
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
