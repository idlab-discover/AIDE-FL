#!/usr/bin/env python3
"""Publish completed PyTorch evidence and verify the remote metrics and files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from publish import same_measurement


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_files(root, hashes):
    for name, expected in hashes.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f"Evidence checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    hashes = json.loads((args.evidence / "checksums.json").read_text())
    verify_files(args.evidence, hashes)
    summary = json.loads((args.evidence / "summary.json").read_text())
    scenario = json.loads((args.evidence / "scenario.json").read_text())
    manifest = json.loads((args.evidence / "manifest.json").read_text())
    metrics = [json.loads(line) for line in (args.evidence / "metrics.jsonl").read_text().splitlines()]
    if [row["round"] for row in metrics] != list(range(1, summary["rounds"] + 1)):
        raise ValueError("incomplete training evidence")
    transport_log = (args.transport / "runtime/run.log").read_text()
    if "PYTORCH_PRODUCTION_APPS_AND_AGGREGATE_VERIFIED" not in transport_log:
        raise ValueError("transport experiment did not pass")
    transport_hashes = {str(p.relative_to(args.transport)): digest(p) for p in sorted(args.transport.rglob("*"))
                        if p.is_file() and p.name != "checksums.json" and "__pycache__" not in p.parts}
    (args.transport / "checksums.json").write_text(json.dumps(transport_hashes, indent=2) + "\n")
    os.environ["WANDB_API_KEY"] = args.key_file.read_text().strip()
    import wandb
    run_id = hashlib.sha256((digest(args.evidence / "checksums.json") + digest(args.transport / "checksums.json")).encode()).hexdigest()[:16]
    with tempfile.TemporaryDirectory(prefix="aide-pytorch-publish-") as tmp:
        with wandb.init(project="aide-wp4-smoke", id=run_id, resume="allow", mode="online",
                        name="pytorch-fedavg-c2-s42-20261111", job_type="pytorch-verification",
                        config={"scenario": scenario, "manifest": manifest}, dir=tmp,
                        tags=["pytorch", "wp4", "smoke-detection", "local-training", "isolated-tls-transport"],
                        settings=wandb.Settings(disable_git=True, save_code=False)) as run:
            if not run.resumed:
                for row in metrics:
                    flat = {"round": row["round"]}
                    for group, values in row.items():
                        if isinstance(values, dict):
                            flat.update({f"{group}/{k}": v for k, v in values.items()})
                    run.log(flat)
            run.summary.update(summary)
            run.summary["transport_verified"] = True
            artifacts = []
            for kind, root, checksums in (("training", args.evidence, hashes), ("transport", args.transport, transport_hashes)):
                artifact = wandb.Artifact(f"pytorch-{kind}-{run_id}", type="experiment-evidence")
                for name in [*checksums, "checksums.json"]:
                    artifact.add_file(str(root / name), name=name)
                committed = run.log_artifact(artifact)
                committed.wait()
                artifacts.append(committed.qualified_name)
            receipt = {"url": run.url, "path": run.path, "id": run.id, "artifacts": artifacts}
        api = wandb.Api()
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
        remote = api.run(receipt["path"])
        for _ in range(10):
            if remote.state == "finished" and remote.summary_metrics.get("transport_verified"):
                break
            time.sleep(2)
            api.flush()
            remote = api.run(receipt["path"])
        if remote.state != "finished" or remote.summary_metrics.get("transport_verified") is not True:
            raise RuntimeError("remote run is not complete")
        for key, value in summary["final_holdout"].items():
            if not same_measurement(remote.summary_metrics.get("final_holdout." + key), value):
                raise RuntimeError(f"remote summary differs: {key}")
        history = remote.history(samples=10000, pandas=False)
        by_round = {row["round"]: row for row in history if "round" in row}
        for row in metrics:
            for group, values in row.items():
                if isinstance(values, dict):
                    for key, value in values.items():
                        if not same_measurement(by_round.get(row["round"], {}).get(f"{group}/{key}"), value):
                            raise RuntimeError("remote history differs")
        for artifact_name in artifacts:
            artifact = api.artifact(artifact_name)
            if artifact.state != "COMMITTED":
                raise RuntimeError("artifact not committed")
            downloaded = Path(artifact.download(root=str(Path(tmp) / artifact.name)))
            verify_files(downloaded, json.loads((downloaded / "checksums.json").read_text()))
        receipt.update(verified_state=remote.state, verified_history_rounds=len(by_round), downloaded_artifact_hashes_verified=True)
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
