#!/usr/bin/env python3
"""Run one isolated Flower round over TLS using the installed Python environment."""

import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


APP = '''import numpy as np
from flwr.app import ArrayRecord, RecordDict, Message, MetricRecord
from flwr.serverapp import ServerApp
from flwr.clientapp import ClientApp
from flwr.serverapp.strategy import FedAvg
server = ServerApp()
client = ClientApp()
@server.main()
def main(grid, context):
    result = FedAvg(min_available_nodes=1, min_train_nodes=1, min_evaluate_nodes=1).start(
        grid=grid, initial_arrays=ArrayRecord([np.array([1.0])]), num_rounds=1)
    assert float(result.arrays.to_numpy_ndarrays()[0][0]) == 2.0
    print("TRANSPORT_TRAINING_PASSED")
@client.train()
def train(message, context):
    arrays = message.content["arrays"].to_numpy_ndarrays()
    return Message(RecordDict({"arrays": ArrayRecord([a + 1 for a in arrays]),
        "metrics": MetricRecord({"num-examples": 1, "loss": 0.5})}), reply_to=message)
@client.evaluate()
def evaluate(message, context):
    return Message(RecordDict({"metrics": MetricRecord({"num-examples": 1, "loss": 0.4})}),
        reply_to=message)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--app-dir", type=Path, help="Use a prepared app instead of the scalar transport fixture")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="aide-flower-tls-") as tmp:
        root = Path(tmp)
        ports = []
        sockets = []
        for _ in range(2):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
        runtime, client_runtime = ports
        for sock in sockets:
            sock.close()
        env = dict(os.environ, PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"],
                   FLWR_HOME=str(root), FLWR_TELEMETRY_ENABLED="0")
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-keyout", str(root / "server.key"), "-out", str(root / "ca.crt"),
                        "-days", "1", "-subj", "/CN=localhost", "-addext",
                        "subjectAltName=IP:127.0.0.1,DNS:localhost"], check=True, capture_output=True)
        (root / "config.toml").write_text(
            f'[superlink.deployment]\naddress = "127.0.0.1:{runtime}"\n'
            f'root-certificates = "{root / "ca.crt"}"\n')
        secret = root / "superexec.secret"
        secret.write_bytes(os.urandom(32))
        secret.chmod(0o600)
        app = root / "app"
        app.mkdir()
        (app / "LICENSE").write_text("Apache-2.0\n")
        (app / "app.py").write_text(APP)
        (app / "pyproject.toml").write_text('''[project]
name = "transport-smoke"
version = "1.0.0"
license = {file = "LICENSE"}
dependencies = ["flwr>=1.43.0,<1.44.0"]
[tool.flwr.app]
publisher = "aide"
fab-format-version = 1
flwr-version-target = "1.43.0"
[tool.flwr.app.components]
serverapp = "app:server"
clientapp = "app:client"
''')
        if args.app_dir:
            shutil.copytree(args.app_dir, app, dirs_exist_ok=True)
        commands = [
            ["flower-superlink", "--ssl-ca-certfile", str(root / "ca.crt"),
             "--ssl-certfile", str(root / "ca.crt"), "--ssl-keyfile", str(root / "server.key"),
             "--isolation", "process", "--host", "127.0.0.1", "--port", str(runtime),
             "--superexec-auth-secret-file", str(secret),
             "--disable-runtime-dependency-installation"],
            ["flower-supernode", "--root-certificates", str(root / "ca.crt"),
             "--superlink", f"127.0.0.1:{runtime}", "--host", "127.0.0.1", "--port", str(client_runtime),
             "--isolation", "process"],
            ["flower-superexec", "--root-certificates", str(root / "ca.crt"),
             "--runtime-api-address", f"127.0.0.1:{runtime}",
             "--superexec-auth-secret-file", str(secret)],
            ["flower-superexec", "--insecure", "--runtime-api-address", f"127.0.0.1:{client_runtime}"],
        ]
        processes, logs = [], []

        def cli(*command):
            result = subprocess.run(["flwr", *command], env=env, capture_output=True, text=True, timeout=30)
            result.check_returncode()
            return result.stdout

        try:
            for index, command in enumerate(commands):
                log = (args.output_dir / f"process-{index}.log").open("w")
                logs.append(log)
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
                time.sleep(1)
            # A client with the CA but without the worker secret cannot acquire tasks.
            from flwr.supercore.runtime_version_compatibility import RuntimeVersionMetadata
            headers = dict(RuntimeVersionMetadata.from_local_component("SuperExec").as_metadata())
            headers["Content-Type"] = "application/protobuf"
            request = urllib.request.Request(
                f"https://127.0.0.1:{runtime}/v1/runtime/acquire-task", data=b"", headers=headers)
            try:
                urllib.request.urlopen(request, context=ssl.create_default_context(cafile=str(root / "ca.crt")), timeout=5)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode()
                if exc.code not in (401, 403) or "auth" not in body.lower():
                    raise RuntimeError(f"Unexpected worker-auth rejection: {exc.code} {body}") from exc
                (args.output_dir / "worker-auth.json").write_text(json.dumps({"status": exc.code, "body": body}, indent=2))
            else:
                raise RuntimeError("Unauthenticated worker unexpectedly acquired a task")
            response = json.loads(cli("run", str(app), "deployment", "--format", "json"))
            if not response.get("success"):
                raise RuntimeError(response)
            (args.output_dir / "submission.json").write_text(json.dumps(response, indent=2))
            run_id = str(response["run-id"])
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("A Flower service exited; inspect process logs")
                status = json.loads(cli("ls", "--run-id", run_id, "deployment", "--format", "json"))
                (args.output_dir / "status.json").write_text(json.dumps(status, indent=2))
                if any(run["status"].startswith("finished") for run in status.get("runs", [])):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("Flower run did not finish within 90 seconds")
            run_log = cli("log", run_id, "deployment", "--show")
            (args.output_dir / "run.log").write_text(run_log)
            if not any(run["status"] == "finished:completed" for run in status.get("runs", [])):
                raise RuntimeError("Flower run failed; inspect run.log")
            if "TRANSPORT_TRAINING_PASSED" not in run_log:
                raise RuntimeError("Missing transport success assertion")
            shutil.copytree(app, args.output_dir / "app", dirs_exist_ok=True)
            print(f"TLS training/evaluation passed: run {run_id}; evidence {args.output_dir}")
        finally:
            for process in processes:
                process.terminate()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for log in logs:
                log.close()


if __name__ == "__main__":
    main()
