#!/usr/bin/env python3
"""Exercise the production PyTorch ClientApp and ServerApp over isolated Flower TLS."""
import argparse
import configparser
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "flower"))

APP = '''import os
from pathlib import Path
import numpy as np
from flwr.serverapp import ServerApp
from flwr.clientapp import ClientApp
from flower.client_app import train as actual_train, evaluate as actual_evaluate
from flower.server_app import main as actual_server
from flower.frameworks import get_adapter
from safetensors.torch import load_file
server = ServerApp()
client = ClientApp()
@server.main()
def main(grid, context):
    os.chdir(os.environ["AIDE_TORCH_SERVER"])
    actual_server(grid, context)
    assert Path("share/done").exists()
    adapter = get_adapter("pytorch")
    model = adapter.load_model("share/smoke-pytorch")
    actual = adapter.initial_arrays(model)
    expected = load_file(os.environ["AIDE_TORCH_EXPECTED"])
    for key in actual:
        np.testing.assert_allclose(actual[key].numpy(), expected[key].numpy(), rtol=1e-5, atol=1e-7)
    print("PYTORCH_PRODUCTION_APPS_AND_AGGREGATE_VERIFIED")
    print("TRANSPORT_TRAINING_PASSED")
@client.train()
def train(message, context):
    os.chdir(os.environ["AIDE_TORCH_CLIENT"])
    return actual_train(message, context)
@client.evaluate()
def evaluate(message, context):
    os.chdir(os.environ["AIDE_TORCH_CLIENT"])
    return actual_evaluate(message, context)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    scenario = json.loads((args.evidence / "scenario.json").read_text())
    scenario["training"]["server_rounds"] = 1
    scenario["aggregation"]["min_available_clients"] = 1
    server = output / "server"
    (server / "share").mkdir(parents=True)
    shutil.copy2(args.evidence / "server/share/hold-out.parquet", server / "share/hold-out.parquet")
    ini = configparser.ConfigParser()
    ini["serverapp"] = {"scenario": json.dumps(scenario)}
    with (server / "share/config.ini").open("w") as file:
        ini.write(file)
    client = output / "client"
    shutil.copytree(args.evidence / "client-0", client)
    from flower.frameworks import get_adapter
    from flower.frameworks.base import ClientDatasets
    from flower.task import load_data
    from safetensors.torch import save_file
    import torch
    torch.set_num_threads(1)
    adapter = get_adapter("pytorch")
    model = adapter.build_model_from_scenario(scenario)
    previous = Path.cwd()
    try:
        os.chdir(client)
        data = ClientDatasets(*load_data(scenario["label_column"], lambda x: adapter.adapt_features(x, model.aide_config)))
    finally:
        os.chdir(previous)
    config = adapter.make_config_record(scenario)
    config["server-round"] = 1
    result = adapter.train(model, adapter.initial_arrays(model), data, config, 0)
    expected = output / "expected.safetensors"
    save_file(result.arrays.to_torch_state_dict(), str(expected))
    env = dict(os.environ, AIDE_TORCH_SERVER=str(server), AIDE_TORCH_CLIENT=str(client),
               AIDE_TORCH_EXPECTED=str(expected), WANDB_MODE="disabled", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="aide-pytorch-app-") as tmp:
        app = Path(tmp)
        (app / "app.py").write_text(APP)
        shutil.copytree(ROOT / "flower/flower", app / "flower", ignore=shutil.ignore_patterns("__pycache__"))
        subprocess.run([sys.executable, str(ROOT / "scripts/check-flower-runtime.py"),
                        "--app-dir", str(app), "--output-dir", str(output / "runtime")], env=env, check=True)
    print(f"PyTorch transport verified; evidence: {output}")


if __name__ == "__main__":
    main()
