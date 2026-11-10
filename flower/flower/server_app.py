import json
import os
import wandb

from flwr.app import Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from flower.frameworks import get_adapter_for_scenario
from flower.frameworks.base import HoldOutDataset
from flower.strategies import build_strategy
from flower.task import load_server_config, load_server_data
from flower.privacy import validate_privacy


def get_centralised_evaluation_fn(adapter, model, x_hold_out, y_hold_out):
    dataset = HoldOutDataset(x_hold_out=x_hold_out, y_hold_out=y_hold_out)

    def evaluate_centralised(server_round, arrays):
        metrics = adapter.evaluate_centralised(model, arrays, dataset)
        wandb.log({f"centralised/{key}": value for key, value in metrics.items()}, step=server_round)
        return MetricRecord(dict(metrics))
    return evaluate_centralised


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    del context
    scenario = load_server_config()
    privacy = validate_privacy(scenario["aggregation"], scenario["framework"])
    if privacy is not None and scenario.get("hold_out_dataset"):
        raise ValueError("central_dp disables holdout evaluation because its metrics are not accounted")
    adapter = get_adapter_for_scenario(scenario)

    scenario_name = scenario["name"]
    aggregation = scenario["aggregation"]
    training = scenario["training"]
    label_column = scenario["label_column"]
    drop_cols = scenario["preprocessing"]["drop_columns"]
    model_config = adapter.model_config_from_scenario(scenario)

    model = adapter.build_model_from_scenario(scenario)
    initial_arrays = adapter.initial_arrays(model)
    evaluate_fn = None
    if scenario["hold_out_dataset"]:
        x_hold_out, y_hold_out = load_server_data(
            label_column,
            drop_cols,
            adapt_features=lambda x: adapter.adapt_features(x, model_config),
        )
        evaluate_fn = get_centralised_evaluation_fn(adapter, model, x_hold_out, y_hold_out)

    strategy = build_strategy(
        aggregation,
        scenario_name=scenario_name,
        model=model,
        adapter=adapter,
        share_dir="share",
        server_rounds=training["server_rounds"],
    )
    try:
        config = adapter.make_config_record(scenario)
        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=training["server_rounds"],
            train_config=config,
            evaluate_config=config,
            evaluate_fn=evaluate_fn,
        )

        centralised_metrics = {
            round_nr: dict(metrics) for round_nr, metrics in result.evaluate_metrics_serverapp.items()
        }
        if centralised_metrics:
            path = os.path.join("share", f"{scenario_name}_centralised_metrics.json")
            write_json(path, centralised_metrics)
            print(f"Updated centralised metrics in {path}")

        open(os.path.join("share", "done"), "x")
    finally:
        wandb.finish()
