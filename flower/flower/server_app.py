import json
import os
import wandb

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from flower.task import *
from flower.strategies import SaveModelFedAvg, aggregate_numeric_metrics


def get_centralised_evaluation_fn(model, x_hold_out, y_hold_out): # testloader loads the centralised test set

    def evaluate_centralised(server_round, arrays):
        model.set_weights(arrays.to_numpy_ndarrays())
        evaluation = model.evaluate(x_hold_out, y_hold_out, verbose=0, return_dict=True)

        metrics = {"loss": float(evaluation.pop("loss"))}
        for key, value in evaluation.items():
            metrics[key] = float(value)

        wandb.log({f"centralised/{key}": value for key, value in metrics.items()}, step=server_round)
        return MetricRecord(metrics)
    
    return evaluate_centralised


def make_on_fit_config(framework: str, model_data: dict, local_epochs: int, batch_size: int, label_column: str):

    print("Preparing train config for clients")
    return ConfigRecord(
        {
            "framework": framework,
            "model_config": json.dumps(model_data),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "label_column": label_column,
        }
    )


def write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=4)


app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    framework, scenario_name, hold_out_dataset, drop_cols, model_data, fraction_fit, fraction_evaluate, min_available_clients, server_rounds, local_epochs, batch_size, label_column = load_server_config()

    model = keras.models.model_from_json(json.dumps(model_data))
    initial_arrays = ArrayRecord(model.get_weights())

    evaluate_fn = None
    if hold_out_dataset:
        x_hold_out, y_hold_out = load_server_data(label_column, drop_cols)
        evaluate_fn = get_centralised_evaluation_fn(model, x_hold_out, y_hold_out)

    strategy = SaveModelFedAvg(
        scenario_name=scenario_name,
        model=model,
        share_dir="share",
        server_rounds=server_rounds,
        fraction_train=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes=1,
        min_evaluate_nodes=1,
        min_available_nodes=min_available_clients,
        arrayrecord_key="arrays",
        configrecord_key="config",
        weighted_by_key="num-examples",
        train_metrics_aggr_fn=aggregate_numeric_metrics,
        evaluate_metrics_aggr_fn=aggregate_numeric_metrics,
    )
    try:
        config = make_on_fit_config(framework, model_data, local_epochs, batch_size, label_column)
        result = strategy.start(
            grid=grid,
            initial_arrays=initial_arrays,
            num_rounds=server_rounds,
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
