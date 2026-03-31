import json

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Error,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.clientapp import ClientApp

from flower.task import *

model = local_epochs = batch_size = None
x_train = y_train = x_eval = y_eval = x_test = y_test = None
STATE_KEY = "tensorflow-client"
ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
HISTORY_KEY = "history"


def _as_int(value):
    if value is None:
        return None
    return int(value)


def _as_float(value):
    if value is None:
        return None
    return float(value)


def _ensure_client_state(context: Context, config: ConfigRecord):
    global model, local_epochs, batch_size
    global x_train, y_train, x_eval, y_eval, x_test, y_test

    state_config = context.state.config_records.get(STATE_KEY)
    model_data_str = config["model_data"]
    label_column = config["label_column"]

    reload_local_state = (
        state_config is None
        or state_config.get("model_data") != model_data_str
        or state_config.get("label_column") != label_column
    )

    local_epochs = _as_int(config["local_epochs"])
    batch_size = _as_int(config["batch_size"])

    if not reload_local_state and model is not None:
        return

    print("Loading model and datasets for current run")
    model_data = json.loads(model_data_str)
    model = keras.models.model_from_json(model_data_str)
    x_train, y_train, x_eval, y_eval, x_test, y_test = load_data(label_column, model_data)

    context.state[STATE_KEY] = ConfigRecord(
        {
            "model_data": model_data_str,
            "label_column": label_column,
        }
    )


def _require_arrays(message: Message) -> ArrayRecord:
    arrays = message.content.array_records.get(ARRAYS_KEY)
    if arrays is None:
        raise ValueError(f"missing '{ARRAYS_KEY}' ArrayRecord")
    return arrays


def _require_config(message: Message) -> ConfigRecord:
    config = message.content.config_records.get(CONFIG_KEY)
    if config is None:
        raise ValueError(f"missing '{CONFIG_KEY}' ConfigRecord")
    return config


def _error_reply(message: Message, exc: Exception) -> Message:
    return Message(error=Error(code=400, reason=str(exc)), reply_to=message)


app = ClientApp()


@app.train()
def train(message: Message, context: Context) -> Message:
    global model, x_train, y_train, x_eval, y_eval

    try:
        config = _require_config(message)
        _ensure_client_state(context, config)

        model.set_weights(_require_arrays(message).to_numpy_ndarrays())
        history = model.fit(
            x=x_train,
            y=y_train,
            epochs=local_epochs,
            batch_size=batch_size,
            validation_data=(x_eval, y_eval),
            verbose=context.run_config.get("verbose"),
        )

        history_metrics = history.history
        reply = RecordDict(
            {
                ARRAYS_KEY: ArrayRecord(model.get_weights()),
                METRICS_KEY: MetricRecord(
                    {
                        "num-examples": len(x_train),
                        "train-loss": _as_float(history_metrics["loss"][-1]),
                        "train-val-loss": _as_float(history_metrics.get("val_loss", [0.0])[-1]),
                    }
                ),
                HISTORY_KEY: ConfigRecord({"history": json.dumps(history_metrics)}),
            }
        )
        return message.create_reply(reply)
    except Exception as exc:
        return _error_reply(message, exc)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    global model, x_test, y_test

    try:
        config = _require_config(message)
        _ensure_client_state(context, config)

        model.set_weights(_require_arrays(message).to_numpy_ndarrays())
        evaluation = model.evaluate(x_test, y_test, verbose=0, return_dict=True)

        metrics = {
            "num-examples": len(x_test),
            "loss": _as_float(evaluation.pop("loss")),
        }
        for key, value in evaluation.items():
            metrics[key] = _as_float(value)

        return message.create_reply(
            RecordDict(
                {
                    METRICS_KEY: MetricRecord(metrics),
                }
            )
        )
    except Exception as exc:
        return _error_reply(message, exc)
