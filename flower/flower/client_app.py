import json

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Error,
    Message,
    RecordDict,
)
from flwr.clientapp import ClientApp

from flower.frameworks import get_adapter
from flower.frameworks.base import ClientDatasets
from flower.task import load_data

model = local_epochs = batch_size = None
x_train = y_train = x_eval = y_eval = x_test = y_test = None
adapter = None
STATE_KEY = "framework-client"
ARRAYS_KEY = "arrays"
CONFIG_KEY = "config"
METRICS_KEY = "metrics"
HISTORY_KEY = "history"


def _as_int(value):
    if value is None:
        return None
    return int(value)


def _ensure_client_state(context: Context, config: ConfigRecord):
    global adapter, model, local_epochs, batch_size
    global x_train, y_train, x_eval, y_eval, x_test, y_test

    state_config = context.state.config_records.get(STATE_KEY)
    framework = config["framework"]
    label_column = config["label_column"]
    adapter = get_adapter(framework)

    reload_local_state = adapter.state_reload_required(state_config, config)

    local_epochs = _as_int(config["local_epochs"])
    batch_size = _as_int(config["batch_size"])

    if not reload_local_state and model is not None:
        return

    print("Loading model and datasets for current run")
    model = adapter.restore_client_model(config)
    model_config = adapter.get_model_config_from_record(config)
    x_train, y_train, x_eval, y_eval, x_test, y_test = load_data(
        label_column,
        adapt_features=lambda x: adapter.adapt_features(x, model_config),
    )

    context.state[STATE_KEY] = adapter.client_state_record(config)


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
    global adapter, model, x_train, y_train, x_eval, y_eval, x_test, y_test

    try:
        config = _require_config(message)
        _ensure_client_state(context, config)

        result = adapter.train(
            model,
            _require_arrays(message),
            ClientDatasets(
                x_train=x_train,
                y_train=y_train,
                x_eval=x_eval,
                y_eval=y_eval,
                x_test=x_test,
                y_test=y_test,
            ),
            config,
            verbose=context.run_config.get("verbose"),
        )

        reply = RecordDict(
            {
                ARRAYS_KEY: result.arrays,
                METRICS_KEY: result.metrics,
            }
        )
        if result.history is not None:
            reply[HISTORY_KEY] = ConfigRecord({"history": json.dumps(result.history)})
        return message.create_reply(reply)
    except Exception as exc:
        return _error_reply(message, exc)


@app.evaluate()
def evaluate(message: Message, context: Context) -> Message:
    global adapter, model, x_train, y_train, x_eval, y_eval, x_test, y_test

    try:
        config = _require_config(message)
        _ensure_client_state(context, config)

        metrics = adapter.evaluate(
            model,
            _require_arrays(message),
            ClientDatasets(
                x_train=x_train,
                y_train=y_train,
                x_eval=x_eval,
                y_eval=y_eval,
                x_test=x_test,
                y_test=y_test,
            ),
            config,
        )

        return message.create_reply(
            RecordDict(
                {
                    METRICS_KEY: metrics,
                }
            )
        )
    except Exception as exc:
        return _error_reply(message, exc)
