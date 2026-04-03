from importlib import import_module

from flower.frameworks.base import FrameworkAdapter


_ADAPTER_TYPES: dict[str, tuple[str, str]] = {
    "tensorflow": ("flower.frameworks.tensorflow_adapter", "TensorFlowAdapter"),
    "scikit_learn": ("flower.frameworks.scikit_learn_adapter", "ScikitLearnAdapter"),
    "xgboost": ("flower.frameworks.xgboost_adapter", "XGBoostAdapter"),
}
_ADAPTERS: dict[str, FrameworkAdapter] = {}


def get_adapter(framework: str) -> FrameworkAdapter:
    adapter = _ADAPTERS.get(framework)
    if adapter is not None:
        return adapter

    try:
        module_name, class_name = _ADAPTER_TYPES[framework]
    except KeyError as exc:
        raise ValueError(f"unsupported framework: {framework}") from exc

    module = import_module(module_name)
    adapter_cls = getattr(module, class_name)
    adapter = adapter_cls()
    _ADAPTERS[framework] = adapter
    return adapter


def get_adapter_for_scenario(scenario: dict) -> FrameworkAdapter:
    framework = scenario.get("framework")
    if not isinstance(framework, str) or framework == "":
        raise ValueError("scenario is missing framework")
    return get_adapter(framework)
