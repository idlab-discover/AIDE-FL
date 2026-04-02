from flower.frameworks.base import FrameworkAdapter
from flower.frameworks.tensorflow_adapter import TensorFlowAdapter


_ADAPTERS: dict[str, FrameworkAdapter] = {
    "tensorflow": TensorFlowAdapter(),
}


def get_adapter(framework: str) -> FrameworkAdapter:
    try:
        return _ADAPTERS[framework]
    except KeyError as exc:
        raise ValueError(f"unsupported framework: {framework}") from exc


def get_adapter_for_scenario(scenario: dict) -> FrameworkAdapter:
    framework = scenario.get("framework")
    if not isinstance(framework, str) or framework == "":
        raise ValueError("scenario is missing framework")
    return get_adapter(framework)
