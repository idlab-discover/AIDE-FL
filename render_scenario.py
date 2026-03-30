#!/usr/bin/env python3

import argparse
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def render_scenario(path: Path) -> dict:
    scenario = load_json(path)
    model = scenario.get("model")
    model_file = scenario.get("model_file")

    if model:
        scenario.pop("model_file", None)
        return scenario

    if not model_file:
        return scenario

    model_path = Path(model_file)
    if not model_path.is_absolute():
        model_path = path.parent / model_path

    scenario["model"] = load_json(model_path)
    scenario.pop("model_file", None)
    return scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a scenario by expanding model_file into model.")
    parser.add_argument("scenario", help="Path to the authored scenario JSON file")
    parser.add_argument("-o", "--output", help="Optional output file path. Defaults to stdout.")
    args = parser.parse_args()

    scenario_path = Path(args.scenario).resolve()
    rendered = render_scenario(scenario_path)

    output = json.dumps(rendered, indent=2)
    if args.output:
        Path(args.output).write_text(output + "\n", encoding="utf-8")
    else:
        sys.stdout.write(output)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
