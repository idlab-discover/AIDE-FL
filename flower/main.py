import json
import os
import subprocess
import sys
import threading

from flask import Flask, jsonify, request
import wandb


flask_app = None
flower_app_path = "."
flower_superlink = "deployment"


def normalize_payload(payload: dict) -> dict:
    payload.setdefault("runs", [])

    if "run-id" in payload:
        payload["run-id"] = str(payload["run-id"])

    for run in payload.get("runs", []):
        if "run-id" in run:
            run["run-id"] = str(run["run-id"])

    return payload


def run_flower_command(command: list[str]) -> tuple[dict, int]:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            check=False,
        )
    except OSError as exc:
        return {
            "success": False,
            "runs": [],
            "error-message": str(exc),
        }, 503

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if not stdout:
        payload = {
            "success": False,
            "runs": [],
            "error-message": stderr or f"command failed with exit code {result.returncode}",
        }
        return payload, 503

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        payload = {
            "success": False,
            "runs": [],
            "error-message": stderr or stdout,
        }
        return payload, 503

    payload = normalize_payload(payload)

    if result.returncode != 0 or payload.get("success") is False:
        if stderr and "error-message" not in payload:
            payload["error-message"] = stderr
        return payload, 503

    return payload, 200


def build_run_command() -> list[str]:
    return ["flwr", "run", flower_app_path, flower_superlink, "--format", "json"]


def build_ls_command(run_id: str | None = None) -> list[str]:
    if run_id:
        return ["flwr", "ls", "--run-id", run_id, flower_superlink, "--format", "json"]
    return ["flwr", "ls", flower_superlink, "--format", "json"]


def build_serverapp_command(args: list[str]) -> list[str]:
    command = ["flower-superexec", "--plugin-type", "serverapp"]

    normalized_args: list[str] = []
    for arg in args:
        if arg == "--serverappio-api-address":
            normalized_args.append("--appio-api-address")
            continue
        normalized_args.append(arg)

    return command + normalized_args

def setup_flask():
    flask_app = Flask(__name__)

    @flask_app.route("/run", methods=["POST"])
    def trigger_training():
        print("Received Flower run request")
        
        return start_training()

    @flask_app.route("/ls", methods=["GET"])
    def get_training_status():
        """Get current training status"""
        
        run_id = request.args.get("runId")
        return training_status(run_id)
        
    @flask_app.route("/stop", methods=["POST"])
    def health_check():
        """Health check endpoint"""
        return

    # Start Flask server in background thread
    flask_thread = threading.Thread(
        target=flask_app.run, 
        kwargs={"host": "0.0.0.0", "port": 8080, "debug": False}
    )
    flask_thread.daemon = True
    flask_thread.start()
    print("Flask server started on port 8080")


def start_training() -> str:
    """Start the federated learning training process"""
    print("Starting Flower federated learning training...")
    payload, status_code = run_flower_command(build_run_command())
    return jsonify(payload), status_code
        
        
def training_status(run_id: str) -> str:
    """Get training status from Flower"""
    print("Running command to get training run status...")

    payload, status_code = run_flower_command(build_ls_command(run_id))
    return jsonify(payload), status_code
        

def main():
    key = os.getenv("WANDB_API_KEY")
    mode = os.getenv("WANDB_MODE") or ("online" if key else "offline")
    if mode not in {"offline", "disabled", "online"}:
        raise ValueError("WANDB_MODE must be offline, disabled, or online")
    os.environ["WANDB_MODE"] = mode
    if mode == "online":
        if not key:
            raise ValueError("Online W&B requires WANDB_API_KEY")
        wandb.login(key=key)

    setup_flask()
    subprocess.run(build_serverapp_command(sys.argv[1:]))


if __name__ == "__main__":
    main()
