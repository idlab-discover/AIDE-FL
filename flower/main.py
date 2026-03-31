import json
import os
import subprocess
import sys
import threading

from flask import Flask, jsonify, request
import wandb


flask_app = None


def run_flower_command(command: list[str]) -> tuple[dict, int]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )

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

    payload.setdefault("runs", [])

    if "run-id" in payload:
        payload["run-id"] = str(payload["run-id"])

    for run in payload.get("runs", []):
        if "run-id" in run:
            run["run-id"] = str(run["run-id"])

    if result.returncode != 0 or payload.get("success") is False:
        if stderr and "error-message" not in payload:
            payload["error-message"] = stderr
        return payload, 503

    return payload, 200

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
    payload, status_code = run_flower_command(["flwr", "run", ".", "--format", "json"])
    return jsonify(payload), status_code
        
        
def training_status(run_id: str) -> str:
    """Get training status from Flower"""
    print("Running command to get training run status...")

    command = ["flwr", "ls", "--format", "json"]
    if run_id:
        command = ["flwr", "ls", "--run-id", run_id, "--format", "json"]

    payload, status_code = run_flower_command(command)
    return jsonify(payload), status_code
        

def main():
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    setup_flask()
    subprocess.run(["wandb", "login"])
    subprocess.run(["flwr-serverapp"] + sys.argv[1:])


if __name__ == "__main__":
    main()
