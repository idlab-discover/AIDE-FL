import threading
import subprocess
import os
import sys
import json

from flask import Flask, jsonify, request
import wandb


flask_app = None

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
    try:
        print("Starting Flower federated learning training...")
        
        # Use subprocess to run flwr run command
        result = subprocess.run(
            ["flwr", "run", ".", "--format", "json"],
            capture_output=True,
            text=True,
            cwd=os.getcwd()  # Ensure we're in the right directory
        )
        
        dict_output = json.loads(result.stdout)
        
        dict_output["run-id"] = str(dict_output["run-id"])
            
        output = json.dumps(dict_output)
        
        return output

    except Exception as e:
        print(f"Training failed with exception: {str(e)}")
        
        
def training_status(run_id: str) -> str:
    """Get training status from Flower"""
    try:
        print("Running command to get training run status...")
        
        if run_id:
            result = subprocess.run(
                ["flwr", "ls", "--run-id", run_id, "--format", "json"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
        else:
            result = subprocess.run(
                ["flwr", "ls", "--format", "json"],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
        dict_output = json.loads(result.stdout)
        
        for run in dict_output["runs"]:
            run["run-id"] = str(run["run-id"])
            
        output = json.dumps(dict_output)
        
        return output
    
    except Exception as e:
        print(f"Getting training status failed with exception: {str(e)}")
        

def main():
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    setup_flask()
    subprocess.run(["wandb", "login"])
    subprocess.run(["flwr-serverapp"] + sys.argv[1:])


if __name__ == "__main__":
    main()