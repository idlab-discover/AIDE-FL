import threading
import subprocess
import os
import sys

from flask import Flask, jsonify
from dotenv import load_dotenv
import wandb


flask_app = None

def setup_flask():
    flask_app = Flask(__name__)

    @flask_app.route("/run", methods=["POST"])
    def trigger_training():
        print("Received Flower run request")
        
        output = start_training()
        
        return output

    @flask_app.route("/ls", methods=["GET"])
    def get_training_status():
        """Get current training status"""
        return

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
        
        # Ouput of flwr command will be in json format
        return result.stdout

    except Exception as e:
        print(f"Training failed with exception: {str(e)}")
        

def main():
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    setup_flask()
    subprocess.run(["wandb", "login"])
    subprocess.run(["flwr-serverapp"] + sys.argv[1:])


if __name__ == "__main__":
    main()