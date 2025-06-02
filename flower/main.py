import threading
import subprocess
import os
import sys

from flask import Flask, jsonify


flask_app = None
training_status = {"is_running": False, "last_result": None}

def setup_flask():
    flask_app = Flask(__name__)

    @flask_app.route("/train", methods=["POST"])
    def trigger_training():
        print("Received training trigger request")
        
        if training_status["is_running"]:
            return jsonify({
                "status": "error",
                "message": "Training is already in progress"
            }), 400
        
        # Start training in background thread
        training_thread = threading.Thread(target=start_training)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            "status": "success",
            "message": "Flower federated learning training started"
        })

    @flask_app.route("/train/status", methods=["GET"])
    def get_training_status():
        """Get current training status"""
        return jsonify(training_status)

    @flask_app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint"""
        return jsonify({"status": "healthy"})

    # Start Flask server in background thread
    flask_thread = threading.Thread(
        target=flask_app.run, 
        kwargs={"host": "0.0.0.0", "port": 8080, "debug": False}
    )
    flask_thread.daemon = True
    flask_thread.start()
    print("Flask server started on port 8080")

def start_training():
    """Start the federated learning training process"""
    try:
        training_status["is_running"] = True
        print("Starting Flower federated learning training...")
        
        # Use subprocess to run flwr run command
        result = subprocess.run(
            ["flwr", "run", "."],
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
            cwd=os.getcwd()  # Ensure we're in the right directory
        )
        
        training_status["last_result"] = {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "success": result.returncode == 0
        }
        
        if result.returncode == 0:
            print("Training completed successfully")
        else:
            print(f"Training failed with return code {result.returncode}")
            print(f"Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("Training timed out")
        training_status["last_result"] = {
            "error": "Training timed out after 1 hour",
            "success": False
        }
    except Exception as e:
        print(f"Training failed with exception: {str(e)}")
        training_status["last_result"] = {
            "error": str(e),
            "success": False
        }
    finally:
        training_status["is_running"] = False


def main():
    setup_flask()
    subprocess.run(["flwr-serverapp"] + sys.argv[1:])


if __name__ == "__main__":
    main()