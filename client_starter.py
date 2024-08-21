import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='Start a federated learning client.')
    parser.add_argument('--cid', required=True, help='Client ID in range [0, 3]')
    parser.add_argument('--experiment', required=True, help='Experiment name (e.g., FedAvg, FedProx, FedBN, APFed)')
    parser.add_argument('--server', default='0.0.0.0:8080', help='Server address, format: xx.xx.xx.xx:yyyy')
    args = parser.parse_args()

    while True:
        print("Connecting to server...")
        subprocess.run(["python", f"-m", f"src.experiments.{args.experiment}.client", "--cid", args.cid, "--server", args.server])
        print("Waiting for server to become available...")
        time.sleep(30)

if __name__ == "__main__":
    main()