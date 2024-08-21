import argparse
import subprocess
import time

def main():
    parser = argparse.ArgumentParser(description='Start the federated learning server.')
    parser.add_argument('--experiment', required=True, help='Experiment name (e.g., FedAvg, FedProx, FedBN, APFed)')
    parser.add_argument('--server', default='0.0.0.0:8080', help='Server address')
    parser.add_argument('--min_num_clients', type=int, default=2, help='Minimum number of clients required to start training')
    parser.add_argument('--rounds', type=int, default=5, help='Number of rounds of training')
    
    args = parser.parse_args()

    while True:
        print("Starting server...")
        subprocess.run(["python", "-m", f"src.experiments.{args.experiment}.server", "--server", args.server, "--min_num_clients", str(args.min_num_clients), "--rounds", str(args.rounds)])
        print("Server stopped. Restarting...")
        time.sleep(5)

if __name__ == "__main__":
    main()