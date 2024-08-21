# 🌸 Federated Learning with Flower Framework 🌸

Welcome to the Federated Learning (FL) project using the powerful Flower framework! 🚀 This repository showcases the fascinating world of Federated Learning, where multiple clients join forces with a central server to collaboratively train a model without sharing their data. 🌐🤝

## 📚 Project Overview

---

## 🚀 Getting Started

### 📂 Cloning the Repository

Start by cloning this repository to your local machine or a designated testbed server:

```bash
git clone <repository-url>
```

### 🛠️ Setting Up the Environment

To set up your environment, you should aim to use the Docker containers locally or to run the system on a (managed) k8s instance.

#### Option 1: Using Docker Containers

Docker provides a consistent environment that simplifies dependency management. Follow these steps to set up using Docker:

1. **Pull Docker Images**

   Pull the pre-built Docker images for both the server and clients from the registry:

   ```cpp
   docker pull [TODO build & push cnl image to github] // Server

   docker pull [TODO build & push cnl image to github] // Client
   ```

2. **Run Server Container**

   Start the server in a Docker container, exposing the necessary ports and mounting volumes if required:

   ```bash
   docker run -p 8080:8080 -v /path/to/data:/app/dataset --network host gitlab.ilabt.imec.be:4567/aide-fl/aide-infra/server \
   --experiment <experiment-name> \
   --server 0.0.0.0:8080
   ```

   This command runs the server, maps port `8080`, and mounts the `data` directory from the host to the container.

3. **Run Client Containers**

   To connect clients to the server, run the client containers:

   ```bash
   docker run -v /path/to/data:/app/dataset --network host gitlab.ilabt.imec.be:4567/aide-fl/aide-infra/client \
   --cid <client-id> \
   --experiment <experiment-name> \
   --server <server-address>
   ```

   Use `--network host` to ensure proper network configuration for local testing. Adjust the server address as needed.

#### Option 2: Setting up on k8s

If you prefer to not run the Docker containers locally, but deploy them directly onto k8s then there are two options.

Option 1 (recommended): CloudNativeLab

Option 2: local k8s (microk8s, kubeadm)

## 📊 Monitoring and Analysis

### Using TensorBoard

Monitor training progress and visualize metrics with TensorBoard:

```bash
tensorboard --logdir=logs
```

This however is not available for every experiment :exclamation:

With these options, you can choose the setup method that best fits your needs. Enjoy exploring Federated Learning with Flower! 🌸
