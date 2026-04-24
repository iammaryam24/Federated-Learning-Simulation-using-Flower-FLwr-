#!/bin/bash

echo "=========================================="
echo "Federated Learning Simulation with Flower"
echo "=========================================="

# Configuration
NUM_CLIENTS=5
NUM_ROUNDS=10
STRATEGY="FedAvg"  # or "FedProx"

# Create results directory
mkdir -p results

# Start the server in background
echo "Starting Flower server..."
python3 -c "
from src.server import run_federated_server
from src.data_loader import FederatedDataLoader
import logging

logging.basicConfig(level=logging.INFO)

# Load test data for evaluation
data_loader = FederatedDataLoader()
x_train, y_train, x_test, y_test = data_loader.load_mnist_data()
_, _, test_dataset = data_loader.get_centralized_data(x_train, y_train, x_test, y_test)

# Run server
run_federated_server(
    num_clients=$NUM_CLIENTS,
    num_rounds=$NUM_ROUNDS,
    strategy='$STRATEGY',
    test_dataset=test_dataset
)
" &
SERVER_PID=$!

# Wait for server to start
sleep 5

# Start multiple clients
echo "Starting $NUM_CLIENTS clients..."
for i in $(seq 0 $((NUM_CLIENTS - 1))); do
    python3 -c "
from src.client import create_client
from src.data_loader import FederatedDataLoader
import flwr as fl

# Load and prepare data
data_loader = FederatedDataLoader(num_clients=$NUM_CLIENTS)
x_train, y_train, x_test, y_test = data_loader.load_mnist_data()
partitions = data_loader.create_client_partitions(x_train, y_train)
datasets = data_loader.create_tf_datasets(partitions)

# Create and start client
client = create_client($i, datasets[$i], None, '$STRATEGY')
fl.client.start_client(
    server_address='localhost:8080',
    client=client.to_client()
)
" &
done

# Wait for all processes
wait $SERVER_PID

echo "Federated learning experiment completed!"