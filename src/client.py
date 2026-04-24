"""Flower Client"""
import flwr as fl
import sys
import numpy as np
from model import create_model, get_weights, set_weights
from data_loader import load_data, split_for_clients

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, X_train, y_train, X_val, y_val, cid):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cid = cid
        self.trained = False

    def get_parameters(self, config):
        return get_weights(self.model)

    def fit(self, parameters, config):
        set_weights(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        self.trained = True
        print(f"[Client {self.cid}] Training done")
        return get_weights(self.model), len(self.X_train), {}

    def evaluate(self, parameters, config):
        set_weights(self.model, parameters)
        acc = self.model.score(self.X_val, self.y_val)
        print(f"[Client {self.cid}] Accuracy: {acc:.4f}")
        return 1.0 - acc, len(self.X_val), {"accuracy": float(acc)}

# Initialize model with ALL classes before starting
def init_model_with_all_classes():
    X, y, _, _ = load_data()
    model = create_model()
    # Fit on samples from ALL 10 classes
    indices = []
    for c in range(10):
        idx = np.where(y == c)[0][:2]
        indices.extend(idx)
    model.fit(X[indices], y[indices])
    return model

if __name__ == "__main__":
    cid = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Client {cid} starting...")
    
    X_train, y_train, _, _ = load_data()
    partitions = split_for_clients(X_train, y_train, num_clients=2)
    X_tr, y_tr, X_val, y_val = partitions[cid]
    
    # Initialize model with all classes
    model = init_model_with_all_classes()
    
    client = FlowerClient(model, X_tr, y_tr, X_val, y_val, cid)
    print(f"Client {cid} connecting to server...")
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)