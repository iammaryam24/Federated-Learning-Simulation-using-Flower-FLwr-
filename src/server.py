"""Flower Server"""
import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np
from model import create_model, set_weights
from data_loader import load_data

def get_initial_params():
    """Initialize model with all 10 classes"""
    X, y, _, _ = load_data()
    model = create_model()
    indices = []
    for c in range(10):
        idx = np.where(y == c)[0][:2]
        indices.extend(idx)
    model.fit(X[indices], y[indices])
    
    from model import get_weights
    params = get_weights(model)
    return fl.common.ndarrays_to_parameters(params)

def main():
    print("Loading data...")
    _, _, X_test, y_test = load_data()
    
    eval_model = create_model()
    # Init with all classes
    indices = []
    for c in range(10):
        idx = np.where(y_test == c)[0]
        if len(idx) > 0:
            indices.append(idx[0])
    eval_model.fit(X_test[indices], y_test[indices])
    
    def evaluate_fn(server_round, parameters, config):
        set_weights(eval_model, parameters)
        acc = eval_model.score(X_test, y_test)
        print(f"\n=== Round {server_round}: Test Accuracy = {acc:.4f} ===\n")
        return 1.0 - acc, {"accuracy": float(acc)}
    
    initial_params = get_initial_params()
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        evaluate_fn=evaluate_fn,
        initial_parameters=initial_params,
    )
    
    print("\n" + "="*50)
    print("FLOWER SERVER STARTING on 0.0.0.0:8080")
    print("="*50 + "\n")
    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()