"""Data loading using sklearn digits dataset"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

def load_data():
    """Load sklearn digits dataset"""
    print("Loading Digits dataset...")
    digits = load_digits()
    X = digits.data.astype('float32') / 16.0
    y = digits.target.astype('int32')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def split_for_clients(X, y, num_clients=2):
    """Split data evenly with all classes in each client"""
    # Sort by labels to ensure each client gets all classes
    idx = np.argsort(y)
    X, y = X[idx], y[idx]
    
    partitions = []
    for i in range(num_clients):
        # Take every num_clients-th sample to distribute classes evenly
        X_part = X[i::num_clients]
        y_part = y[i::num_clients]
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_part, y_part, test_size=0.2, random_state=42
        )
        partitions.append((X_tr, y_tr, X_val, y_val))
        print(f"Client {i}: Train={len(X_tr)}, Val={len(X_val)}, Classes={np.unique(y_tr)}")
    return partitions