"""MLP model using sklearn"""
from sklearn.neural_network import MLPClassifier
import numpy as np

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        max_iter=1,
        warm_start=True,
        random_state=42
    )

def get_weights(model):
    weights = []
    for coef in model.coefs_:
        weights.append(coef.copy())
    for intercept in model.intercepts_:
        weights.append(intercept.copy())
    return weights

def set_weights(model, weights):
    n = len(model.coefs_)
    model.coefs_ = [w.copy() for w in weights[:n]]
    model.intercepts_ = [w.copy() for w in weights[n:]]