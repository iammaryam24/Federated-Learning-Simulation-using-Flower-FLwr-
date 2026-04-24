#!/bin/bash

echo "=========================================="
echo "Centralized Neural Network Training"
echo "=========================================="

# Create results directory
mkdir -p results

# Run centralized training
python3 -c "
from src.centralized import run_centralized_experiment
from src.plot_results import ResultPlotter
import json

# Run experiment
results = run_centralized_experiment(epochs=30, batch_size=64)

# Plot results
plotter = ResultPlotter()
plotter.plot_loss_curves(
    federated_history=None,
    centralized_history=results['history'],
    save=True,
    show=False
)

print('Centralized training completed!')
print(f'Final accuracy: {results[\"final_accuracy\"]:.4f}')
print(f'Training time: {results[\"training_time\"]:.2f}s')
"