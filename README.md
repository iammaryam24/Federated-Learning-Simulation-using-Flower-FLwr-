# Federated Learning Simulation using Flower

## Overview
This project implements a federated learning simulation using the Flower framework (FLwr) to compare distributed machine learning approaches against traditional centralized training.

## Features
- Federated learning with multiple aggregation strategies (FedAvg, FedProx)
- CPU-based training using TensorFlow/Scikit-learn
- MNIST dataset partitioning across simulated clients
- Comprehensive accuracy/speedup analysis
- Visualization and comparison plots

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd federated-learning-flower

# Install dependencies
pip install -r requirements.txt