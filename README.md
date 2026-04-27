# Federated Learning Simulation using Flower (FLwr)

![Python](https://img.shields.io/badge/Python-3.13.13-blue)
![Flower](https://img.shields.io/badge/Flower-1.29.0-pink)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

**A Comprehensive Federated Learning Simulation Demonstrating Distributed Machine Learning with Privacy-Preserving Architecture**

*Parallel & Distributed Computing | Semester 6 | Spring 2026*

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Code Explanation](#code-explanation)
- [Challenges & Solutions](#challenges--solutions)
- [Team Fireweed](#team-fireweed)
- [References](#references)
- [Academic Context](#academic-context)

---

## Project Overview

### What is Federated Learning?

Federated Learning (FL) is a machine learning technique where models are trained across multiple decentralized devices holding local data, **without exchanging the raw data**. Only model parameters (weights) are shared, ensuring data privacy.

**Traditional ML:** Data is collected on a central server for training.

**Federated Learning:** Model goes to the data; data never leaves the client.

### Project Description

This project implements a federated learning simulation using the **Flower (FLwr) framework**. It demonstrates distributed training across **2 simulated clients**, each holding local data partitions from the **Scikit-learn Digits dataset**. The server employs the **FedAvg (Federated Averaging)** strategy to aggregate client model updates across **5 communication rounds**.

### Objectives Achieved

| # | Objective | Status |
|---|-----------|--------|
| 1 | Install Flower framework & set up federated learning environment | Done |
| 2 | Implement distributed training across simulated clients | Done |
| 3 | Add FedAvg aggregation strategy | Done |
| 4 | Compare federated vs. centralized training performance | Done |
| 5 | Generate accuracy/speedup analysis and comparison plots | Done |
| 6 | Create GitHub repository with complete documentation | Done |
| 7 | Prepare project report and presentation | Done |

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Flower Framework** | Production-ready federated learning infrastructure with gRPC communication |
| **FedAvg Strategy** | Federated Averaging algorithm for weighted model aggregation |
| **MLP Neural Network** | Multi-Layer Perceptron with 2 hidden layers (64 and 32 neurons) |
| **Multi-Client Training** | 2 simulated clients with independent local data partitions |
| **Privacy-Preserving** | Raw data remains on clients; only model weights are shared |
| **Real-Time Tracking** | Accuracy and loss metrics tracked every communication round |
| **Visualization** | Comprehensive plots comparing federated vs. centralized training |
| **Centralized Baseline** | Traditional ML training for performance comparison |
| **Single File Option** | Full working code available in one Python file |

### Communication Flow (One Round)

1. Server sends global model weights to all clients
2. Clients train locally on their private data
3. Clients return updated weights to server
4. Server aggregates weights using FedAvg: `θ_new = (n0/N) * θ0 + (n1/N) * θ1`
5. Clients evaluate on local validation data
6. Server evaluates global model on test set
7. Move to next round

---

## Project Structure
federated-learning-flower/
├── src/

│ ├── init.py # Package initialization

│ ├── model.py # MLP neural network model

│ ├── data_loader.py # Data loading & partitioning

│ ├── client.py # Flower client implementation

│ ├── server.py # Flower server with FedAvg

│ ├── centralized.py # Centralized training baseline

│ ├── utils.py # Utility functions

│ └── plot_results.py # Visualization & comparison

├── results/ # Generated outputs

├── tests/ # Unit tests

├── documentation/ # Project documentation

├── federated_learning.py # Single-file implementation

├── requirements.txt # Dependencies

├── README.md # This file

└── .gitignore # Git ignore rules


---

## Results

### Server-Side Results

| Round | Test Accuracy | Loss |
|-------|---------------|------|
| 0 (Init) | 13.33% | - |
| 1 | 18.33% | 0.8646 |
| 2 | 21.67% | 0.8133 |
| 3 | 25.28% | 0.7799 |
| 4 | 35.56% | 0.6556 |
| 5 | **42.78%** | **0.5903** |

### Client-Side Results

| Round | Client 0 | Client 1 |
|-------|----------|----------|
| 1 | 18.06% | 9.03% |
| 2 | 22.22% | 13.89% |
| 3 | 32.64% | 19.44% |
| 4 | 38.19% | 30.56% |
| 5 |
giv me short & to the point
📄 README.md - Short & Clean
markdown
# 🌸 Federated Learning Simulation using Flower (FLwr)

![Python](https://img.shields.io/badge/Python-3.13.13-blue)
![Flower](https://img.shields.io/badge/Flower-1.29.0-pink)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Federated Learning simulation with FedAvg strategy, Scikit-learn MLP, and multi-client distributed training on Digits dataset.**

---

## 📊 Overview

This project implements a federated learning simulation using **Flower (FLwr)**, demonstrating distributed machine learning across 2 clients without sharing raw data. Only model weights are exchanged, preserving data privacy.

---

## 🎯 Features

- Federated Learning with Flower framework
- FedAvg aggregation strategy
- MLP neural network (64→32 neurons, 6,570 parameters)
- 2 clients with local data partitions
- 5 communication rounds
- Real-time accuracy tracking
- Centralized training baseline comparison

---

## 🏗️ Architecture
Server (FedAvg) → gRPC → Client 0 (575 samples) + Client 1 (574 samples)
↓
Global Model

text

---

## 📁 Project Structure
src/
├── model.py # MLP model

├── data_loader.py # Data handling

├── client.py # Flower client

├── server.py # Flower server

├── centralized.py # Baseline

└── plot_results.py # Visualization

federated_learning.py # Single-file version
requirements.txt # Dependencies

---

## 📈 Results

| Round | Accuracy |
|-------|----------|
| 0 | 13.3% |
| 1 | 18.3% |
| 2 | 21.7% |
| 3 | 25.3% |
| 4 | 35.6% |
| 5 | **42.8%** |

- **Training Time:** 11.68 seconds
- **Improvement:** +29.5%

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/iammaryam24/Federated-Learning-Simulation-using-Flower-FLwr-.git
cd Federated-Learning-Simulation-using-Flower-FLwr-

# Setup
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Run (3 terminals)
python src/server.py          # Terminal 1
python src/client.py 0        # Terminal 2
python src/client.py 1        # Terminal 3

# OR single command
python federated_learning.py
