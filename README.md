# 🌸 Federated Learning Simulation using Flower (FLwr)

[![Python](https://img.shields.io/badge/Python-3.13.13-blue)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/Flower-1.29.0-pink)](https://flower.ai/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.8.0-orange)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Completed-success)]()

> **A comprehensive federated learning simulation implementing FedAvg strategy with Scikit-learn MLPClassifier, demonstrating distributed machine learning across multiple clients without sharing raw data.**

---

## 📊 Project Overview

This project implements a **Federated Learning (FL) simulation** using the **Flower (FLwr) framework** as part of the **Parallel & Distributed Computing** course (Semester 6, Spring 2026).

The system demonstrates distributed machine learning where:
- 🤖 Model training happens locally on each client
- 🔒 Raw data **never leaves** the client devices
- 🔄 Only model weights/parameters are shared
- 📈 Accuracy improves collaboratively across rounds

---

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| 🏗️ **Flower Framework** | Production-ready federated learning infrastructure |
| 📊 **FedAvg Strategy** | Federated Averaging for model weight aggregation |
| 🧠 **MLP Neural Network** | Scikit-learn MLPClassifier (64→32 neurons) |
| 👥 **Multi-Client** | 2 simulated clients with local data partitions |
| 📈 **Accuracy Tracking** | Real-time metrics per communication round |
| 🔒 **Privacy-Preserving** | Raw data stays local, only weights shared |
| 📉 **Visualization** | Comprehensive plots and analysis |
| 📚 **Documentation** | Complete project report with screenshots |

---

## 📈 Results Summary

| Metric | Initial | Final (Round 5) | Improvement |
|--------|---------|-----------------|-------------|
| **Server Test Accuracy** | 13.3% | **42.8%** | +29.5% |
| **Client 0 Accuracy** | 18.1% | **43.8%** | +25.7% |
| **Client 1 Accuracy** | 9.0% | **38.2%** | +29.2% |
| **Total Training Time** | - | **11.68 seconds** | 5 rounds |
| **Data Privacy** | ✅ | ✅ | 100% preserved |

---

## 🏗️ System Architecture
┌──────────────────────────────────────────────────────────┐
│ FLOWER SERVER │
│ │
│ ┌─────────────────────────────────────────────────┐ │
│ │ FedAvg Strategy │ │
│ │ • Aggregates client model weights │ │
│ │ • Evaluates global model on test data │ │
│ │ • Manages 5 communication rounds │ │
│ └─────────────────────────────────────────────────┘ │
│ │
│ Port: 0.0.0.0:8080 │
└──────────────────────────────────────────────────────────┘
▲ ▲
│ │
gRPC Protocol (Protobuf)
│ │
▼ ▼
┌──────────────────────┐ ┌──────────────────────┐
│ CLIENT 0 │ │ CLIENT 1 │
│ │ │ │
│ ┌────────────────┐ │ │ ┌────────────────┐ │
│ │ Local Data │ │ │ │ Local Data │ │
│ │ Train: 575 │ │ │ │ Train: 574 │ │
│ │ Val: 144 │ │ │ │ Val: 144 │ │
│ └────────────────┘ │ │ └────────────────┘ │
│ │ │ │
│ ┌────────────────┐ │ │ ┌────────────────┐ │
│ │ MLP Model │ │ │ │ MLP Model │ │
│ │ 64→32 neurons │ │ │ │ 64→32 neurons │ │
│ │ 6,570 params │ │ │ │ 6,570 params │ │
│ └────────────────┘ │ │ └────────────────┘ │
│ │ │ │
│ 🔒 DATA NEVER LEAVES│ │ 🔒 DATA NEVER LEAVES│
└──────────────────────┘ └──────────────────────┘


## 📁 Project Structure
federated-learning-flower/
│
├── 📂 src/ # Source code
│ ├── 📄 init.py # Package initialization
│ ├── 📄 model.py # MLP neural network model
│ ├── 📄 data_loader.py # Data loading & client partitioning
│ ├── 📄 client.py # Flower client implementation
│ ├── 📄 server.py # Flower server with FedAvg
│ ├── 📄 centralized.py # Centralized training baseline
│ ├── 📄 utils.py # Utility functions
│ └── 📄 plot_results.py # Visualization & comparison tools
│
├── 📂 results/ # Generated outputs (plots, metrics)
├── 📂 tests/ # Unit tests
├── 📂 documentation/ # Project documentation
├── 📄 requirements.txt # Python dependencies
├── 📄 README.md # Project documentation (this file)
└── 📄 .gitignore # Git ignore rules


## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.8+** (developed on Python 3.13.13)
- **pip** package manager
- **Git** (optional, for cloning)

### Step 1: Clone Repository

git clone https://github.com/iammaryam24/Federated-Learning-Simulation-using-Flower-FLwr-.git
cd Federated-Learning-Simulation-using-Flower-FLwr-
Step 2: Set Up Virtual Environment

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
Step 3: Install Dependencies

pip install -r requirements.txt
Step 4: Run the System
Open 3 separate terminals:

Terminal 1 - Start Server:

python src/server.py

Terminal 2 - Start Client 0:

python src/client.py0

Terminal 3 - Start Client 1:

python src/client.py 1

📊 Detailed Results:

Accuracy Progression Per Round
Round	Server Accuracy	Client 0 Accuracy	Client 1 Accuracy
0 (Init)	13.33%	-	-
1	18.33%	18.06%	9.03%
2	21.67%	22.22%	13.89%
3	25.28%	32.64%	19.44%
4	35.56%	38.19%	30.56%
5	42.78%	43.75%	38.19%
Loss Convergence
Round	Distributed Loss	Centralized Loss
1	0.8646	0.9301
2	0.8133	0.9083
3	0.7799	0.8361
4	0.6556	0.7444
5	0.5903	0.7222

🔧 Technologies Used:

Technology	Version	Purpose
Python	3.13.13	Primary programming language
Flower (FLwr)	1.29.0	Federated Learning framework
Scikit-learn	1.8.0	ML backend (MLPClassifier)
NumPy	2.4.4	Numerical computations
Pandas	3.0.2	Data manipulation
Matplotlib	3.10.9	Data visualization
Seaborn	0.13.2	Statistical plots
SciPy	1.17.1	Scientific computing
gRPC	1.80.0	Communication protocol

📊 Model Architecture

┌─────────────────────────────────────────┐
│          MLP NEURAL NETWORK              │
│                                          │
│   Input Layer:     64 neurons (8×8 px)  │
│         ↓                                │
│   Hidden Layer 1:  64 neurons + ReLU    │
│         ↓                                │
│   Hidden Layer 2:  32 neurons + ReLU    │
│         ↓                                │
│   Output Layer:    10 neurons + Softmax │
│                                          │
│   Total Parameters: 6,570               │
└─────────────────────────────────────────┘


📅 Project Timeline
Phase	Duration	Tasks
Week 1	Setup	Python, Flower, environment configuration
Week 2	Development	Model, data loader, client-server code
Week 3	Testing	Integration testing, debugging, fixes
Week 4	Documentation	Report, README, presentation, GitHub

⚠️ Challenges & Solutions
Challenge	Solution
TensorFlow AVX incompatibility	Switched to Scikit-learn MLPClassifier
Windows Device Guard blocking pip	Used python -m pip instead of pip
Pandas/Numpy version conflict	Used sklearn's built-in load_digits()
MLPClassifier warm_start issue	Pre-initialized model with all 10 classes
Path with spaces in Windows	Wrapped paths in quotes
📚 References
McMahan, B., et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.

Li, T., et al. (2020). "Federated Optimization in Heterogeneous Networks." MLSys 2020.

Flower Framework Documentation: https://flower.ai/docs/

Scikit-learn Documentation: https://scikit-learn.org/

Kairouz, P., et al. (2021). "Advances and Open Problems in Federated Learning." Foundations and Trends in ML.

