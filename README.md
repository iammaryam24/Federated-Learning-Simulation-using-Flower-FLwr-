🌸 Federated Learning Simulation using Flower (FLwr)
<div align="center">
https://img.shields.io/badge/Python-3.13.13-blue.svg?style=for-the-badge&logo=python&logoColor=white
https://img.shields.io/badge/Flower-1.29.0-pink.svg?style=for-the-badge
https://img.shields.io/badge/Scikit--learn-1.8.0-orange.svg?style=for-the-badge&logo=scikit-learn&logoColor=white
https://img.shields.io/badge/NumPy-2.4.4-013243.svg?style=for-the-badge&logo=numpy&logoColor=white
https://img.shields.io/badge/Status-Completed-success.svg?style=for-the-badge


A Comprehensive Federated Learning Simulation Demonstrating Distributed Machine Learning with Privacy-Preserving Architecture

Parallel & Distributed Computing | Semester 6 | Spring 2026

</div>
📑 Table of Contents
📊 Project Overview

🎯 Key Features

🏗️ System Architecture

📁 Project Structure

📈 Results

🔧 Technologies Used

🚀 Quick Start

📖 Usage Guide

💻 Code Explanation

📊 Model Architecture

🔄 Communication Flow

⚠️ Challenges & Solutions

📊 Visualization

👥 Team Fireweed

📅 Project Timeline

📚 References

📝 Academic Context

🙏 Acknowledgments

📞 Contact

📊 Project Overview
What is Federated Learning?
Federated Learning (FL) is a revolutionary machine learning paradigm where models are trained across multiple decentralized devices holding local data, without exchanging the raw data. Only model parameters (weights) are shared, ensuring data privacy.

text
┌─────────────────────────────────────────────────────────────────┐
│                    TRADITIONAL MACHINE LEARNING                  │
│                                                                  │
│   [Data Source A] ──┐                                           │
│   [Data Source B] ──┼──→ [Central Server] ──→ [Trained Model]  │
│   [Data Source C] ──┘         ↑                                  │
│                          ALL DATA COLLECTED                      │
│                          ⚠️ Privacy Risk                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    FEDERATED LEARNING                            │
│                                                                  │
│   [Client A + Data] ──→ [Local Model Update] ──┐               │
│   [Client B + Data] ──→ [Local Model Update] ──┤               │
│   [Client C + Data] ──→ [Local Model Update] ──┤               │
│                                                  ▼               │
│                                         [Server Aggregation]     │
│                                                  │               │
│                                          [Global Model]          │
│                                                                  │
│                    ✅ Data NEVER leaves clients                  │
│                    ✅ Only model weights shared                  │
└─────────────────────────────────────────────────────────────────┘
Project Description
This project implements a complete federated learning simulation using the Flower (FLwr) framework. It demonstrates distributed training across 2 simulated clients, each holding local data partitions from the Scikit-learn Digits dataset. The server employs the FedAvg (Federated Averaging) strategy to aggregate client model updates across 5 communication rounds.


Objectives Achieved
#	Objective	Status
1	Install Flower framework & set up federated learning environment	✅
2	Implement distributed training across simulated clients	✅
3	Add FedAvg aggregation strategy	✅
4	Compare federated vs. centralized training performance	✅
5	Generate accuracy/speedup analysis and comparison plots	✅
6	Create GitHub repository with complete documentation	✅
7	Prepare project report and presentation	✅

🎯 Key Features

Feature	Description
🌸 Flower Framework	Production-ready federated learning infrastructure with gRPC communication
📊 FedAvg Strategy	Federated Averaging algorithm for weighted model aggregation
🧠 MLP Neural Network	Multi-Layer Perceptron with 2 hidden layers (64→32 neurons)
👥 Multi-Client Training	2 simulated clients with independent local data partitions
🔒 Privacy-Preserving	Raw data remains on clients; only model weights are shared
📈 Real-Time Tracking	Accuracy and loss metrics tracked every communication round
📉 Visualization	Comprehensive plots comparing federated vs. centralized training
🔄 Centralized Baseline	Traditional ML training for performance comparison
📚 Documentation	Complete project report, README, and code comments
🐍 Single File Option	Full working code available in one Python file
🏗️ System Architecture
High-Level Architecture
text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                              │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         FLOWER SERVER (server.py)                        │ │
│  │                                                                          │ │
│  │  ┌─────────────────────────┐    ┌──────────────────────────────────┐    │ │
│  │  │   Server Configuration  │    │        FedAvg Strategy            │    │ │
│  │  │   • Address: 0.0.0.0   │    │                                   │    │ │
│  │  │   • Port: 8080         │    │  • fraction_fit: 1.0              │    │ │
│  │  │   • Rounds: 5          │    │  • fraction_evaluate: 1.0         │    │ │
│  │  │   • Protocol: gRPC     │    │  • min_fit_clients: 2             │    │ │
│  │  └─────────────────────────┘    │  • min_available_clients: 2      │    │ │
│  │                                  │  • evaluate_fn: Custom          │    │ │
│  │  ┌─────────────────────────┐    │  • initial_parameters: Provided  │    │ │
│  │  │   Evaluation Engine     │    └──────────────────────────────────┘    │ │
│  │  │   • Model: MLP          │                                            │ │
│  │  │   • Test Set: 360       │    ┌──────────────────────────────────┐    │ │
│  │  │   • Frequency: Per round│    │    Client Manager                │    │ │
│  │  └─────────────────────────┘    │    • Track connected clients     │    │ │
│  │                                  │    • Sample for each round       │    │ │
│  └──────────────────────────────────┴──────────────────────────────────┘    │ │
│                                                                              │
│                               ▲           ▲                                   │
│                               │           │                                   │
│                          gRPC Protocol (Protobuf)                              │
│                               │           │                                   │
│                               ▼           ▼                                   │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                          │ │
│  │  ┌─────────────────────────────┐    ┌─────────────────────────────┐     │ │
│  │  │        CLIENT 0             │    │        CLIENT 1             │     │ │
│  │  │        (client.py)          │    │        (client.py)          │     │ │
│  │  │                             │    │                             │     │ │
│  │  │  ┌─────────────────────┐    │    │  ┌─────────────────────┐    │     │ │
│  │  │  │   Local Dataset     │    │    │  │   Local Dataset     │    │     │ │
│  │  │  │   • Train: 575      │    │    │  │   • Train: 574      │    │     │ │
│  │  │  │   • Val:   144      │    │    │  │   • Val:   144      │    │     │ │
│  │  │  │   • Features: 64    │    │    │  │   • Features: 64    │    │     │ │
│  │  │  │   • Classes: 10     │    │    │  │   • Classes: 10     │    │     │ │
│  │  │  └─────────────────────┘    │    │  └─────────────────────┘    │     │ │
│  │  │                             │    │                             │     │ │
│  │  │  ┌─────────────────────┐    │    │  ┌─────────────────────┐    │     │ │
│  │  │  │   MLP Model         │    │    │  │   MLP Model         │    │     │ │
│  │  │  │   • Hidden: 64→32   │    │    │  │   • Hidden: 64→32   │    │     │ │
│  │  │  │   • Activation: ReLU│    │    │  │   • Activation: ReLU│    │     │ │
│  │  │  │   • Output: 10      │    │    │  │   • Output: 10      │    │     │ │
│  │  │  │   • Params: 6,570   │    │    │  │   • Params: 6,570   │    │     │ │
│  │  │  └─────────────────────┘    │    │  └─────────────────────┘    │     │ │
│  │  │                             │    │                             │     │ │
│  │  │  Operations:                │    │  Operations:                │     │ │
│  │  │  • fit()                    │    │  • fit()                    │     │ │
│  │  │  • evaluate()               │    │  • evaluate()               │     │ │
│  │  │  • get_parameters()         │    │  • get_parameters()         │     │ │
│  │  │                             │    │                             │     │ │
│  │  │  🔒 DATA NEVER LEAVES       │    │  🔒 DATA NEVER LEAVES       │     │ │
│  │  └─────────────────────────────┘    └─────────────────────────────┘     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
Communication Flow (Single Round)
text
ROUND X: Federated Learning Communication

1. SERVER ────────────→ CLIENT 0 (Send Global Model Weights)
2. SERVER ────────────→ CLIENT 1 (Send Global Model Weights)
                         
3.                     CLIENT 0: fit(X_train, y_train)  [Local training]
4.                     CLIENT 1: fit(X_train, y_train)  [Local training]
                         
5. SERVER ←──────────── CLIENT 0 (Return Updated Weights)
6. SERVER ←──────────── CLIENT 1 (Return Updated Weights)
                         
7. SERVER: θ_new = (n0/N) * θ0 + (n1/N) * θ1  [FedAvg Aggregation]
                         
8.                     CLIENT 0: evaluate(X_val, y_val)  [Local eval]
9.                     CLIENT 1: evaluate(X_val, y_val)  [Local eval]
                         
10. SERVER ←─────────── CLIENT 0 (Return Metrics)
11. SERVER ←─────────── CLIENT 1 (Return Metrics)
                         
12. SERVER: Test global model on test set
                         
13. Move to ROUND X+1

📁 Project Structure
federated-learning-flower/
│
├── 📂 src/                              # Source code directory
│   ├── 📄 __init__.py                   # Package initialization (301 bytes)
│   ├── 📄 model.py                      # MLP neural network model (3,501 bytes)
│   ├── 📄 data_loader.py                # Data loading & partitioning (6,385 bytes)
│   ├── 📄 client.py                     # Flower client implementation (8,878 bytes)
│   ├── 📄 server.py                     # Flower server with FedAvg (9,351 bytes)
│   ├── 📄 centralized.py                # Centralized training baseline (4,963 bytes)
│   ├── 📄 utils.py                      # Utility functions (3,433 bytes)
│   └── 📄 plot_results.py               # Visualization & comparison (10,183 bytes)
│
├── 📂 results/                          # Generated outputs
│   ├── 📄 .gitkeep                      # Keep directory in Git
│   ├── 📄 federated_learning_results.png # Accuracy comparison plot
│   └── 📄 results_summary.txt           # Training results summary
│
├── 📂 tests/                            # Unit tests directory
│   └── 📄 __init__.py
│
├── 📂 documentation/                    # Project documentation
│   └── 📄 project_report.md             # Full project report
│
├── 📄 federated_learning.py             # Single-file complete implementation
├── 📄 requirements.txt                  # Python dependencies list
├── 📄 README.md                         # This documentation file
├── 📄 LICENSE                           # MIT License
├── 📄 .gitignore                        # Git ignore rules
├── 📄 run_federated.sh                  # Linux/Mac run script
└── 📄 run_centralized.sh                # Linux/Mac run script
📈 Results

Server-Side Results (All Rounds)

Round	Test Accuracy	Loss (Distributed)	Loss (Centralized)	Improvement
0 (Init)	13.33%	-	-	-
1	18.33%	0.8646	0.9301	+5.00%
2	21.67%	0.8133	0.9083	+3.34%
3	25.28%	0.7799	0.8361	+3.61%
4	35.56%	0.6556	0.7444	+10.28%
5	42.78%	0.5903	0.7222	+7.22%

Client-Side Results

Round	Client 0 Accuracy	Client 1 Accuracy
1	18.06%	9.03%
2	22.22%	13.89%
3	32.64%	19.44%
4	38.19%	30.56%
5	43.75%	38.19%

Performance Summary

Metric	Value
Initial Server Accuracy	13.33%
Final Server Accuracy	42.78%
Total Improvement	+29.45%
Best Client Accuracy	43.75% (Client 0, Round 5)
Total Training Time	11.68 seconds (5 rounds)
Data Privacy Preserved	✅ 100%
Centralized vs Federated Comparison
Metric	Federated (FedAvg)	Centralized
Training Approach	Distributed (2 clients)	Single machine
Data Access	Local only	Full dataset
Communication Rounds	5	N/A
Local Epochs per Round	1	50
Final Test Accuracy	42.78%	93.89%
Training Time	11.68s	~5s
Privacy	✅ Preserved	❌ All data exposed
Scalability	✅ High (add clients)	❌ Limited
🔧 Technologies Used
Technology	Version	Purpose	Logo
Python	3.13.13	Primary programming language	🐍
Flower (FLwr)	1.29.0	Federated Learning framework	🌸
Scikit-learn	1.8.0	ML backend (MLPClassifier)	📊
NumPy	2.4.4	Numerical computations	🔢
Pandas	3.0.2	Data manipulation	🐼
Matplotlib	3.10.9	Data visualization	📈
Seaborn	0.13.2	Statistical plots	📉
SciPy	1.17.1	Scientific computing	🔬
gRPC	1.80.0	Communication protocol	🔄
Protobuf	6.33.6	Data serialization	📦
Visual Studio Code	Latest	Development environment	💻

🚀 Quick Start

Prerequisites
Requirement	Version
Python	3.8 or higher
pip	Latest
Git	Optional
OS	Windows / Linux / macOS
Installation (2 Options)
Option 1: Clone Repository
bash
# Clone the repository
git clone https://github.com/iammaryam24/Federated-Learning-Simulation-using-Flower-FLwr-.git
cd Federated-Learning-Simulation-using-Flower-FLwr-

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
Option 2: Single File Download
bash
# Download just the single file
# Save as federated_learning.py

# Install dependencies
pip install flwr scikit-learn numpy matplotlib seaborn pandas

# Run
python federated_learning.py
Run the Project
Method 1: Multi-Terminal (Original)
Terminal 1 - Start Server:

bash
venv\Scripts\activate
python src/server.py
Terminal 2 - Start Client 0:

bash
venv\Scripts\activate
python src/client.py 0
Terminal 3 - Start Client 1:

bash
venv\Scripts\activate
python src/client.py 1
Method 2: Single Command (Auto)
bash
# Single file runs everything automatically
python federated_learning.py

📖 Usage Guide
Expected Output

╔══════════════════════════════════════════════════════════════╗
║  FEDERATED LEARNING SIMULATION USING FLOWER (FLwr)          ║
║  Team Fireweed - Parallel & Distributed Computing           ║
║  2 Clients | 5 Rounds | FedAvg Strategy                     ║
╚══════════════════════════════════════════════════════════════╝

[DATA] Loading Digits dataset...
[DATA] Train: (1437, 64), Test: (360, 64)
[DATA] Client 0: Train=575, Val=144, Classes=[0 1 2 3 4 5 6 7 8 9]
[DATA] Client 1: Train=574, Val=144, Classes=[0 1 2 3 4 5 6 7 8 9]

[SYSTEM] Starting Flower server...
==============================================================
  FLOWER FEDERATED LEARNING SERVER
==============================================================
  Address: 127.0.0.1:8080
  Rounds: 5
  Clients: 2
  Strategy: FedAvg
==============================================================

[SYSTEM] Starting clients...
[CLIENT 0] Connecting to server at 127.0.0.1:8080...
[CLIENT 1] Connecting to server at 127.0.0.1:8080...

==============================================================
  ROUND 1 COMPLETED
  Global Test Accuracy: 0.1833 (18.3%)
==============================================================

...

==============================================================
  ROUND 5 COMPLETED
  Global Test Accuracy: 0.4278 (42.8%)
==============================================================

==============================================================
  FINAL RESULTS SUMMARY
==============================================================

  📊 FEDERATED LEARNING (FedAvg):
     Final Accuracy: 42.8%
     Rounds: 5

  📈 CENTRALIZED TRAINING:
     Test Accuracy: 93.9%
     Training Time: 4.85s
==============================================================
  ✅ PROJECT COMPLETED SUCCESSFULLY!
==============================================================
💻 Code Explanation
Key Components
1. Model (src/model.py)
python
from sklearn.neural_network import MLPClassifier

def create_model():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),    # Two hidden layers
        activation='relu',              # ReLU activation
        solver='adam',                  # Adam optimizer
        max_iter=1,                     # 1 iteration per round
        warm_start=True,                # Continue from previous weights
        random_state=42                 # Reproducibility
    )
2. Client (src/client.py)
python
class FlowerClient(fl.client.NumPyClient):
    def fit(self, parameters, config):
        # Train on local data
        set_weights(self.model, parameters)
        self.model.fit(self.X_train, self.y_train)
        return get_weights(self.model), len(self.X_train), {}
    
    def evaluate(self, parameters, config):
        # Evaluate on local validation data
        set_weights(self.model, parameters)
        acc = self.model.score(self.X_val, self.y_val)
        return 1.0 - acc, len(self.X_val), {"accuracy": float(acc)}
3. Server (src/server.py)
python
strategy = FedAvg(
    fraction_fit=1.0,          # Use all clients
    fraction_evaluate=1.0,     # Evaluate all clients
    min_fit_clients=2,         # Minimum clients for training
    min_evaluate_clients=2,    # Minimum clients for evaluation
    min_available_clients=2,   # Minimum available clients
    evaluate_fn=evaluate_fn,   # Custom evaluation function
    initial_parameters=initial_params,
)

fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
)

📊 Model Architecture

┌─────────────────────────────────────────────────────────────┐
│                   MLP NEURAL NETWORK                         │
│                                                              │
│   INPUT LAYER                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  64 neurons (8×8 pixel values)                      │   │
│   └────────────────────────┬────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│   HIDDEN LAYER 1                                            │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  64 neurons                                         │   │
│   │  Activation: ReLU                                   │   │
│   │  Parameters: 64×64 + 64 = 4,160                     │   │
│   └────────────────────────┬────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│   HIDDEN LAYER 2                                            │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  32 neurons                                         │   │
│   │  Activation: ReLU                                   │   │
│   │  Parameters: 64×32 + 32 = 2,080                     │   │
│   └────────────────────────┬────────────────────────────┘   │
│                            │                                 │
│                            ▼                                 │
│   OUTPUT LAYER                                               │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  10 neurons (digits 0-9)                            │   │
│   │  Activation: Softmax                                │   │
│   │  Parameters: 32×10 + 10 = 330                       │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
│   TOTAL TRAINABLE PARAMETERS: 6,570                          │
└─────────────────────────────────────────────────────────────┘

🔄 Communication Flow
Federated Learning Round Lifecycle

┌─────────────────────────────────────────────────────────────────────────┐
│                        ROUND LIFECYCLE                                   │
│                                                                          │
│  STEP 1: Server Initialization                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Create initial global model                                     │ │
│  │  • Pre-train on 2 samples from each class (20 total)               │ │
│  │  • Extract weights as initial parameters                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 2: Client Selection                                                 │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Server samples clients (fraction_fit=1.0 → all clients)        │ │
│  │  • Sends global model weights to selected clients                  │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 3: Local Training (Each Client)                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Client receives global weights                                  │ │
│  │  • Sets local model weights = global weights (warm_start)          │ │
│  │  • Trains on local data: fit(X_train, y_train)                     │ │
│  │  • Extracts updated weights                                        │ │
│  │  • Returns weights to server                                       │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 4: Aggregation (FedAvg)                                           │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Server collects all client weight updates                       │ │
│  │  • Computes weighted average: θ_new = Σ(n_k/N) * θ_k               │ │
│  │  • Updates global model                                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 5: Local Evaluation (Each Client)                                  │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Client evaluates on local validation data                       │ │
│  │  • Returns accuracy/loss metrics to server                         │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 6: Centralized Evaluation                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  • Server evaluates global model on test set (360 samples)         │ │
│  │  • Records accuracy and loss metrics                               │ │
│  │  • Prints round summary                                            │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  STEP 7: Repeat (5 rounds total)                                        │
└─────────────────────────────────────────────────────────────────────────┘

⚠️ Challenges & Solutions

#	Challenge	Error Message	Root Cause	Solution
1	TensorFlow AVX	DLL load failed - INITIALIZATION FAILED (0x45A)	CPU lacks AVX/AVX2 instructions	Switched to Scikit-learn MLPClassifier
2	Device Guard	pip.exe was blocked by Device Guard policy	University laptop security	Used python -m pip instead of pip
3	Pandas/Numpy Conflict	AttributeError: partially initialized module 'pandas'	Incompatible pandas 3.0.2 + numpy 2.4.4	Used sklearn.datasets.load_digits() directly
4	warm_start Classes	ValueError: warm_start can only be used where y has same classes	Initial fit on 6 classes, later saw all 10	Pre-initialize with 2 samples from ALL classes
5	Path with Spaces	'DISTRIBUTED' is not recognized	Windows CMD doesn't auto-quote paths	Wrapped paths in double quotes
6	Flower Deprecation Warning	start_client() is deprecated	Newer Flower version	Accepted (functionality unaffected)

📊 Visualization
The project generates the following plots:

1. Accuracy per Round (Federated Learning)
X-axis: Communication Round (0-5)

Y-axis: Accuracy (%)

Shows consistent improvement across rounds

2. Federated vs Centralized Comparison
Bar chart comparing final accuracies

Side-by-side comparison of training approaches

Generated Files:

results/
├── federated_learning_results.png    # Main comparison plot
└── results_summary.txt               # Numerical results


📅 Project Timeline
Week	Phase	Activities	Deliverables
Week 1	Planning & Setup	Research FL frameworks, install Python, set up environment	Environment ready, dependencies installed
Week 2	Core Development	Implement model, data loader, client, server code	Working FL system prototype
Week 3	Testing & Debugging	Integration testing, bug fixes, performance tuning	Stable working system
Week 4	Documentation	Report writing, README, presentation, GitHub upload	Complete project submission

📚 References
McMahan, B., Moore, E., Ramage, D., Hampson, S., & Agüera y Arcas, B. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." Proceedings of the 20th International Conference on Artificial Intelligence and Statistics (AISTATS 2017).

Li, T., Sahu, A. K., Zaheer, M., Sanjabi, M., Talwalkar, A., & Smith, V. (2020). "Federated Optimization in Heterogeneous Networks." Proceedings of Machine Learning and Systems (MLSys 2020).

Kairouz, P., et al. (2021). "Advances and Open Problems in Federated Learning." Foundations and Trends in Machine Learning, 14(1-2), 1-210.

Flower Framework Documentation. https://flower.ai/docs/

Scikit-learn Documentation. https://scikit-learn.org/stable/

Python Documentation. https://docs.python.org/3/

📝 Academic Context:

Field	Details
Course	Parallel & Distributed Computing
Semester	6th Semester (Spring 2026)
Department	Computer Science
Project Type	Semester Project
Submission Date	June 2, 2026

🙏 Acknowledgments:

Flower (FLwr) Development Team - For creating an excellent federated learning framework

Scikit-learn Community - For robust machine learning tools

Python Community - For the comprehensive ecosystem

Course Instructor - For guidance and support throughout the project

Department of Computer Science - For providing necessary resources

📞 Contact
Contact	Details
GitHub	iammaryam24
Repository	Federated-Learning-Simulation-using-Flower-FLwr-
Issues	Create Issue
Pull Requests	Create PR
⭐ Support
If you find this project helpful, please consider:

⭐ Starring the repository on GitHub

🔄 Forking for your own use

📢 Sharing with others interested in federated learning

🐛 Reporting issues or suggesting improvements


<div align="center">
text
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     🌸 Federated Learning Simulation using Flower (FLwr)     ║
║                                                              ║
║     Made with ❤️ by Team Fireweed                            ║
║                                                              ║
║     Zainab (089) | Malaka (093) | Maryam (057) | Fireweed (061) ║
║                                                              ║
║     Parallel & Distributed Computing                         ║
║     Semester 6 | Spring 2026                                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
</div>
<p align="center"> <sub>© 2026 Team Fireweed. All Rights Reserved.</sub><br> <sub>This project is part of the Parallel & Distributed Computing course.</sub> </p> ```
