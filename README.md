**Federated Learning with Differential Privacy Strategies**

This repository implements and compares five different Differential Privacy (DP) strategies in Federated Learning (FL) settings using the MNIST dataset.

🎯 **Strategies Implemented**
1. Static DP
Location: server_static_dp.py, client_static_dp.py

Port: 8080

Description: Fixed ε allocation for all clients at every round

Features:

Simple uniform ε allocation

Fixed privacy budget per round

Baseline implementation

Predictable privacy consumption

2. Adaptive DP Scheduling
Location: server_adaptive_dp.py, client_adaptive_dp.py

Port: 8081

Description: Pre-defined ε schedule based on training progress

Based on: Yuan et al. [2]

Features:

Exponential/cosine decay scheduling

Progress-aware privacy allocation

Global schedule across all clients

More efficient than static allocation

3. Personalized DP
Location: server_personalized_dp.py, client_personalized_dp.py

Port: 8082

Description: Client-specific ε allocation based on characteristics

Based on: Tan et al. [3]

Features:

Dataset-size aware allocation

Sensitivity-based protection levels

Fairness considerations

Client heterogeneity adaptation

4. DP-SGD (Standard)
Location: server_dp_sgd.py, client_dp_sgd.py

Port: 8083

Description: Canonical DP-SGD with per-sample gradient processing

Based on: Abadi et al. [4]

Features:

Per-sample gradient clipping

Gaussian noise injection

Strong theoretical guarantees

De facto standard implementation

5. Agentic DP (Our Proposal)
Location: server_agentic_dp.py, client_agentic_dp.py

Port: 8084

Description: Decentralized, autonomous agent-based privacy allocation

Features:

Autonomous per-client privacy agents

Dynamic ε allocation based on gradient importance

Context-aware budget management

Performance-adaptive privacy

Urgency-based allocation decisions

🚀 Quick Start
Prerequisites
bash
pip install torch torchvision flwr numpy
Running the Experiments
Start all servers (in separate terminals):

bash
# Static DP
python server_static_dp.py

# Adaptive DP Scheduling
python server_adaptive_dp.py

# Personalized DP
python server_personalized_dp.py

# DP-SGD
python server_dp_sgd.py

# Agentic DP
python server_agentic_dp.py
Run clients for each strategy (4 clients each):

bash
# Example for Static DP
python client_static_dp.py 0
python client_static_dp.py 1
python client_static_dp.py 2
python client_static_dp.py 3

# Repeat for other strategies with respective client files
Automated Execution
Use the provided script to run all experiments sequentially:

bash
chmod +x run_comparative_study.sh
./run_comparative_study.sh
📊 Metrics Collected
Each strategy generates comprehensive metrics:

Accuracy: Final and per-round test accuracy

Privacy Consumption: Total ε used, per-client allocation

Fairness: Distribution fairness of privacy budget (Gini coefficient)

Efficiency: Accuracy improvement per unit of privacy budget

Performance Metrics: Loss progression, training convergence

Budget Utilization: Remaining privacy budget analysis

🏗️ Architecture
Server Components
Strategy Handler: Implements FL strategy with DP

Privacy Allocator: Strategy-specific ε allocation logic

Metrics Collector: Performance and privacy metrics tracking

Model Aggregator: Federated averaging with DP guarantees

Client Components
Local Trainer: DP-enabled training implementation

Privacy Engine: Gradient clipping and noise addition

Data Handler: MNIST dataset management and partitioning

Communication Manager: Flower client implementation

🔧 Configuration
Key Parameters
Total Rounds: 15 communication rounds

Target ε: 10.0 (for strategies with fixed targets)

Target δ: 1e-5 (fixed for all strategies)

Clip Norm: 1.5-2.0 (strategy-dependent)

Learning Rate: 0.001 (consistent across strategies)

Batch Size: 64
