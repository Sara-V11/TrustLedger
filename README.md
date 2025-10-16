# TrustLedger | AI-Powered Fraud & Risk Detection System

> **A Graph Neural Network (GNN)-based system for detecting fraudulent financial transactions.**

TrustLedger analyzes **relationships between accounts** using **edge-feature-aware graph learning** to identify suspicious transaction patterns that traditional methods often miss.  

---

## Table of Contents
- [ Overview](#-overview)
- [ Tech Stack](#-tech-stack)
- [ Dataset](#-dataset)
- [ How It Works](#-how-it-works)
- [ Model Architecture](#️-model-architecture)
- [ Training & Evaluation](#-training--evaluation)
- [ Visualizations](#-visualizations)
- [ Key Learnings](#-key-learnings)
- [ Future Enhancements](#-future-enhancements)
- [ Author](#-author)

---

##  Overview

Fraud detection systems in banking typically treat transactions as **independent events**, missing patterns of **coordinated fraud** among users.  
**TrustLedger** takes a **network-based approach**, representing transactions as a **graph** to reveal hidden relationships and detect fraud rings.

Each **node** represents an account, and each **edge** represents a transaction between accounts.  
By applying **Graph Neural Networks (GNNs)**, the model learns to detect suspicious accounts based on both **node features** and **edge features** (e.g., transaction amount and timing).

---

## Tech Stack

| Category | Technology |
|-----------|-------------|
| **Language** | Python |
| **Libraries** | PyTorch, PyTorch Geometric, Scikit-learn, Pandas, NetworkX |
| **Visualization** | Matplotlib |
| **Environment** | Google Colab / Jupyter Notebook |
| **Dataset Source** | [Kaggle: Synthetic Financial Transactions Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1) |

---

##  Dataset

The dataset consists of **6.3 million** financial transactions between customers.  
For faster experimentation, **50,000 transactions** were sampled.

### Key Columns:
| Column | Description |
|---------|--------------|
| `step` | Time step of the transaction |
| `type` | Transaction type (e.g., CASH_OUT, TRANSFER) |
| `amount` | Amount of money transferred |
| `nameOrig`, `nameDest` | Sender and receiver accounts |
| `isFraud` | 1 if fraudulent, 0 otherwise |

---

## How It Works

### **1️⃣ Data Preprocessing**
- Encoded sender (`nameOrig`) and receiver (`nameDest`) accounts as numeric IDs.  
- Normalized numerical features (`amount`, `step`).  
- Created additional **node-level features**:
  - Degree
  - In-degree
  - Out-degree
  - Total transaction amount

### ** Graph Construction**
Each transaction was modeled as an edge:
```python
edge_index: [2, 50000]
edge_attr:  [50000, 2]

Node features matrix (x) and labels (y) were created to classify whether an account was fraudulent (1) or legitimate (0).

## Model Architecture

The model uses Edge-feature-aware Graph Convolution via NNConv (from PyTorch Geometric):
Input: Node and edge features

Layers: 2 NNConv layers + 1 Linear layer

Loss Function: Weighted CrossEntropy (to handle class imbalance)

Optimizer: Adam (learning rate = 0.01)

## Training & Evaluation
Training Summary
Epoch 30 | Loss: 0.5625 | Accuracy: 0.6323

| Metric                       | Score               |
| ---------------------------- | ------------------- |
| Precision                    | 0.0218              |
| Recall                       | 0.3000              |
| F1-Score                     | 0.0406              |
| Fraudulent Accounts Detected | 1,931 out of 50,000 |

## Visualizations

 Transaction Graph

Red nodes → flagged as potential fraud

Blue nodes → legitimate

Node size ∝ predicted fraud probability

nx.draw(subG, node_color=colors, node_size=sizes, with_labels=False)

## Account Category Distribution

Bar chart showing predicted fraud vs legitimate account distribution.

## Key Learnings

Built an end-to-end GNN-based fraud detection pipeline.

Gained experience with graph feature engineering and PyTorch Geometric.

Addressed class imbalance using weighted losses.

Visualized fraud propagation in transaction networks




Author
Sara Vartak
📍 Dubai, UAE


