# 🔍 Advanced Knowledge Representation using GNNs and GATs

This repository implements **MetaGAT**, a novel architecture that combines **Graph Neural Networks (GNNs)** and **Graph Attention Networks (GATs)** for dynamic, scalable, and multi-relational knowledge representation. This project is based on our research paper titled **"Advanced Knowledge Representation using Graph Neural Networks and Graph Attention Networks"**.

## 📄 Abstract

As data becomes more interconnected, representing it in graph structures allows capturing rich relational knowledge. Traditional models often fail in such non-Euclidean domains. Our approach enhances representation using GCN layers and GATs, specifically designed to:

- Perform **node classification**, **link prediction**, and **clustering**.
- Leverage **multi-head attention** to dynamically focus on important features.
- Improve contextual awareness in domains such as **knowledge graphs**, **recommendation systems**, **semantic analysis**, **disease prediction**, and **social networks**.

## 🧠 Proposed Architecture

![System Architecture of MetaGAT](./images/system_architecture.png)

> Figure: System Architecture of Advanced KR using GNN and GAT

The architecture uses PyTorch Geometric, combining GCN and GAT mechanisms with graph-specific preprocessed datasets. Enhanced scalability and accuracy are achieved via multi-head attention.

## 🌐 Application Use Case

![Social Media Application Diagram](./images/application_socialmedia.png)

> Figure: Application Diagram for Social Media Use Case

We simulate real-world graphs using synthetic datasets that represent disease networks and social profile relations. These help validate our model’s effectiveness in real-world scenarios.

## 📊 Experimental Results

### Table 1: Enhanced MetaGAT Performance

| Dataset   | Accuracy (%) | F1-Score (%) |
|-----------|--------------|--------------|
| Cora      | 87.5         | 85.3         |
| Citeseer  | 84.2         | 82.1         |
| PubMed    | 91.0         | 89.5         |
| DBLP      | 85.0         | 83.0         |
| **Average** | **86.7**     | **84.0**     |

### Table 2: Transfer Learning Comparison

| Model        | Accuracy (%) | F1-Score (%) |
|--------------|--------------|--------------|
| GCN          | 78.5         | 76.3         |
| MetaGCN      | 85.6         | 83.9         |
| GATv2        | 80.2         | 78.7         |
| **MetaGAT**  | **86.4**     | **85.1**     |

### Table 3: Logical Rule Complexity Impact

| Rule Type      | Accuracy (%) | F1-Score (%) |
|----------------|--------------|--------------|
| Simple Rules   | 88.0         | 84.5         |
| Complex Rules  | 89.5         | 85.7         |

## 📁 Dataset

- **[SynDisNet – Synthetic Disease Network](https://www.kaggle.com/datasets/aayusic/synthetic-disease-network-dataset-syndisnet)**
- **[Relational – Synthetic Social Profiles](https://www.kaggle.com/datasets/atharv01/synthetic-social-profiles-dataset-relational)**

Both datasets are generated for healthcare and social network analysis and are publicly hosted on Kaggle.

## 🛠️ Tech Stack

- **Python 3.9+**
- **PyTorch Geometric**
- **DGL / NetworkX / Scikit-learn**
- **Matplotlib** for visualization

## 📂 Repository Structure

```bash
.
├── data/
│   └── syndisnet/          # Synthetic Disease Network
│   └── social_profiles/    # Relational social graphs
├── models/
│   ├── gnn_model.py
│   ├── gat_model.py
│   ├── meta_gat.py         # Proposed MetaGAT Model
├── experiments/
│   ├── train.py
│   ├── evaluate.py
├── utils/
│   └── preprocess.py
├── images/
│   ├── system_architecture.png
│   ├── application_socialmedia.png
├── README.md
└── requirements.txt
