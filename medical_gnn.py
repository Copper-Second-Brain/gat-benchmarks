import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv
import random 

# 1. Data Loading and Graph Construction
def load_medical_data():
    file_path = '../gat-representation/datasets/SynDisNet.csv'

    # Parameters
    desired_sample_size = 100
    chunksize = 1000  # Adjust this chunk size based on memory limits

    # Initialize an empty list to store sampled rows
    sampled_data = []

    # Iterate over chunks
    for chunk in pd.read_csv(file_path, chunksize=chunksize):
        # Calculate the sample size for this chunk
        fraction = desired_sample_size / len(chunk)
        # Ensure fraction is <= 1
        fraction = min(1, fraction)
        
        # Sample from the chunk
        sampled_chunk = chunk.sample(frac=fraction, random_state=random.randint(0, 100))
        
        # Append to the sampled data
        sampled_data.append(sampled_chunk)
        
        # Check if we have enough samples
        if sum(len(df) for df in sampled_data) >= desired_sample_size:
            break

    # Combine all sampled chunks into a single DataFrame
    df = pd.concat(sampled_data).head(desired_sample_size)

    # Result

    # Feature engineering
    feature_cols = ['systolic_bp', 'diastolic_bp', 'cholesterol', 'BMI', 
                   'heart_rate', 'fatigue_severity', 'cough_type', 
                   'blood_glucose', 'chest_pain', 'smoking_status']
    
    # Normalize features
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols])
    
    # Create graph structure (k-NN with k=2)
    nn = NearestNeighbors(n_neighbors=2)
    nn.fit(features)
    _, indices = nn.kneighbors(features)
    
    # Create edge indices
    edge_list = []
    for i, neighbors in enumerate(indices):
        for j in neighbors:
            if i != j:
                edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df['disease_family'])
    
    return Data(
        x=torch.tensor(features, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(labels, dtype=torch.long)
    )

# 2. Model Architectures (Modified for Medical Data)
class GCNv1(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

class KeGCNv2(GCNv1):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.enhance = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x, edge_index):
        x = super().forward(x, edge_index)
        return x + self.enhance(x)

class GATv1(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

class KeGATv2(GATv1):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.enhance = nn.MultiheadAttention(out_dim, num_heads=1)
        
    def forward(self, x, edge_index):
        x = super().forward(x, edge_index)
        attn_out, _ = self.enhance(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        return attn_out.squeeze(0)

# 3. Training and Evaluation
def train_model(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        model.eval()
        _, pred = model(data.x, data.edge_index).max(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.y.size(0)
        best_acc = max(best_acc, acc)
    
    return best_acc

def compare_models():
    data = load_medical_data()
    in_dim = data.x.size(1)
    out_dim = data.y.unique().size(0)
    
    models = {
        'GCNv1': GCNv1(in_dim, 8, out_dim),
        'KeGCNv2': KeGCNv2(in_dim, 8, out_dim),
        'GATv1': GATv1(in_dim, 8, out_dim),
        'KeGATv2': KeGATv2(in_dim, 8, out_dim)
    }
    
    results = {}
    for name, model in models.items():
        acc = train_model(model, data)
        results[name] = acc
    
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.ylabel('Accuracy')
    plt.title('Medical Diagnosis Model Comparison')
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    else:
        print("Use --compare to run comparison")