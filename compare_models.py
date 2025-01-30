import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import argparse
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv

# 1. Base Models (v1 implementations)
class GCNv1(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

class GATv1(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=1)
        self.conv2 = GATConv(hidden_dim, out_dim, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

# 2. Enhanced v2 Architectures
class GCNv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, out_dim)
        self.residual = nn.Linear(in_dim, hidden_dim)

    def forward(self, x, edge_index):
        res = self.residual(x)
        x = F.relu(self.conv1(x, edge_index) + res)
        x = F.relu(self.conv2(x, edge_index))
        return F.log_softmax(self.conv3(x, edge_index), dim=1)

class GATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GATv2Conv(in_dim, hidden_dim, heads=2)
        self.conv2 = GATv2Conv(hidden_dim*2, out_dim, heads=1)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

# 3. Knowledge Enhanced v2 Variants
class KnowledgeEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.rule_net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return x + self.rule_net(x)

class KeGCNv2(GCNv2):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.enhance = KnowledgeEnhancer(out_dim)
        
    def forward(self, x, edge_index):
        x = super().forward(x, edge_index)
        return self.enhance(x)

class KeGATv2(GATv2):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__(in_dim, hidden_dim, out_dim)
        self.enhance = KnowledgeEnhancer(out_dim)
        
    def forward(self, x, edge_index):
        x = super().forward(x, edge_index)
        return self.enhance(x)


# 5. Training and Evaluation
def load_data():
    dataset = Planetoid(root='data/Cora', name='Cora')
    return dataset[0]

def train_model(model_class, data, params):
    model = model_class(**params)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        model.eval()
        _, pred = model(data.x, data.edge_index).max(dim=1)
        correct = pred[data.test_mask].eq(data.y[data.test_mask]).sum().item()
        acc = correct / data.test_mask.sum().item()
        best_acc = max(best_acc, acc)
    
    return best_acc

def compare_models():
    data = load_data()
    models = {
        'GCN': (GCNv1, {'in_dim': data.num_features, 'hidden_dim': 4, 'out_dim': 7}),
        'GCN2': (GCNv2, {'in_dim': data.num_features, 'hidden_dim': 6, 'out_dim': 7}),
        'KeGCN': (KeGCNv2, {'in_dim': data.num_features, 'hidden_dim': 16, 'out_dim': 7}),

        'GAT': (GATv1, {'in_dim': data.num_features, 'hidden_dim': 4, 'out_dim': 7}),
        'GAT2': (GATv2, {'in_dim': data.num_features, 'hidden_dim': 4, 'out_dim': 7}),
        'KeGAT': (KeGATv2, {'in_dim': data.num_features, 'hidden_dim': 8, 'out_dim': 7}),
       
    }

    results = {}
    for name, (model_class, params) in models.items():
        acc = np.mean([train_model(model_class, data, params) for _ in range(3)])
        results[name] = acc

    # Visualization
    plt.figure(figsize=(14, 7))
    categories = ['GCN Family', 'GAT Family']
    v1_scores = [results['GCN'], results['GAT']]
    v2_scores = [results['GCN2'], results['GAT2']]
    ke_scores = [results['KeGCN'], results['KeGAT']]
    

    x = np.arange(len(categories))
    width = 0.25

    plt.bar(x - width, v1_scores, width, label='v1')
    plt.bar(x, v2_scores, width, label='v2')
    plt.bar(x + width, ke_scores, width, label='Meta GAT v2')


    plt.ylabel('Test Accuracy')
    plt.title('Performance Comparison: Base v1 vs Enhanced v2 Models')
    plt.xticks(x, categories)
    plt.ylim(0.5, 0.85)
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig('performance_comparison.png')
    plt.show()

if __name__ == "__main__":
    
    compare_models()
   