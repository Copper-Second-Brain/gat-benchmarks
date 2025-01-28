import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class KnowledgeEnhancementLayer(nn.Module):
    def __init__(self, in_dim, rule_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, rule_dim)
        self.gate = nn.Linear(rule_dim, 1)
        self.rule_weights = nn.Parameter(torch.rand(1))
    
    def forward(self, H, rules):
        # Apply rule-based adjustments (simplified example)
        rule_out = self.W(H)
        gate = torch.sigmoid(self.gate(rule_out))
        delta = gate * self.rule_weights
        return H + delta

class KeGNN(nn.Module):
    def __init__(self, base_model, in_dim, hidden_dim, out_dim, num_rules=3):
        super().__init__()
        self.base_model = base_model
        if base_model == 'GCN':
            self.conv1 = GCNConv(in_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, out_dim)
        elif base_model == 'GAT':
            self.conv1 = GATConv(in_dim, hidden_dim, heads=1)
            self.conv2 = GATConv(hidden_dim, out_dim, heads=1)
        self.enhance = KnowledgeEnhancementLayer(out_dim, num_rules)
    
    def forward(self, x, edge_index, rules):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        x = self.enhance(x, rules)
        return F.log_softmax(x, dim=1)