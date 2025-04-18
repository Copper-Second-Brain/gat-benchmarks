import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import remove_self_loops, add_self_loops


class RuleLayer(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_rules=3):
        super(RuleLayer, self).__init__()
        self.num_rules = num_rules
        
        # Rule embeddings
        self.rule_weights = nn.Parameter(torch.Tensor(num_rules, in_dim))
        self.rule_bias = nn.Parameter(torch.Tensor(num_rules))
        
        # Knowledge integration
        self.knowledge_attention = nn.Linear(in_dim, num_rules)
        
        # Initialization
        nn.init.xavier_uniform_(self.rule_weights)
        nn.init.zeros_(self.rule_bias)

    def forward(self, x):
        batch_size = x.size(0)
        
        # Calculate rule activations
        rule_activations = torch.matmul(x, self.rule_weights.t()) + self.rule_bias
        rule_activations = torch.sigmoid(rule_activations)  # [batch_size, num_rules]
        
        # Calculate knowledge attention weights
        knowledge_attn = F.softmax(self.knowledge_attention(x), dim=1)  # [batch_size, num_rules]
        
        # Apply knowledge-enhanced weighting
        weighted_rules = knowledge_attn * rule_activations
        
        # Reshape for multiplication with rule weights
        weighted_rules = weighted_rules.unsqueeze(2)  # [batch_size, num_rules, 1]
        expanded_rule_weights = self.rule_weights.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_rules, in_dim]
        
        # Calculate knowledge-enhanced features
        knowledge_features = torch.sum(weighted_rules * expanded_rule_weights, dim=1)  # [batch_size, in_dim]
        
        # Combine with original features
        enhanced_x = x + knowledge_features
        
        return enhanced_x


class KeGCNv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_rules=3, dropout=0.5):
        super(KeGCNv2, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Knowledge enhancement layers
        self.rule_layer1 = RuleLayer(in_channels, hidden_channels, num_rules)
        self.rule_layer2 = RuleLayer(hidden_channels, out_channels, num_rules)
        
        # Other parameters
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Apply knowledge enhancement to input features
        x = self.rule_layer1(x)
        
        # First GCN layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply knowledge enhancement to hidden features
        x = self.rule_layer2(x)
        
        # Second GCN layer
        x = self.conv2(x, edge_index)
        
        return x


class KeGATv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_rules=3, heads=8, dropout=0.5):
        super(KeGATv2, self).__init__()
        
        # GAT layers
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # The output of conv1 will be hidden_channels * heads
        self.conv2 = GATConv(hidden_channels * heads, out_channels, dropout=dropout)
        
        # Knowledge enhancement layers
        self.rule_layer1 = RuleLayer(in_channels, hidden_channels, num_rules)
        self.rule_layer2 = RuleLayer(hidden_channels * heads, out_channels, num_rules)
        
        # Other parameters
        self.dropout = dropout
        self.heads = heads

    def forward(self, x, edge_index):
        # Apply knowledge enhancement to input features
        x = self.rule_layer1(x)
        
        # First GAT layer
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply knowledge enhancement to hidden features
        x = self.rule_layer2(x)
        
        # Second GAT layer
        x = self.conv2(x, edge_index)
        
        return x