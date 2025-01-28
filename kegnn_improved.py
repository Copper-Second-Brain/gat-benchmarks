
from kegnn import KeGNN
from torch import nn
class ImprovedKnowledgeEnhancementLayer(nn.Module):
    def __init__(self, in_dim, rule_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=in_dim, num_heads=2)
        self.W = nn.Linear(in_dim, rule_dim)
    
    def forward(self, H, rules):
        attn_out, _ = self.attention(H, H, H)
        rule_out = self.W(attn_out)
        return H + rule_out

class ImprovedKeGNN(KeGNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enhance = ImprovedKnowledgeEnhancementLayer(self.out_dim, kwargs.get('num_rules', 3))