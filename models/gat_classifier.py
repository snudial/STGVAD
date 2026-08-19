import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from .registry import register_model

@register_model("gat")
class SimpleGATClassifier(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64, heads=2, num_labels=3):
        super().__init__()
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_labels),  # num_labels=3
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index, batch):
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.gat2(x, edge_index)
        x = F.relu(x)

        outputs = []
        for gid in batch.unique():
            node_mask = (batch == gid)
            graph_x = x[node_mask]
            pooled = graph_x.mean(dim=0, keepdim=True)
            pred = self.classifier(pooled)  # shape = (1, 3)
            outputs.append(pred)
        return torch.cat(outputs, dim=0)  # shape = (N, 3) for N graphs
