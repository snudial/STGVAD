import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # GraphSAGE는 SAGEConv로 구현됨

from .registry import register_model

@register_model("transformer_gcn")
@register_model("transformer_gat")
@register_model("transformer_sage")
class GenericGNNTransformer(torch.nn.Module):
    def __init__(
        self,
        gnn_type="gcn",              # "gcn", "gat", "sage"
        input_dim=5,
        gnn_hidden=16,
        transformer_hidden=32,
        out_dim=1,
        residual=True,
        num_heads=4,
    ):
        super().__init__()
        self.residual = residual
        self.gnn_type = gnn_type

        # GNN 레이어 타입 설정
        GNN = {
            "gcn": GCNConv,
            "gat": GATConv,
            "sage": SAGEConv,
        }[gnn_type]

        # 두 개의 GNN 레이어
        self.gnn1 = GNN(input_dim, gnn_hidden)
        self.gnn2 = GNN(gnn_hidden, transformer_hidden)

        # projection for residual
        if residual and input_dim != gnn_hidden:
            self.input_proj = torch.nn.Linear(input_dim, transformer_hidden)
        else:
            self.input_proj = None

        # Transformer
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=transformer_hidden, nhead=num_heads, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classification layer
        self.final_linear = torch.nn.Linear(transformer_hidden, out_dim)

    def forward(self, x, edge_index, batch):
        out1 = F.relu(self.gnn1(x, edge_index))
        out2 = F.relu(self.gnn2(out1, edge_index))

        if self.residual:
            if self.input_proj is not None:
                out2 = out2 + self.input_proj(x)
            else:
                out2 = out2 + x

        # reshape into (B, 30, d_model)
        B_size = out2.size(0) // 30
        x_3d = out2.view(B_size, 30, -1)

        # split into 3 tracks
        trackA, trackB, trackC = x_3d[:, 0:10, :], x_3d[:, 10:20, :], x_3d[:, 20:30, :]

        # transformer encoding
        outA = self.transformer(trackA)
        outB = self.transformer(trackB)
        outC = self.transformer(trackC)

        finalA, finalB, finalC = outA[:, -1, :], outB[:, -1, :], outC[:, -1, :]
        predA = self.final_linear(finalA)
        predB = self.final_linear(finalB)
        predC = self.final_linear(finalC)

        return torch.sigmoid(torch.cat([predA, predB, predC], dim=1))
