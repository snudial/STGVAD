import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv  # GraphSAGE is SAGEConv

from .registry import register_model


@register_model("lstm_gcn")
@register_model("lstm_gat")
@register_model("lstm_sage")
class GenericGNNLSTM(torch.nn.Module):
    def __init__(
        self,
        gnn_type="gcn",             # "gcn", "gat", "sage"
        input_dim=5,
        gnn_hidden=16,
        lstm_hidden=32,
        out_dim=1,
        residual=True
    ):
        super().__init__()
        self.residual = residual
        self.gnn_type = gnn_type

        # GNN 선택
        GNN = {
            "gcn": GCNConv,
            "gat": GATConv,
            "sage": SAGEConv
        }[gnn_type]

        self.gnn1 = GNN(input_dim, gnn_hidden)
        self.gnn2 = GNN(gnn_hidden, gnn_hidden)

        # Projection layer (if needed for residual)
        if residual and input_dim != gnn_hidden:
            self.input_proj = torch.nn.Linear(input_dim, gnn_hidden)
        else:
            self.input_proj = None

        # LSTM
        self.lstm = torch.nn.LSTM(input_size=gnn_hidden, hidden_size=lstm_hidden, batch_first=True)

        # Final classifier
        self.final_linear = torch.nn.Linear(lstm_hidden, out_dim)

    def forward(self, x, edge_index, batch):
        out1 = F.relu(self.gnn1(x, edge_index))
        out2 = F.relu(self.gnn2(out1, edge_index))

        if self.residual:
            if self.input_proj is not None:
                out2 = out2 + self.input_proj(x)
            else:
                out2 = out2 + x

        B = out2.size(0) // 30
        x_3d = out2.view(B, 30, -1)

        trackA, trackB, trackC = x_3d[:, 0:10, :], x_3d[:, 10:20, :], x_3d[:, 20:30, :]

        outA, _ = self.lstm(trackA)
        outB, _ = self.lstm(trackB)
        outC, _ = self.lstm(trackC)

        finalA, finalB, finalC = outA[:, -1, :], outB[:, -1, :], outC[:, -1, :]
        predA = self.final_linear(finalA)
        predB = self.final_linear(finalB)
        predC = self.final_linear(finalC)

        return torch.sigmoid(torch.cat([predA, predB, predC], dim=1))