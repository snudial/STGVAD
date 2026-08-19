import torch
from .registry import register_model

@register_model("mlp")
class SimpleMLPClassifier(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        # 한 트랙(노드10*피처5=50차원)을 처리하는 하나의 MLP
        track_in_dim = 10 * input_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(track_in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1)  # 이진출력
        )

    def forward(self, x, edge_index=None, batch=None):
        """
        x: shape [B*30, 5], 즉 batch_size * (track 3개 * node 10개) * feature=5
        인자로 edge_index/batch가 오더라도 무시 (에러 방지용)
        """
        # (1) 배치 크기 구하기
        B = x.size(0) // 30

        # (2) reshape => [B, 30, 5]
        x_3d = x.view(B, 30, -1)  # shape [B,30,5]

        # (3) 트랙별로 10개 노드(5차원) → flatten(10*5=50차원)
        trackA = x_3d[:, 0:10, :].reshape(B, -1)  # [B,50]
        trackB = x_3d[:, 10:20, :].reshape(B, -1)
        trackC = x_3d[:, 20:30, :].reshape(B, -1)

        # (4) 한 개의 mlp 를 3번 호출
        outA = self.mlp(trackA)  # [B,1]
        outB = self.mlp(trackB)  # [B,1]
        outC = self.mlp(trackC)  # [B,1]

        # (5) 합쳐서 [B,3]
        out = torch.cat([outA, outB, outC], dim=1)
        return torch.sigmoid(out)  # e.g. BCELoss 쓸 때

