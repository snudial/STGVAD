import torch
import torch.nn.functional as F

from .registry import register_model

@register_model("transformer")
class Transformer(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        # 입력 차원 -> Transformer 차원 매핑 레이어
        self.input_linear = torch.nn.Linear(input_dim, hidden_dim)

        # 하나의 transformer만 선언 → 트랙마다 재사용
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 분류 레이어
        self.final_linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear = torch.nn.Linear(hidden_dim, 1)  # 이진 출력

    def forward(self, x, edge_index=None, batch=None):
        """
        x: [B*30, 5], B: batch_size, 3트랙×각10노드=30
        """
        B = x.size(0) // 30  # 배치 크기
        x_3d = x.view(B, 30, -1)

        # 각 트랙 분리
        tracks = [x_3d[:, i*10:(i+1)*10, :] for i in range(3)]
        scores = []

        for track in tracks:
            # 입력 임베딩
            emb = self.input_linear(track)  # [B,10,hidden_dim]
            # transformer 인코딩
            trans_out = self.transformer(emb)  # [B,10,hidden_dim]
            # 마지막 시점 임베딩 추출
            last_hidden = trans_out[:, -1, :]  # [B,hidden_dim]
            # 분류를 위한 피드포워드
            hidden = F.relu(self.final_linear(last_hidden))  # [B,hidden_dim]
            score = self.linear(hidden)  # [B,1]
            scores.append(score)

        # [B,3]
        out = torch.cat(scores, dim=1)
        return torch.sigmoid(out)