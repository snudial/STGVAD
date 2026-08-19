import torch
from .registry import register_model

@register_model("lstm")
class LSTM(torch.nn.Module):
    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()
        # 하나의 LSTM만 선언 → 트랙마다 재사용
        self.lstm = torch.nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = torch.nn.Linear(hidden_dim, 1) 

    def forward(self, x, edge_index=None, batch=None):
        """
        x: [B*30, 5], B: batch_size, 3트랙×각10노드=30
        """
        B = x.size(0) // 30  # 배치 크기
        # reshape => [B, 30, 5]
        x_3d = x.view(B, 30, -1)

        # 트랙 A = x_3d[:,0:10,:], B= x_3d[:,10:20,:], C= x_3d[:,20:30,:]
        trackA = x_3d[:, 0:10, :]  # [B,10,5]
        trackB = x_3d[:, 10:20, :]
        trackC = x_3d[:, 20:30, :]

        # 하나의 LSTM을 각 트랙에 대해 순차 호출
        outA, _ = self.lstm(trackA)  # [B,10,hidden_dim]
        outB, _ = self.lstm(trackB)
        outC, _ = self.lstm(trackC)

        # 최종 스텝 hidden state만 사용(마지막 시점)
        # outA[:,-1,:] => [B, hidden_dim]
        finalA = self.linear(outA[:, -1, :])  # => [B,1]
        finalB = self.linear(outB[:, -1, :])  # => [B,1]
        finalC = self.linear(outC[:, -1, :])  # => [B,1]

        # [B,3]
        out = torch.cat([finalA, finalB, finalC], dim=1)
        return torch.sigmoid(out)
