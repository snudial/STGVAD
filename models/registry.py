import inspect
from typing import Type

# 1) 전역 레지스트리
MODEL_REGISTRY: dict[str, Type] = {}

# 2) 모델 등록 데코레이터
def register_model(name: str):
    def deco(cls: Type):
        MODEL_REGISTRY[name] = cls
        return cls
    return deco

# 3) 모델 생성 함수
def get_model(name: str, **kwargs):
    """
    name:
      - "lstm", "transformer", "gat", "mlp"           -> 단일 모델
      - "lstm_<gnn_type>", "transformer_<gnn_type>"   -> GNN+백본 모델
    kwargs:
      - input_dim, gnn_hidden, lstm_hidden, transformer_hidden, out_dim, residual 등
    """
    # 1) 이름이 레지스트리에 있는지 확인
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}")
    
    cls = MODEL_REGISTRY[name]

    # 2) composite 모델의 경우(키에 '_'가 포함되어 있다면),
    #    e.g. "lstm_gcn" → gnn_type="gcn" 자동 삽입
    if "_" in name:
        _, spatial = name.split("_", 1)
        kwargs["gnn_type"] = spatial

    # 3) inspect로 __init__ 파라미터만 골라 전달
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters) - {"self", "args", "kwargs"}
    filtered = {k: v for k, v in kwargs.items() if k in valid}

    return cls(**filtered)