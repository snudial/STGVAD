# STGVAD — Spatio-Temporal Graph Vessel Anomaly Detection

AIS(Automatic Identification System) 선박 항적 데이터를 **시공간 그래프**로 변환하고,
GNN + 시퀀스 모델(LSTM / Transformer) 조합으로 **항로(route) 단위 이상 탐지**를 수행하는 실험 코드입니다.

정답 라벨이 없는 실데이터를 다루기 위해, 물리적으로 그럴듯한 **이상치를 합성 주입**해
라벨을 만들고 그 위에서 모델을 비교합니다.

---

## 파이프라인

```
AIS CSV (OMTAD)
   │  data/loader.py          1시간 간격 리샘플링·보간, COURSE → (sin, cos)
   ▼
시각별 OPTICS 클러스터링
   │  cluster/optics.py       매 시각 t 의 (LAT, LON) 으로 선박을 군집화
   │  graph/builder.py        t 직전 h=10 시간 구간을 잘라 클러스터당 k=3개 항적 샘플링
   ▼
클러스터 그래프  (3 tracks × 10 nodes = 노드 30개)
   │                          storage/nx_graphs_original.pkl
   ▼
이상치 주입
   │  graph/inject_anomaly.py 항적을 route 단위로 분리 → 일부 route의 연속 구간에
   │                          가속도·선회율 이상(μ + 3.5σ)을 주입 → 다시 클러스터로 병합
   ▼
주입 그래프  storage/nx_graphs_injected/nx_graphs_injected_r={route}_n={node}.pkl
   │  graph/convert.py        PyTorch Geometric `Data` 로 변환
   ▼
학습 / 평가  scripts/train.py → results/{model}/r{route}_n{node}/
```

**노드 특징 (5차원)**: `LON, LAT, SPEED, COURSE_SIN, COURSE_COS`

**엣지 구성** (`graph/convert.py:create_cluster_edges`)
- 같은 항적 내: 과거 → 미래 방향으로 fully connected
- 다른 항적 간: 노드 `i` → 상대 항적의 노드 `k` (`k ≥ i`), 즉 미래 정보 누수 없음

**라벨 (`y`, 3차원)**: 각 항적이 이상인지 여부. 한 항적 안에서 이상 노드가
`block_size = 10 × node_ratio` 개 이상 **연속**으로 나타나면 1.

**주입 비율 두 가지**
| 인자 | 의미 |
| --- | --- |
| `route_ratio` | 전체 항적 중 이상으로 만들 비율 (0.1 / 0.3 / 0.5) |
| `node_ratio` | 한 항적 안에서 이상으로 만들 연속 노드 비율 (0.3 / 0.5 / 0.7) |

---

## Setup

### 1. 저장소 클론

```bash
git clone https://github.com/snudial/STGVAD.git
cd STGVAD
```

### 2. Python 환경

Python 3.10 이상을 권장합니다.

```bash
python -m venv .venv && source .venv/bin/activate   # 또는 conda create -n stgvad python=3.10

pip install -r requirements.txt
```

> **PyTorch / PyG 설치 주의**
> `requirements.txt` 는 CUDA 12.4 + `torch==2.6.0` 기준입니다.
> 다른 CUDA 버전을 쓴다면 torch 와 torch_geometric 은 공식 안내에 맞춰 따로 설치하세요.
>
> ```bash
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> pip install torch_geometric==2.6.1
> ```

### 3. 데이터셋 준비

AIS 원본은 공개 데이터셋 **OMTAD (Open Maritime Traffic Analysis Dataset)** 를 사용하며,
용량이 커서 이 저장소에는 포함하지 않습니다.

```bash
git clone https://github.com/EdithCowan/OMTAD.git data/OMTAD
```

기본 경로는 `data/OMTAD/West Grid` 이고, 다른 곳에 두었다면 환경변수로 지정합니다.

```bash
export OMTAD_ROOT="/path/to/OMTAD/West Grid"
```

기대하는 디렉터리 구조:

```
data/OMTAD/West Grid/
├── 2018/
│   ├── cargo/{Jan,Feb,...}/MPF_{Mon}_2018_Grid_Cargo.csv
│   ├── passenger/...
│   └── tanker/...
├── 2019/
└── 2020/
```

사용 연도는 `pipeline/generation_runner.py` 의 `year_list` 에서 바꿀 수 있습니다 (기본 2018–2020).

### 4. 디스크 공간

그래프 pickle 이 매우 큽니다. 실행 전에 여유 공간을 확인하세요.

| 산출물 | 크기 |
| --- | --- |
| `storage/nx_graphs_original.pkl` | 약 0.8 GB |
| `storage/nx_graphs_injected/*.pkl` (조합당) | 약 2.3 GB |
| 9개 조합 전부 생성 시 | **약 21 GB** |

---

## 실행

모든 명령은 저장소 루트에서 실행합니다. `Makefile` 이 `PYTHONPATH` 를 잡아 줍니다.
`make help` 로 전체 명령을 볼 수 있습니다.

### 1. 원본 그래프 생성 (최초 1회)

```bash
make cluster
```
→ `storage/nx_graphs_original.pkl`

### 2. 이상치 주입

```bash
make inject route=0.5 node=0.5   # 한 조합만
make inject_all                  # route 3종 × node 3종 = 9개 조합 전부
```
→ `storage/nx_graphs_injected/nx_graphs_injected_r=0.5_n=0.5.pkl`

### 3. 학습

단일 실행:

```bash
make train model=lstm_gcn route=0.5 node=0.5 seed=42
CUDA_VISIBLE_DEVICES=1 make train model=lstm_gcn route=0.5 node=0.5 seed=42
```

한 모델의 전체 조합(9조합 × 5시드 = 45회)을 순차 실행하고 자동 집계까지:

```bash
make run model=lstm_gcn gpu=0
# 또는 직접
bash scripts/run_experiments.sh lstm_gcn 0
```

시드·비율은 환경변수로 덮어쓸 수 있습니다.

```bash
SEEDS="42" ROUTE_RATIOS="0.5" NODE_RATIOS="0.5" bash scripts/run_experiments.sh lstm 0
```

### 4. 결과 집계 / 시각화

```bash
make tables                                  # 모델 × 조합 비교표 생성 → final_results/
python tools/plot_f1_curves.py --save_dir figures
```

---

## 사용 가능한 모델

`--model` 에 넣는 이름입니다.

| 이름 | 구조 |
| --- | --- |
| `mlp` | 항적별 flatten MLP (그래프 미사용 baseline) |
| `lstm` | 항적별 LSTM (그래프 미사용 baseline) |
| `transformer` | 항적별 Transformer encoder (그래프 미사용 baseline) |
| `gat` | GAT 2층 + mean pooling |
| `lstm_gcn`, `lstm_gat`, `lstm_sage` | GNN 2층 (+residual) → 항적별 LSTM |
| `transformer_gcn`, `transformer_gat`, `transformer_sage` | GNN 2층 (+residual) → 항적별 Transformer |

모든 모델은 `(x, edge_index, batch)` 를 받아 `[B, 3]` 확률을 반환합니다 (항적 A/B/C 각각의 이상 확률).

### 새 모델 추가하기

1. `models/my_model.py` 에 `torch.nn.Module` 을 작성하고 `@register_model("my_model")` 을 붙입니다.
2. `models/__init__.py` 에 import 를 한 줄 추가합니다 (레지스트리 등록 시점 확보).
3. 바로 실행됩니다.

```bash
make run model=my_model gpu=0
```

`models/registry.py` 는 `__init__` 시그니처를 검사해 필요한 인자만 골라 넘기므로,
모델이 쓰지 않는 하이퍼파라미터는 무시됩니다. 이름에 `_` 가 있으면
(`lstm_gcn` → `gnn_type="gcn"`) 뒷부분이 `gnn_type` 으로 자동 주입됩니다.

---

## 결과 형식

학습 산출물은 `.gitignore` 처리되어 저장소에는 포함되지 않습니다.

```
results/{model}/r{route}_n{node}/
├── results_seed_2.csv        # 시드별 테스트 지표 (Metric, Value)
├── results_seed_12.csv
├── ...
├── best_model_s2.pt          # early stopping 체크포인트
└── aggregated_results.csv    # 시드 평균 ± 표준편차

final_results/
├── F1_score_comparison.csv   # 행=모델, 열=r{route}_n{node}
├── AUC_ROC_comparison.csv
├── ...
└── best_results_summary.csv
```

기록되는 지표: `Accuracy`, `Precision`, `Recall`, `F1-score`, `Precision@100`,
`BalancedAcc`, `MCC`, `AUC-ROC`, `PR-AUC` (`train_utils/evaluate.py`).

학습 설정: Adam(lr=1e-3), BCELoss, batch 64, 최대 30 epoch,
validation loss 기준 early stopping(patience=5), split 7:1.5:1.5.

---

## 디렉터리 구조

```
.
├── data/
│   └── loader.py                    # AIS CSV 로드, 1시간 리샘플링·보간
├── cluster/
│   └── optics.py                    # 시각별 OPTICS 군집화
├── graph/
│   ├── builder.py                   # 클러스터 → NetworkX 시간 그래프
│   ├── inject_anomaly.py            # route 단위 이상치 주입
│   ├── convert.py                   # 엣지 구성 + PyG Data 변환, 라벨 생성
│   └── io.py                        # pickle 저장/로드
├── models/
│   ├── registry.py                  # @register_model 레지스트리, get_model()
│   ├── mlp_classifier.py
│   ├── lstm.py / transformer.py
│   ├── gat_classifier.py
│   └── gnn_lstm.py / gnn_transformer.py
├── train_utils/
│   ├── early_stopping.py
│   ├── evaluate.py                  # 9개 지표 계산
│   └── random_seed.py
├── pipeline/
│   ├── generation_runner.py         # 로드 → 군집화 → 그래프 생성
│   └── injection_runner.py          # 주입 / 로드 후 PyG 변환
├── scripts/
│   ├── cluster_once.py              # make cluster
│   ├── inject_graphs.py             # make inject
│   ├── train.py                     # make train
│   ├── aggregate_results.py         # 시드별 결과 → mean ± std
│   └── run_experiments.sh           # make run (전체 조합 반복)
├── tools/
│   ├── make_comparison_tables.py    # make tables
│   └── plot_f1_curves.py            # F1 곡선 그림
├── Makefile
└── requirements.txt
```

---

## 참고

- 데이터셋: [OMTAD — Open Maritime Traffic Analysis Dataset](https://github.com/EdithCowan/OMTAD)
