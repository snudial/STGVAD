<div align="center">

# ⛴️ STGVAD: Spatio-Temporal Graph-Based <br> Vessel Behavior Anomaly Detection

[![Paper on IEEE Access](https://img.shields.io/badge/Paper-IEEE%20Access-00629B?logo=ieee&logoColor=white)](https://ieeexplore.ieee.org/document/11165046)

</div>


## News

- **2026**: Our paper is published in **IEEE Access**, vol. 14, pp. 2152–2165. 🎉


## Overview

Spatio-Temporal GNNs usually assume fixed spatial anchors such as sensors or road segments.
The open sea has no such anchors, so STGVAD builds the graph differently: **every timestamped
vessel state becomes a node**, and a unified multi-ship trajectory graph is formed by linking
temporally adjacent nodes and by grouping spatially proximate vessels with **OPTICS** clustering.

Because real AIS data carries no anomaly labels, this repository also synthesizes labels by
injecting physically grounded kinematic anomalies (excess acceleration and turn rate) into a
controlled fraction of trajectories.


## Features

- **Cluster**: hourly OPTICS clustering over `(LAT, LON)`, producing one graph per vessel cluster
- **Inject**: route-level synthetic anomaly injection with controllable route / node ratios
- **Train**: 10 registered models — LSTM, Transformer, MLP, GAT, and GNN+sequence hybrids
- **Aggregate**: multi-seed mean ± std aggregation and cross-model comparison tables
- **Plot**: F1-score curves over anomaly ratios


## Pipeline

```
AIS CSV (OMTAD)
   │  data/loader.py            1-hour resampling & interpolation, COURSE → (sin, cos)
   ▼
Hourly OPTICS clustering
   │  cluster/optics.py         cluster vessels by (LAT, LON) at each target hour t
   │  graph/builder.py          take the preceding h=10 hours, sample k=3 tracks per cluster
   ▼
Cluster graph  (3 tracks × 10 nodes = 30 nodes)
   │                            storage/nx_graphs_original.pkl
   ▼
Anomaly injection
   │  graph/inject_anomaly.py   split into per-route subgraphs → inject acceleration /
   │                            turn-rate anomalies (μ + 3.5σ) over a contiguous node block
   │                            → regroup into cluster graphs
   ▼
Injected graphs  storage/nx_graphs_injected/nx_graphs_injected_r={route}_n={node}.pkl
   │  graph/convert.py          build edges, convert to PyTorch Geometric `Data`
   ▼
Train / evaluate  scripts/train.py → results/{model}/r{route}_n{node}/
```

**Node features (5-dim)**: `LON, LAT, SPEED, COURSE_SIN, COURSE_COS`

**Edge construction** (`graph/convert.py:create_cluster_edges`)
- Within a track: past → future, fully connected
- Across tracks: node `i` → node `k` of another track with `k ≥ i`, so no future information leaks

**Label (`y`, 3-dim)**: one binary label per track. A track is anomalous when at least
`block_size = 10 × node_ratio` of its nodes are anomalous **consecutively**.

**Injection ratios**

| Argument | Meaning | Values used in the paper |
| --- | --- | --- |
| `route_ratio` | fraction of trajectories turned anomalous | 0.1 / 0.3 / 0.5 |
| `node_ratio` | fraction of consecutive nodes made anomalous inside one trajectory | 0.3 / 0.5 / 0.7 |


## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training)
- ~21 GB of free disk space for the generated graph pickles

### Setup

```bash
# Clone repository
git clone https://github.com/snudial/STGVAD.git
cd STGVAD

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **PyTorch / PyG note**
> `requirements.txt` pins `torch==2.6.0` built for CUDA 12.4. For a different CUDA
> version, install torch and torch_geometric separately:
>
> ```bash
> pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
> pip install torch_geometric==2.6.1
> ```


## Quick Start

All commands are run from the repository root. The `Makefile` sets `PYTHONPATH` for you;
run `make help` to list every target.

### 1. Prepare Data

AIS tracks come from the public **OMTAD (Open Maritime Traffic Analysis Dataset)**, which is
not bundled here because of its size.

```bash
git clone https://github.com/EdithCowan/OMTAD.git data/OMTAD
```

The default root is `data/OMTAD/West Grid`. Point elsewhere with an environment variable:

```bash
export OMTAD_ROOT="/path/to/OMTAD/West Grid"
```

Expected layout:

```
data/OMTAD/West Grid/
├── 2018/
│   ├── cargo/{Jan,Feb,...}/MPF_{Mon}_2018_Grid_Cargo.csv
│   ├── passenger/...
│   └── tanker/...
├── 2019/
└── 2020/
```

Years are configured by `year_list` in `pipeline/generation_runner.py` (2018–2020 by default).

### 2. Build the Original Graphs (run once)

```bash
make cluster
```
→ `storage/nx_graphs_original.pkl` (~0.8 GB)

### 3. Inject Anomalies

```bash
make inject route=0.5 node=0.5   # one combination (~2.3 GB each)
make inject_all                  # all 3 × 3 = 9 combinations (~21 GB total)
```
→ `storage/nx_graphs_injected/nx_graphs_injected_r=0.5_n=0.5.pkl`

### 4. Train

Single run:

```bash
make train model=lstm_gcn route=0.5 node=0.5 seed=42
CUDA_VISIBLE_DEVICES=1 make train model=lstm_gcn route=0.5 node=0.5 seed=42
```

Full sweep for one model (9 ratio combinations × 5 seeds), aggregation included:

```bash
make run model=lstm_gcn gpu=0
# equivalently
bash scripts/run_experiments.sh lstm_gcn 0
```

Seeds and ratios can be overridden through environment variables:

```bash
SEEDS="42" ROUTE_RATIOS="0.5" NODE_RATIOS="0.5" bash scripts/run_experiments.sh lstm 0
```

### 5. Aggregate and Plot

```bash
make tables                                  # cross-model comparison tables → final_results/
python tools/plot_f1_curves.py --save_dir figures
```


## Models

Names accepted by `--model`:

| Name | Architecture |
| --- | --- |
| `mlp` | per-track flattened MLP (graph-free baseline) |
| `lstm` | per-track LSTM (graph-free baseline) |
| `transformer` | per-track Transformer encoder (graph-free baseline) |
| `gat` | 2-layer GAT + mean pooling |
| `lstm_gcn`, `lstm_gat`, `lstm_sage` | 2-layer GNN (+residual) → per-track LSTM |
| `transformer_gcn`, `transformer_gat`, `transformer_sage` | 2-layer GNN (+residual) → per-track Transformer |

Every model takes `(x, edge_index, batch)` and returns a `[B, 3]` tensor of per-track
anomaly probabilities.

### Adding a New Model

1. Write a `torch.nn.Module` in `models/my_model.py` and decorate it with `@register_model("my_model")`.
2. Add one import line to `models/__init__.py` so the decorator runs.
3. Run it:

```bash
make run model=my_model gpu=0
```

`models/registry.py` inspects `__init__` and forwards only the arguments your model declares,
so unused hyperparameters are ignored. If the name contains `_`, the suffix is injected as
`gnn_type` (`lstm_gcn` → `gnn_type="gcn"`).


## Data Format

### Input (AIS CSV)

| Column | Description |
| --- | --- |
| `CRAFT_ID` | vessel identifier |
| `Track_ID` | trajectory identifier |
| `LON`, `LAT` | coordinates |
| `SPEED` | speed over ground |
| `COURSE` | course over ground (degrees) |
| `TIMESTAMP` | datetime |

### Output (Results)

Training artifacts are gitignored and stay local.

```
results/{model}/r{route}_n{node}/
├── results_seed_2.csv        # per-seed test metrics (Metric, Value)
├── results_seed_12.csv
├── ...
├── best_model_s2.pt          # early-stopping checkpoint
└── aggregated_results.csv    # mean ± std across seeds

final_results/
├── F1_score_comparison.csv   # rows = models, columns = r{route}_n{node}
├── AUC_ROC_comparison.csv
├── ...
└── best_results_summary.csv
```

Reported metrics: `Accuracy`, `Precision`, `Recall`, `F1-score`, `Precision@100`,
`BalancedAcc`, `MCC`, `AUC-ROC`, `PR-AUC` (see `train_utils/evaluate.py`).

Training setup: Adam (lr=1e-3), BCELoss, batch size 64, up to 30 epochs,
early stopping on validation loss (patience=5), 7:1.5:1.5 train/val/test split.


## Project Structure

```
STGVAD/
├── data/
│   └── loader.py                    # AIS CSV loading, 1-hour resampling & interpolation
├── cluster/
│   └── optics.py                    # hourly OPTICS clustering
├── graph/
│   ├── builder.py                   # cluster → NetworkX temporal graph
│   ├── inject_anomaly.py            # route-level anomaly injection
│   ├── convert.py                   # edge construction, PyG Data conversion, labeling
│   └── io.py                        # pickle save / load
├── models/
│   ├── registry.py                  # @register_model registry, get_model()
│   ├── mlp_classifier.py
│   ├── lstm.py / transformer.py
│   ├── gat_classifier.py
│   └── gnn_lstm.py / gnn_transformer.py
├── train_utils/
│   ├── early_stopping.py
│   ├── evaluate.py                  # 9 evaluation metrics
│   └── random_seed.py
├── pipeline/
│   ├── generation_runner.py         # load → cluster → build graphs
│   └── injection_runner.py          # inject / load and convert to PyG
├── scripts/
│   ├── cluster_once.py              # make cluster
│   ├── inject_graphs.py             # make inject
│   ├── train.py                     # make train
│   ├── aggregate_results.py         # per-seed results → mean ± std
│   └── run_experiments.sh           # make run (full sweep)
├── tools/
│   ├── make_comparison_tables.py    # make tables
│   └── plot_f1_curves.py            # F1-score curves
├── Makefile
└── requirements.txt
```


## 📖 Citation

If you find this project useful, welcome to cite us.

```bibtex
@article{kim2026stgvad,
  title={STGVAD: Spatio-Temporal Graph-Based Vessel Behavior Anomaly Detection},
  author={Kim, Jeehong and Kim, Minchan and Hwang, Youngseok and Bae, Sungho and Cho, Deuk Jae and Lee, Wonhee and Park, Hyunwoo},
  journal={IEEE Access},
  volume={14},
  pages={2152--2165},
  year={2026},
  doi={10.1109/ACCESS.2025.3609783}
}
```


## Acknowledgements

AIS tracks are taken from [OMTAD — Open Maritime Traffic Analysis Dataset](https://github.com/EdithCowan/OMTAD).


## Contact

For questions or issues, please open an issue on GitHub.
