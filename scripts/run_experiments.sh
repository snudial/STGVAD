#!/bin/bash
# 한 모델에 대해 (route_ratio × node_ratio × seed) 전체 조합을 학습하고 결과를 집계한다.
#
#   bash scripts/run_experiments.sh <model> [gpu_id]
#
# 예)
#   bash scripts/run_experiments.sh lstm_gcn 0
#
# 하이퍼파라미터는 환경변수로 덮어쓸 수 있다.
#   SEEDS="42" ROUTE_RATIOS="0.5" NODE_RATIOS="0.5" bash scripts/run_experiments.sh lstm 0
set -u

MODEL="${1:-}"
GPU_ID="${2:-0}"

if [ -z "$MODEL" ]; then
  echo "usage: bash scripts/run_experiments.sh <model> [gpu_id]"
  echo "models: lstm transformer mlp gat lstm_gcn lstm_gat lstm_sage transformer_gcn transformer_gat transformer_sage"
  exit 1
fi

SEEDS="${SEEDS:-2 12 22 32 42}"
ROUTE_RATIOS="${ROUTE_RATIOS:-0.1 0.3 0.5}"
NODE_RATIOS="${NODE_RATIOS:-0.3 0.5 0.7}"

echo "Starting experiments for $MODEL on GPU $GPU_ID"
echo "==============================================="

for route in $ROUTE_RATIOS; do
  for node in $NODE_RATIOS; do
    combo_dir="results/$MODEL/r${route}_n${node}"
    mkdir -p "$combo_dir"
    echo "Training $MODEL with route=$route, node=$node"

    for seed in $SEEDS; do
      echo "  seed: $seed"
      if CUDA_VISIBLE_DEVICES=$GPU_ID python scripts/train.py \
        --model "$MODEL" \
        --route_ratio "$route" \
        --node_ratio "$node" \
        --seed "$seed"; then
        echo "  done seed $seed"
      else
        echo "  FAILED seed $seed"
      fi
    done

    echo "  aggregating r${route}_n${node}"
    python scripts/aggregate_results.py \
      --input_dir "$combo_dir" \
      --output_file "$combo_dir/aggregated_results.csv"
    echo "  ----------------------------"
  done
done

echo "All experiments completed for $MODEL"
echo "Results saved in: results/$MODEL/"
