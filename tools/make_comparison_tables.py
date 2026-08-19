"""
모델별 집계 결과(results/{model}/r{route}_n{node}/aggregated_results.csv)를 모아
지표별 비교 테이블을 만든다.

    python tools/make_comparison_tables.py

출력:
    final_results/{Metric}_comparison.csv   # 행=모델, 열=r{route}_n{node}
    final_results/best_results_summary.csv  # 조합별 최고 성능 모델
"""
import argparse
import os

import numpy as np
import pandas as pd

MODELS = [
    "lstm",
    "transformer",
    "lstm_gcn",
    "lstm_gat",
    "lstm_sage",
    "transformer_gcn",
    "transformer_gat",
    "transformer_sage",
]
ROUTE_RATIOS = [0.1, 0.3, 0.5]
NODE_RATIOS = [0.3, 0.5, 0.7]


def collect(results_dir, models):
    """{model: {combo: {metric: "mean ± std"}}} 형태로 집계 결과를 읽어온다."""
    all_results = {}
    for model in models:
        model_results = {}
        for route in ROUTE_RATIOS:
            for node in NODE_RATIOS:
                combo = f"r{route}_n{node}"
                path = os.path.join(results_dir, model, combo, "aggregated_results.csv")
                if not os.path.exists(path):
                    print(f"  [SKIP] {model}/{combo}: {path} 없음")
                    model_results[combo] = {}
                    continue
                df = pd.read_csv(path)
                model_results[combo] = dict(zip(df["Metric"], df["Result"]))
                print(f"  [OK]   {model}/{combo}: {len(model_results[combo])} metrics")
        all_results[model] = model_results
    return all_results


def build_table(all_results, metric):
    combos = [f"r{r}_n{n}" for r in ROUTE_RATIOS for n in NODE_RATIOS]
    rows = []
    for model, model_results in all_results.items():
        row = {"Model": model}
        for combo in combos:
            row[combo] = model_results.get(combo, {}).get(metric, "N/A")
        rows.append(row)
    return pd.DataFrame(rows)[["Model"] + combos]


def best_summary(tables):
    """지표 × 조합마다 평균값이 가장 높은 모델을 뽑는다."""
    rows = []
    for metric, table in tables.items():
        for combo in [c for c in table.columns if c != "Model"]:
            means, models = [], []
            for _, row in table.iterrows():
                value = str(row[combo])
                if "±" not in value:
                    continue
                try:
                    means.append(float(value.split("±")[0]))
                    models.append(row["Model"])
                except ValueError:
                    continue
            if not means:
                continue
            best = int(np.argmax(means))
            rows.append(
                {
                    "Metric": metric,
                    "Combination": combo,
                    "Best_Model": models[best],
                    "Best_Value": means[best],
                    "Full_Result": table.loc[table["Model"] == models[best], combo].iloc[0],
                }
            )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results")
    parser.add_argument("--output_dir", default="final_results")
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()

    print("Collecting aggregated results...")
    all_results = collect(args.results_dir, args.models)

    metrics = sorted({m for r in all_results.values() for c in r.values() for m in c})
    if not metrics:
        raise SystemExit(
            f"{args.results_dir} 아래에서 aggregated_results.csv 를 찾지 못했습니다. "
            "먼저 scripts/run_experiments.sh 로 실험을 돌리세요."
        )
    print(f"Found metrics: {', '.join(metrics)}")

    os.makedirs(args.output_dir, exist_ok=True)
    tables = {m: build_table(all_results, m) for m in metrics}
    for metric, table in tables.items():
        safe = metric.replace("-", "_").replace("@", "_at_").replace(" ", "_")
        path = os.path.join(args.output_dir, f"{safe}_comparison.csv")
        table.to_csv(path, index=False)
        print(f"  saved {path}")

    summary_path = os.path.join(args.output_dir, "best_results_summary.csv")
    best_summary(tables).to_csv(summary_path, index=False)
    print(f"  saved {summary_path}")

    for metric in ["Accuracy", "F1-score", "AUC-ROC"]:
        if metric in tables:
            print(f"\n{metric}")
            print("-" * 80)
            print(tables[metric].to_string(index=False))


if __name__ == "__main__":
    main()
