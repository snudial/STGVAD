"""
final_results/F1_score_comparison.csv 를 읽어
route ratio별로 node ratio에 따른 F1-score 곡선을 그린다 (LSTM 계열 / Transformer 계열 각 1장).

    python tools/make_comparison_tables.py      # 먼저 비교 테이블 생성
    python tools/plot_f1_curves.py --save_dir figures
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

ROUTE_RATIOS = [0.1, 0.3, 0.5]
NODE_RATIOS = [0.3, 0.5, 0.7]

# 파일 안의 모델 키 -> 그래프 범례 이름
LSTM_FAMILY = {
    "lstm": "LSTM",
    "lstm_gcn": "LSTM+GCN",
    "lstm_gat": "LSTM+GAT",
    "lstm_sage": "LSTM+GraphSAGE",
}
TRANSFORMER_FAMILY = {
    "transformer": "Transformer",
    "transformer_gcn": "Transformer+GCN",
    "transformer_gat": "Transformer+GAT",
    "transformer_sage": "Transformer+GraphSAGE",
}
COLORS = ["black", "orangered", "orange", "forestgreen"]
MARKERS = ["o", "^", "s", "*"]


def parse_mean(value):
    """'0.823 ± 0.021' -> 0.823"""
    try:
        return float(str(value).split("±")[0])
    except ValueError:
        return np.nan


def plot_family(df, family, save_path=None):
    fig, axes = plt.subplots(1, len(ROUTE_RATIOS), figsize=(24, 7), sharey=True)
    fig.tight_layout(pad=4.0)

    for col_idx, route_ratio in enumerate(ROUTE_RATIOS):
        ax = axes[col_idx]
        for i, (key, label) in enumerate(family.items()):
            row = df[df["Model"] == key]
            if row.empty:
                print(f"[WARN] '{key}' 행이 없습니다. 건너뜁니다.")
                continue
            means = [parse_mean(row[f"r{route_ratio}_n{n}"].iloc[0]) for n in NODE_RATIOS]
            ax.plot(
                NODE_RATIOS,
                means,
                marker=MARKERS[i % len(MARKERS)],
                markersize=16,
                linewidth=3,
                color=COLORS[i % len(COLORS)],
                label=label,
            )

        ax.set_ylim(0.7, 1.0)
        ax.set_title(f"Trajectory Anomaly Ratio = {route_ratio}", fontsize=28, pad=20)
        ax.set_xlabel("Node Anomaly Ratio", fontsize=28)
        if col_idx == 0:
            ax.set_ylabel("F1-score", fontsize=28, labelpad=15)
        ax.set_xticks(NODE_RATIOS)
        ax.tick_params(axis="both", which="major", labelsize=28)
        ax.grid(True)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))

    fig.legend(
        list(family.values()),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=4,
        fontsize=28,
    )
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=200)
        print(f"[INFO] 저장: {save_path}")
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="final_results/F1_score_comparison.csv")
    parser.add_argument("--save_dir", default=None, help="지정하면 PNG로 저장하고, 없으면 화면에 띄운다")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f"{args.input} 가 없습니다. 먼저 `python tools/make_comparison_tables.py` 를 실행하세요."
        )
    df = pd.read_csv(args.input)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    plot_family(
        df, LSTM_FAMILY,
        os.path.join(args.save_dir, "f1_lstm_family.png") if args.save_dir else None,
    )
    plot_family(
        df, TRANSFORMER_FAMILY,
        os.path.join(args.save_dir, "f1_transformer_family.png") if args.save_dir else None,
    )

    if not args.save_dir:
        plt.show()


if __name__ == "__main__":
    main()
