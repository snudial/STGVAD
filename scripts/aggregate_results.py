"""
한 실험 조합(results/{model}/r{route}_n{node}) 안의 seed별 지표 CSV를 모아
평균 ± 표준편차 형태로 집계한다.

    python scripts/aggregate_results.py \
        --input_dir results/lstm_gcn/r0.5_n0.5 \
        --output_file results/lstm_gcn/r0.5_n0.5/aggregated_results.csv
"""
import argparse
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def aggregate(input_dir):
    files = sorted(glob.glob(os.path.join(input_dir, "results_seed_*.csv")))
    if not files:
        raise FileNotFoundError(f"{input_dir} 안에 results_seed_*.csv 가 없습니다.")

    values = defaultdict(list)
    for path in files:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            values[row["Metric"]].append(float(row["Value"]))

    rows = []
    for metric in sorted(values):
        arr = np.array(values[metric], dtype=float)
        mean = float(np.mean(arr))
        # 시드 간 표본표준편차 (ddof=1). 시드가 하나뿐이면 0으로 둔다.
        std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        rows.append(
            {
                "Metric": metric,
                "Mean": round(mean, 3),
                "Std": round(std, 3),
                "Result": f"{mean:.3f} ± {std:.3f}",
                "N_Seeds": len(arr),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="seed별 CSV가 들어 있는 디렉터리")
    parser.add_argument("--output_file", required=True, help="집계 결과를 저장할 CSV 경로")
    args = parser.parse_args()

    df = aggregate(args.input_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    df.to_csv(args.output_file, index=False)

    print(df.to_string(index=False))
    print(f"[INFO] 집계 결과 저장: {args.output_file}")


if __name__ == "__main__":
    main()
