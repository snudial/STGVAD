"""
이상치가 주입된 클러스터 그래프로 모델 하나를 학습하고 테스트 지표를 CSV로 남긴다.

    python scripts/train.py --model lstm_gcn --route_ratio 0.5 --node_ratio 0.5 --seed 42

결과 저장 위치:
    results/{model}/r{route}_n{node}/results_seed_{seed}.csv   # 테스트 지표
    results/{model}/r{route}_n{node}/best_model_s{seed}.pt     # early stopping 체크포인트
"""
import argparse
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from models.registry import get_model
from pipeline.injection_runner import load_injected_graphs_and_convert
from train_utils.early_stopping import EarlyStopping
from train_utils.evaluate import evaluate_metrics
from train_utils.random_seed import seed_all


def train_model(model_name, pyg_data, seed, output_dir, epochs=30, device="cpu"):
    train_data, test_valid_data = train_test_split(pyg_data, test_size=0.3, random_state=seed)
    val_data, test_data = train_test_split(test_valid_data, test_size=0.5, random_state=seed)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    model = get_model(
        model_name,
        input_dim=5,
        gnn_hidden=16,
        lstm_hidden=32,
        transformer_hidden=32,
        residual=True,
        hidden_dim=64,
        num_labels=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.BCELoss()
    model_path = os.path.join(output_dir, f"best_model_s{seed}.pt")
    stopper = EarlyStopping(patience=5, path=model_path)

    for ep in range(1, epochs + 1):
        model.train()
        total_loss = total_correct = total_examples = 0
        for batch in tqdm(train_loader, desc=f"[Epoch {ep:02d}] train", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_prob = model(batch.x, batch.edge_index, batch.batch)
            y_true = batch.y.view(-1, 3)
            loss = criterion(pred_prob, y_true)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += ((pred_prob >= 0.5).float() == y_true).float().sum().item()
            total_examples += y_true.numel()

        avg_train_loss = total_loss / len(train_loader)
        avg_train_acc = total_correct / total_examples

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_prob = model(batch.x, batch.edge_index, batch.batch)
                y_true = batch.y.view(-1, 3)
                val_loss += criterion(pred_prob, y_true).item()
                val_correct += ((pred_prob >= 0.5).float() == y_true).float().sum().item()
                val_total += y_true.numel()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / val_total

        print(
            f"[Epoch {ep:02d}] "
            f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {avg_val_acc:.4f}"
        )

        stopper(avg_val_loss, model)
        if stopper.early_stop:
            print(f"Early stopping at epoch {ep}")
            break

    model.load_state_dict(torch.load(model_path))
    test_metrics = evaluate_metrics(model.to(device), test_loader, device=device)

    csv_path = os.path.join(output_dir, f"results_seed_{seed}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        for k, v in test_metrics.items():
            writer.writerow([k, v])

    print(f"[INFO] 결과 저장 완료: {csv_path}")
    return test_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="예: lstm, transformer, lstm_gcn, transformer_gat")
    parser.add_argument("--route_ratio", type=float, default=0.5)
    parser.add_argument("--node_ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()

    seed_all(args.seed)

    out_dir = f"results/{args.model}/r{args.route_ratio}_n{args.node_ratio}"
    os.makedirs(out_dir, exist_ok=True)

    pyg_data = load_injected_graphs_and_convert(
        route_ratio=args.route_ratio,
        node_ratio=args.node_ratio,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(args.model, pyg_data, args.seed, out_dir, epochs=args.epochs, device=device)


if __name__ == "__main__":
    main()
