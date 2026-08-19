import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_metrics(model, loader, device="cpu"):
    model.eval()
    y_true, y_prob = [], []
    with torch.no_grad():
        for batch_data in loader:
            batch_data = batch_data.to(device)
            output = model(batch_data.x, batch_data.edge_index, batch_data.batch)

            #) 튜플이면 첫 번째 텐서 사용, 아니면 그대로
            if isinstance(output, tuple):
                pred_prob = output[0]
            else:
                pred_prob = output

            y_true.extend(batch_data.y.view(-1).cpu().numpy())
            y_prob.extend(pred_prob.view(-1).cpu().numpy())
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    y_pred = (y_prob >= 0.5).astype(int)
    
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    eps = 1e-9
    metrics = {
        "Accuracy": (TP+TN)/(TP+TN+FP+FN+eps),
        "Precision": TP/(TP+FP+eps),
        "Recall": TP/(TP+FN+eps),
        "F1-score": 2*TP/(2*TP+FP+FN+eps),
        "Precision@100": np.sum(y_true[np.argsort(-y_prob)[:min(100, len(y_prob))]]==1)/min(100, len(y_prob)),
        "BalancedAcc": 0.5*((TP/(TP+FN+eps))+(TN/(TN+FP+eps))),
        "MCC": ((TP*TN - FP*FN)/np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)+eps)),
        "AUC-ROC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else float("nan"),
        "PR-AUC": average_precision_score(y_true, y_prob)
    }
    return metrics