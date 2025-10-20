import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# ------------------------------------------------------------------
# Evaluate model on test data
# ------------------------------------------------------------------
def evaluate(model, loader, device, detailed=False):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1).cpu().numpy()
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds)

    # --- Basic accuracy ---
    acc = np.mean(np.array(y_true) == np.array(y_pred))

    if not detailed:
        return acc

    # --- Detailed metrics ---
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    except ValueError:
        # fallback for multi-class case
        tn = fp = fn = tp = 0

    return acc, prec, rec, f1, tp, fp, tn, fn


# ------------------------------------------------------------------
# Manual computation for binary classification metrics
# ------------------------------------------------------------------
def compute_classification_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = int(np.sum((y_true == 1) & (y_pred == 1)))
    TN = int(np.sum((y_true == 0) & (y_pred == 0)))
    FP = int(np.sum((y_true == 0) & (y_pred == 1)))
    FN = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (TP + TN) / len(y_true)
    precision = TP / (TP + FP + 1e-12)
    recall = TP / (TP + FN + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return TP, FP, TN, FN, acc, precision, recall, f1
