# evaluate_esm_model.py

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from models.esm.model import SiameseMLP


def evaluate_model(model: SiameseMLP, test_loader: DataLoader) -> None:
    device = torch.device("cuda")
    model.eval()

    # Evaluación
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            logits = model(x1, x2)
            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).long()

            all_labels.extend(y.tolist())
            all_preds.extend(preds.tolist())
            all_probs.extend(probs.tolist())

    # Métricas
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test F1 Score : {f1:.4f}")
    print(f"Test ROC AUC  : {auc:.4f}")
