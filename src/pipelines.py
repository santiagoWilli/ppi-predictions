import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_save_path: str,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 3,
) -> nn.Module:
    device = torch.device("cuda")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(model_save_path, "_best.pth")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for x1, x2, y in tqdm(train_loader, desc="Training", leave=False):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x1, x2)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc="Validation", leave=False):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                logits = model(x1, x2)
                loss = criterion(logits, y)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) > 0.5).long()
                correct += (preds == y.long()).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Val loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print("-- Best model saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"!! Early stopping at epoch {epoch + 1}")
                break

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": avg_val_loss
        }, os.path.join(model_save_path, f"_ckpt_epoch{epoch + 1}.pth"))

    torch.save(model.state_dict(), os.path.join(model_save_path, "_final.pth"))
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    return model

def evaluate_model(model, test_loader, device) -> None:
    model.eval()
    y_true = []
    y_pred = []
    probs_all = []

    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            logits = model(x1, x2)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            y_true.extend(y.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            probs_all.extend(probs.cpu().numpy())

    # Metrics
    acc = sum([yt == yp for yt, yp in zip(y_true, y_pred)]) / len(y_true)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, probs_all)

    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test F1 Score : {f1:.4f}")
    print(f"Test ROC AUC  : {roc_auc:.4f}\n")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, probs_all)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], '--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.grid(False)
    plt.tight_layout()
    plt.show()
