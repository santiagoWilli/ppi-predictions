import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from models.convolutive_siamese_model import ConvolutiveSiameseModel

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    vocab_size: int,
    model_save_path: str,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 1e-3,
    epochs: int = 30,
    patience: int = 3,
) -> ConvolutiveSiameseModel:

    device = torch.device("cuda")

    model = ConvolutiveSiameseModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(model_save_path, "convolutive_siamese_best.pth")

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
        }, os.path.join(model_save_path, f"convolutive_siamese_ckpt_epoch{epoch + 1}.pth"))

    torch.save(model.state_dict(), os.path.join(model_save_path, "convolutive_siamese_final.pth"))
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    return model
