import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tqdm import tqdm
from models.esm.model import SiameseMLP

def train_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_save_path: str,
    input_dim: int = 320,
    hidden_dim: int = 512,
    lr: float = 1e-3,
    epochs: int = 10,
    patience: int = 3,
) -> SiameseMLP:

    device = torch.device("cuda")

    model = SiameseMLP(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    best_model_path = f"{model_save_path}/esm_siamese_best.pth"

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        print(f"\nEpoch {epoch + 1}/{epochs}")

        for x1, x2, y in tqdm(train_loader, desc="Training", leave=False):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            optimizer.zero_grad()
            with autocast("cuda"):
                logits = model(x1, x2)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Train loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x1, x2, y in tqdm(val_loader, desc="Validation", leave=False):
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                with autocast("cuda"):
                    logits = model(x1, x2)
                    loss = criterion(logits, y)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) >= 0.5).long()
                correct += (preds == y.long()).sum().item()
                total += y.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total
        print(f"Val loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

        # Early stopping
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

        # Checkpoint por época
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "val_loss": avg_val_loss
        }, f"{model_save_path}/esm_siamese_ckpt_epoch{epoch + 1}.pth")

    torch.save(model.state_dict(), f"{model_save_path}/esm_siamese_final.pth")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    return model
