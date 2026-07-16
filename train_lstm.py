# train_lstm.py  (patched - stratified train/val split)
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from data.keypoint_dataset import KeypointDataset
from models.pose_lstm import PoseLSTM
import numpy as np

# stratified split
try:
    from sklearn.model_selection import StratifiedShuffleSplit
except Exception as e:
    raise ImportError("scikit-learn is required for stratified split. Install with: pip install scikit-learn") from e

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * x.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total

def eval_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)
    return total_loss / total, correct / total

def create_stratified_splits(ds, test_size=0.2, seed=42):
    """
    Returns (train_indices, val_indices) using StratifiedShuffleSplit.
    ds.samples contains tuples (filename, label).
    """
    labels = [label for _, label in ds.samples]
    # convert labels to numeric indices consistent with ds.label2idx
    y = np.array([ds.label2idx[l] for l in labels])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y)), y))
    return train_idx.tolist(), val_idx.tolist()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels_csv", default="data/labels.csv")
    parser.add_argument("--keypoint_dir", default="data/keypoints")
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--save_path", default="checkpoints/best_model.pth")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    ds = KeypointDataset(args.labels_csv, args.keypoint_dir, seq_len=args.seq_len)

    if len(ds) < 2:
        raise RuntimeError("Dataset too small. Need at least a few samples per class.")

    # Create stratified train/val splits
    train_idx, val_idx = create_stratified_splits(ds, test_size=args.val_split, seed=args.seed)
    print(f"Total samples: {len(ds)}  Train: {len(train_idx)}  Val: {len(val_idx)}")

    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=0)

    num_classes = len(ds.label2idx)
    model = PoseLSTM(feature_dim=225, hidden=args.hidden, num_layers=args.num_layers, num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, opt, loss_fn, device)
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "labels": ds.label2idx,
                "epoch": epoch,
                "val_acc": val_acc
            }, args.save_path)
            print("Saved best model ->", args.save_path)

if __name__ == "__main__":
    main()
