# data/keypoint_dataset.py
import os
import numpy as np
from torch.utils.data import Dataset
import torch

class KeypointDataset(Dataset):
    """
    Expects:
      - labels_csv: CSV with rows "filename.npy,label"
      - keypoint_dir: folder containing those .npy files
      - seq_len: length of the sequences (used earlier in preprocess)
    Returns (x, y):
      x: torch.FloatTensor (seq_len, feature_dim)  e.g. (32,225)
      y: torch.LongTensor scalar (label index)
    """
    def __init__(self, labels_csv, keypoint_dir, seq_len=32):
        self.keypoint_dir = keypoint_dir
        self.seq_len = seq_len

        # read csv
        self.samples = []
        with open(labels_csv, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) != 2:
                    continue
                filename, label = parts
                self.samples.append((filename, label))

        # labels -> indices
        labels = sorted(list({label for _, label in self.samples}))
        self.label2idx = {lab: i for i, lab in enumerate(labels)}
        self.idx2label = {i: lab for lab, i in self.label2idx.items()}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.keypoint_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Keypoint file not found: {path}")
        arr = np.load(path)  # shape (seq_len, feature_dim)

        # safety: ensure correct temporal length
        if arr.shape[0] != self.seq_len:
            if arr.shape[0] < self.seq_len:
                pad = np.zeros((self.seq_len - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            else:
                idxs = np.linspace(0, arr.shape[0]-1, self.seq_len).astype(int)
                arr = arr[idxs]

        x = torch.tensor(arr, dtype=torch.float32)     # (T, F)
        y = torch.tensor(self.label2idx[label], dtype=torch.long)
        return x, y
