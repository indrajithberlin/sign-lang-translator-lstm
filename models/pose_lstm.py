# models/pose_lstm.py
import torch
import torch.nn as nn

class PoseLSTM(nn.Module):
    def __init__(self, feature_dim=225, hidden=128, num_layers=2, num_classes=5, dropout=0.2):
        super().__init__()
        self.fc_in = nn.Linear(feature_dim, hidden)
        self.lstm = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden//2, num_classes)
        )

    def forward(self, x):
        # x: (B, T, F)
        b, t, f = x.shape
        x = self.fc_in(x)          # (B, T, hidden)
        out, (h, c) = self.lstm(x) # out: (B, T, hidden)
        feat = out.mean(dim=1)     # temporal average -> (B, hidden)
        logits = self.classifier(feat)
        return logits

