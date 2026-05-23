from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class ReliabilityNet(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.reliable_head = nn.Linear(hidden_dim, 1)
        self.failure_head = nn.Linear(hidden_dim, 5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.reliable_head(h).squeeze(-1), self.failure_head(h)


def focal_bce(logits: torch.Tensor, targets: torch.Tensor, gamma: float) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    prob = torch.sigmoid(logits)
    pt = prob * targets + (1.0 - prob) * (1.0 - targets)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def train_reliability_net(
    x_train: np.ndarray,
    y_train: np.ndarray,
    f_train: np.ndarray,
    x_val: np.ndarray,
    cfg: dict,
    device: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ReliabilityNet(x_train.shape[1], cfg["hidden_dim"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(f_train, dtype=torch.float32),
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)

    model.train()
    for _ in range(cfg["epochs"]):
        for xb, yb, fb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            fb = fb.to(device, non_blocking=True)
            rel_logits, fail_logits = model(xb)
            rel_loss = focal_bce(rel_logits, yb, cfg["focal_gamma"])
            fail_loss = nn.functional.binary_cross_entropy_with_logits(fail_logits, fb)
            prob = torch.sigmoid(rel_logits)
            cal_loss = torch.mean((prob - yb).pow(2))
            loss = rel_loss + cfg["failure_loss_weight"] * fail_loss + cfg["calibration_loss_weight"] * cal_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    return predict_reliability_net(model, x_val, device)


@torch.no_grad()
def predict_reliability_net(model: ReliabilityNet, x: np.ndarray, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    fail_preds = []
    loader = DataLoader(torch.tensor(x, dtype=torch.float32), batch_size=16384, shuffle=False)
    for xb in loader:
        rel_logits, fail_logits = model(xb.to(device))
        preds.append(torch.sigmoid(rel_logits).cpu().numpy())
        fail_preds.append(torch.sigmoid(fail_logits).cpu().numpy())
    return np.concatenate(preds), np.vstack(fail_preds)

