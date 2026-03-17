"""
boom_challenge/src/models/pinn_model.py
=========================================
Physics-Informed Neural Network (PINN) for predicting P80 and R95.

Architecture:
  - Multi-layer residual neural network
  - GELU activations + BatchNorm
  - Dual output heads (P80, R95 in log-space)
  - Physics constraint loss (monotonicity + power-law prior)
  - Monte Carlo Dropout for uncertainty quantification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# ─────────────────────────────────────────────────────────────
# RESIDUAL BLOCK
# ─────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Residual block with BatchNorm, GELU, and Dropout."""

    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


# ─────────────────────────────────────────────────────────────
# PINN MODEL
# ─────────────────────────────────────────────────────────────

class ImpactPINN(nn.Module):
    """
    Physics-Informed Neural Network for impact ejecta prediction.

    Input:  Feature vector (raw + physics features, standardized)
    Output: [log_P80, log_R95] predictions
    """

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [256, 256, 128, 64],
                 dropout: float = 0.15):
        super().__init__()

        self.input_bn = nn.BatchNorm1d(input_dim)

        # Entry projection
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual core
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dims[0], dropout=dropout)
            for _ in range(2)
        ])

        # Downsampling layers
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ]
        self.neck = nn.Sequential(*layers)

        # Dual output heads
        out_dim = hidden_dims[-1]
        self.head_P80 = nn.Linear(out_dim, 1)   # log_P80
        self.head_R95 = nn.Linear(out_dim, 1)   # log_R95

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Returns tensor of shape [batch, 2]: columns are [log_P80, log_R95]
        """
        x = self.input_bn(x)
        x = self.entry(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.neck(x)
        log_P80 = self.head_P80(x)
        log_R95 = self.head_R95(x)
        return torch.cat([log_P80, log_R95], dim=1)


# ─────────────────────────────────────────────────────────────
# PHYSICS LOSS
# ─────────────────────────────────────────────────────────────

def physics_regularization_loss(model: ImpactPINN,
                                X: torch.Tensor,
                                ke_idx: int,
                                lambda_phys: float = 0.1) -> torch.Tensor:
    """
    Enforce physics monotonicity constraints via gradient penalties:
      - dP80/d(log_KE)  should be ≤ 0 (higher energy → finer fragments)
      - dR95/d(log_KE)  should be ≥ 0 (higher energy → wider scatter)

    Parameters
    ----------
    model    : ImpactPINN
    X        : input tensor requiring grad [batch, n_features]
    ke_idx   : column index of log_KE in X
    lambda_phys : weight of physics loss term

    Returns
    -------
    physics loss scalar tensor
    """
    X_req = X.clone().detach().requires_grad_(True)
    preds = model(X_req)          # [batch, 2]
    log_P80 = preds[:, 0].sum()
    log_R95 = preds[:, 1].sum()

    grad_P80 = torch.autograd.grad(log_P80, X_req,
                                   create_graph=True)[0][:, ke_idx]
    grad_R95 = torch.autograd.grad(log_R95, X_req,
                                   create_graph=True)[0][:, ke_idx]

    # Penalize violations
    loss_P80_mono = torch.relu(grad_P80).mean()   # should be ≤ 0
    loss_R95_mono = torch.relu(-grad_R95).mean()  # should be ≥ 0

    return lambda_phys * (loss_P80_mono + loss_R95_mono)


# ─────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────

class PINNTrainer:
    """End-to-end training wrapper for ImpactPINN."""

    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [256, 256, 128, 64],
                 dropout: float = 0.15,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 lambda_phys: float = 0.05,
                 ke_feature_idx: int = 0,
                 device: str = 'cpu'):
        self.device = torch.device(device)
        self.lambda_phys = lambda_phys
        self.ke_idx = ke_feature_idx

        self.model = ImpactPINN(input_dim, hidden_dims, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=50)
        self.history = {'train_loss': [], 'val_loss': []}

    def _mse_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(preds, targets)

    def fit(self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            epochs: int = 300,
            batch_size: int = 256,
            patience: int = 40,
            verbose: bool = True) -> 'PINNTrainer':
        """
        Train the PINN with early stopping.

        y_train / y_val: [n, 2] array of [log_P80, log_R95]
        """
        # Convert to tensors
        Xt = torch.FloatTensor(X_train).to(self.device)
        yt = torch.FloatTensor(y_train).to(self.device)
        Xv = torch.FloatTensor(X_val).to(self.device)
        yv = torch.FloatTensor(y_val).to(self.device)

        dataset = TensorDataset(Xt, yt)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_val_loss = float('inf')
        no_improve    = 0
        best_state    = None

        for epoch in range(epochs):
            # ── Train
            self.model.train()
            train_losses = []
            for Xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(Xb)
                loss  = self._mse_loss(preds, yb)
                # Physics regularization every 5 epochs (expensive)
                if epoch % 5 == 0 and self.lambda_phys > 0:
                    loss += physics_regularization_loss(
                        self.model, Xb, self.ke_idx, self.lambda_phys)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            self.scheduler.step()

            # ── Validate
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(Xv)
                val_loss  = self._mse_loss(val_preds, yv).item()

            self.history['train_loss'].append(np.mean(train_losses))
            self.history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                best_state    = {k: v.cpu().clone()
                                 for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | train={np.mean(train_losses):.5f}"
                      f" | val={val_loss:.5f} | best={best_val_loss:.5f}")

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict({k: v.to(self.device)
                                        for k, v in best_state.items()})
        return self

    def predict(self, X: np.ndarray, mc_samples: int = 1) -> np.ndarray:
        """
        Predict log-space targets.

        Parameters
        ----------
        X          : feature array [n, d]
        mc_samples : if > 1, uses Monte Carlo Dropout for uncertainty

        Returns
        -------
        np.ndarray [n, 2] — columns: [log_P80, log_R95]
        """
        Xt = torch.FloatTensor(X).to(self.device)

        if mc_samples <= 1:
            self.model.eval()
            with torch.no_grad():
                return self.model(Xt).cpu().numpy()
        else:
            self.model.train()  # keep dropout active
            preds = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    preds.append(self.model(Xt).cpu().numpy())
            return np.stack(preds, axis=0).mean(axis=0)

    def predict_with_uncertainty(self, X: np.ndarray,
                                  mc_samples: int = 100) -> tuple:
        """Returns (mean_pred, std_pred) for uncertainty quantification."""
        Xt = torch.FloatTensor(X).to(self.device)
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(self.model(Xt).cpu().numpy())
        preds = np.stack(preds, axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str):
        torch.save({'model_state': self.model.state_dict(),
                    'history': self.history}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.history = ckpt.get('history', {})
        return self


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    torch.manual_seed(42)
    n, d = 500, 40
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randn(n, 2).astype(np.float32)

    trainer = PINNTrainer(input_dim=d, hidden_dims=[128, 128, 64],
                          dropout=0.1, lr=5e-4, lambda_phys=0.0)
    trainer.fit(X[:400], y[:400], X[400:], y[400:],
                epochs=100, batch_size=64, patience=20, verbose=True)

    preds = trainer.predict(X[400:])
    print(f"Prediction shape: {preds.shape}")
    print(f"Sample predictions:\n{preds[:5]}")
