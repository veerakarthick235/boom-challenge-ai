"""
boom_challenge/src/models/pinn_model.py
=========================================
Physics-Informed Neural Network (PINN) for predicting 6 targets.
Uses L1 (MAE) loss to match competition scoring.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

class ResidualBlock(nn.Module):
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

class ImpactPINN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256, 128, 64], dropout: float = 0.15):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        self.entry = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dims[0], dropout=dropout) for _ in range(2)])
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers += [
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout * 0.5),
            ]
        self.neck = nn.Sequential(*layers)
        out_dim = hidden_dims[-1]
        
        # Changed to output all 6 targets
        self.head_out = nn.Linear(out_dim, 6) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.entry(x)
        for block in self.res_blocks:
            x = block(x)
        x = self.neck(x)
        return self.head_out(x) # Returns [batch, 6]

def physics_regularization_loss(model: ImpactPINN, X: torch.Tensor, ke_idx: int, lambda_phys: float = 0.1) -> torch.Tensor:
    X_req = X.clone().detach().requires_grad_(True)
    preds = model(X_req)
    
    # Physics constraints only apply to P80 (idx 0) and R95 (idx 1)
    log_P80 = preds[:, 0].sum()
    log_R95 = preds[:, 1].sum()

    grad_P80 = torch.autograd.grad(log_P80, X_req, create_graph=True)[0][:, ke_idx]
    grad_R95 = torch.autograd.grad(log_R95, X_req, create_graph=True)[0][:, ke_idx]

    loss_P80_mono = torch.relu(grad_P80).mean()
    loss_R95_mono = torch.relu(-grad_R95).mean()

    return lambda_phys * (loss_P80_mono + loss_R95_mono)

class PINNTrainer:
    def __init__(self, input_dim: int, hidden_dims: list = [256, 256, 128, 64], dropout: float = 0.15,
                 lr: float = 1e-3, weight_decay: float = 1e-4, lambda_phys: float = 0.05,
                 ke_feature_idx: int = 0, device: str = 'cpu'):
        self.device = torch.device(device)
        self.lambda_phys = lambda_phys
        self.ke_idx = ke_feature_idx

        self.model = ImpactPINN(input_dim, hidden_dims, dropout).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=50)
        self.history = {'train_loss': [], 'val_loss': []}

    def _l1_loss(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Changed to L1Loss (MAE) to match rubric
        return F.l1_loss(preds, targets)

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
            epochs: int = 300, batch_size: int = 256, patience: int = 40, verbose: bool = True) -> 'PINNTrainer':
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
            self.model.train()
            train_losses = []
            for Xb, yb in loader:
                self.optimizer.zero_grad()
                preds = self.model(Xb)
                loss  = self._l1_loss(preds, yb) # Using L1 Loss
                if epoch % 5 == 0 and self.lambda_phys > 0:
                    loss += physics_regularization_loss(self.model, Xb, self.ke_idx, self.lambda_phys)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                train_losses.append(loss.item())

            self.scheduler.step()
            self.model.eval()
            with torch.no_grad():
                val_preds = self.model(Xv)
                val_loss  = self._l1_loss(val_preds, yv).item() # Using L1 Loss

            self.history['train_loss'].append(np.mean(train_losses))
            self.history['val_loss'].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve    = 0
                best_state    = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1

            if verbose and epoch % 50 == 0:
                print(f"Epoch {epoch:4d} | train={np.mean(train_losses):.5f} | val={val_loss:.5f} | best={best_val_loss:.5f}")

            if no_improve >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        if best_state:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})
        return self

    def predict(self, X: np.ndarray, mc_samples: int = 1) -> np.ndarray:
        Xt = torch.FloatTensor(X).to(self.device)
        if mc_samples <= 1:
            self.model.eval()
            with torch.no_grad():
                return self.model(Xt).cpu().numpy()
        else:
            self.model.train()
            preds = []
            with torch.no_grad():
                for _ in range(mc_samples):
                    preds.append(self.model(Xt).cpu().numpy())
            return np.stack(preds, axis=0).mean(axis=0)

    def predict_with_uncertainty(self, X: np.ndarray, mc_samples: int = 100) -> tuple:
        Xt = torch.FloatTensor(X).to(self.device)
        self.model.train()
        preds = []
        with torch.no_grad():
            for _ in range(mc_samples):
                preds.append(self.model(Xt).cpu().numpy())
        preds = np.stack(preds, axis=0)
        return preds.mean(axis=0), preds.std(axis=0)

    def save(self, path: str):
        torch.save({'model_state': self.model.state_dict(), 'history': self.history}, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state'])
        self.history = ckpt.get('history', {})
        return self

if __name__ == '__main__':
    torch.manual_seed(42)
    n, d = 500, 40
    X = np.random.randn(n, d).astype(np.float32)
    y = np.random.randn(n, 6).astype(np.float32) # Updated to 6 targets

    trainer = PINNTrainer(input_dim=d, hidden_dims=[128, 128, 64], dropout=0.1, lr=5e-4, lambda_phys=0.0)
    trainer.fit(X[:400], y[:400], X[400:], y[400:], epochs=100, batch_size=64, patience=20, verbose=True)
    preds = trainer.predict(X[400:])
    print(f"Prediction shape: {preds.shape}")