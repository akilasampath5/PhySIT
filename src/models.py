
"""
models.py
---------
Physics-Informed Sea Ice Thickness (PhySIT) model architectures.

Models included:
    - NeuralNetwork       : Feedforward physics-guided neural network (PGNN)
    - PhysicsLSTM         : Physics-constrained cuDNN LSTM
    - TransformerNet      : Transformer-based sequence model with positional encoding
    - Spline / KANBlock / KANNet : Kolmogorov-Arnold Network (KAN) with LSTM layers
    - BPINN               : Bayesian Physics-Informed Neural Network wrapper

Reference:
    Sampath et al., "Physics-Informed Machine Learning for Sea Ice Thickness Prediction,"
    IEEE ICKG 2024. https://doi.org/10.1109/ICKG63256.2024.00048
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 1234) -> None:
    """Set random seeds for reproducibility across numpy, torch, and CUDA."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# 1. Physics-Guided Neural Network (PGNN)
# ---------------------------------------------------------------------------

class NeuralNetwork(nn.Module):
    """
    Feedforward physics-guided neural network for sea ice thickness prediction.

    Args:
        input_size  : Number of input features.
        hidden_sizes: List of hidden layer sizes.
        output_size : Number of output targets.
    """

    def __init__(self, input_size: int, hidden_sizes: list, output_size: int):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for i, hidden_size in enumerate(hidden_sizes):
            in_dim = input_size if i == 0 else hidden_sizes[i - 1]
            self.layers.append(nn.Linear(in_dim, hidden_size))
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.tanh(layer(x))
        return self.output_layer(x)


# ---------------------------------------------------------------------------
# 2. Physics-Constrained cuDNN LSTM
# ---------------------------------------------------------------------------

class PhysicsLSTM(nn.Module):
    """
    Two-layer stacked LSTM for physics-constrained sea ice thickness prediction.

    Args:
        input_dim  : Number of input features.
        hidden_dim : Hidden state size for each LSTM layer.
        output_dim : Number of output targets.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(PhysicsLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, bias=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, bias=True, batch_first=True)
        self.fc    = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _  = self.lstm1(x)
        h, _  = self.lstm2(h)
        return self.fc(h[:, -1])          # use last timestep


# ---------------------------------------------------------------------------
# 3. Transformer with Positional Encoding
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for Transformer models.

    Args:
        d_model : Embedding dimension.
        max_len : Maximum sequence length.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(0), :].repeat(1, x.shape[1], 1)


class TransformerNet(nn.Module):
    """
    Transformer encoder for sea ice thickness sequence prediction.

    Args:
        feature_size: Number of input features (also d_model).
        num_layers  : Number of TransformerEncoder layers.
        dropout     : Dropout probability.
    """

    def __init__(self, feature_size: int = 3, num_layers: int = 1, dropout: float = 0.1):
        super(TransformerNet, self).__init__()
        self.pos_encoder       = PositionalEncoding(feature_size)
        encoder_layer          = nn.TransformerEncoderLayer(
            d_model=feature_size, nhead=3, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder           = nn.Linear(feature_size, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.1, 0.1)

    def _generate_mask(self, size: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(size, size, device=device)).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, 0.0)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask   = self._generate_mask(src.size(0), src.device)
        src    = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)
        return self.decoder(output)


# ---------------------------------------------------------------------------
# 4. KAN (Kolmogorov-Arnold Network) with LSTM layers
# ---------------------------------------------------------------------------

class Spline(nn.Module):
    """Learnable spline activation layer."""

    def __init__(self, num_features: int, spline_order: int = 4):
        super(Spline, self).__init__()
        self.spline_order = spline_order
        self.coefficients = nn.Parameter(torch.randn(num_features, spline_order))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.stack([x ** i for i in range(self.spline_order)], dim=-1)
        return torch.sum(basis * self.coefficients, dim=-1)


class KANBlock(nn.Module):
    """Single KAN block: linear projection followed by spline activation."""

    def __init__(self, in_features: int, out_features: int, spline_order: int = 4):
        super(KANBlock, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.spline  = Spline(out_features, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spline(self.linear(x))


class KANNet(nn.Module):
    """
    Hybrid RNN-LSTM-KAN network for ablation experiments.

    Args:
        input_dim     : Number of input features.
        hidden_dim    : Hidden state size.
        output_dim    : Number of output targets.
        num_kan_layers: Number of KAN blocks interleaved with LSTM layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_kan_layers: int = 3,
    ):
        super(KANNet, self).__init__()
        self.rnn        = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.lstm_layers = nn.ModuleList([
            nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            for _ in range(3)
        ])
        self.kan_layers  = nn.ModuleList([
            KANBlock(hidden_dim, hidden_dim) for _ in range(num_kan_layers)
        ])
        self.dropout    = nn.Dropout(0.2)
        self.activation = nn.LeakyReLU()
        self.fc         = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        h = None
        for kan in self.kan_layers:
            for lstm in self.lstm_layers:
                out, h = lstm(out, h)
            out = self.dropout(self.activation(out))
        return self.fc(out[:, -1])


# ---------------------------------------------------------------------------
# 5. BPINN — Bayesian Physics-Informed Neural Network wrapper
# ---------------------------------------------------------------------------

class BPINN(nn.Module):
    """
    Bayesian Physics-Informed Neural Network wrapper.

    Normalizes inputs, applies physics residual loss, and trains
    the wrapped network with combined MSE + physics loss.

    Args:
        x_u, y_u   : Observed data inputs and targets.
        x_f        : Collocation points for physics residual.
        X_star     : Validation inputs.
        u_star     : Validation targets.
        net        : Backbone neural network (NeuralNetwork, TransformerNet, etc.).
        nepochs    : Number of training epochs.
        Omega_mse  : Weight for MSE loss term.
        noise      : Gaussian noise level added to training targets.
    """

    def __init__(
        self,
        x_u, y_u, x_f,
        X_star, u_star,
        net: nn.Module,
        nepochs: int,
        Omega_mse: float,
        noise: float = 0.1,
    ):
        super(BPINN, self).__init__()
        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Normalize using collocation point statistics
        self.Xmean, self.Xstd = x_f.mean(0), x_f.std(0)
        self.x_f       = (x_f    - self.Xmean) / self.Xstd
        self.x_u       = (x_u    - self.Xmean) / self.Xstd
        X_star_norm    = (X_star  - self.Xmean) / self.Xstd

        # Jacobians for PDE normalization
        self.Jacobian_X = 1.0 / self.Xstd[0]
        self.Jacobian_Y = 1.0 / self.Xstd[1]
        self.Jacobian_T = 1.0 / self.Xstd[2]

        self.y_u = y_u + noise * np.std(y_u) * np.random.randn(*y_u.shape)
        self.net = net
        self.net_optim = torch.optim.Adam(net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.nepochs   = nepochs
        self.Omega_mse = Omega_mse

        # Convert to tensors
        self.train_x_u  = torch.tensor(self.x_u,    requires_grad=True).float()
        self.train_y_u  = torch.tensor(self.y_u,    requires_grad=True).float()
        self.train_x_f  = torch.tensor(self.x_f,    requires_grad=True).float()
        self.X_star_norm = torch.tensor(X_star_norm, requires_grad=True).float()
        self.u_star      = torch.tensor(u_star,      requires_grad=True).float()

        self.train_loader = DataLoader(
            list(zip(self.train_x_u, self.train_y_u)),
            batch_size=100, shuffle=True
        )
        self.val_loader = DataLoader(
            list(zip(self.X_star_norm, self.u_star)),
            batch_size=100, shuffle=True
        )

    def phy_residual(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        t: torch.Tensor,
        u: torch.Tensor,
        nu: float = 0.5,
    ) -> torch.Tensor:
        """Compute physics PDE residual via automatic differentiation."""
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        return u_t + u * u_x - nu * u_xx

    def get_residual(self, X: torch.Tensor):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float()
        y = torch.tensor(X[:, 1:2], requires_grad=True).float()
        t = torch.tensor(X[:, 2:3], requires_grad=True).float()
        y_pred = self.net(torch.cat([x, y, t], dim=1))
        u = y_pred[:, 0:1]
        f = self.phy_residual(x, y, t, u)
        return y_pred, f

    def train_model(self, patience: int = 10) -> tuple:
        """
        Train the BPINN with early stopping.

        Returns:
            (train_losses, val_losses) as numpy arrays of length nepochs.
        """
        train_losses = np.zeros(self.nepochs)
        val_losses   = np.zeros(self.nepochs)
        min_val_loss = float("inf")
        no_improve   = 0

        for epoch in range(self.nepochs):
            self.net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in self.train_loader:
                self.net_optim.zero_grad()
                y_pred   = self.net(x_batch)
                mse_loss = F.mse_loss(y_batch, y_pred)
                loss     = self.Omega_mse * mse_loss
                loss.backward()
                self.net_optim.step()
                epoch_loss += loss.item()

            train_losses[epoch] = epoch_loss / len(self.train_loader)

            # Validation
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, y_val in self.val_loader:
                    val_loss += F.mse_loss(y_val, self.net(x_val)).item()
            val_losses[epoch] = val_loss / len(self.val_loader)

            # Early stopping
            if val_losses[epoch] < min_val_loss:
                min_val_loss = val_losses[epoch]
                no_improve   = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d} | Train Loss: {train_losses[epoch]:.6f} | Val Loss: {val_losses[epoch]:.6f}")

        return train_losses, val_losses
