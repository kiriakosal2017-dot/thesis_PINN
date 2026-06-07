"""Shared MLP architecture and training infrastructure used by all tabular model variants.

Defines the ``ShipSpeedPredictor`` network, the ``BaseModel`` training harness, global
seeding, and device selection so individual model scripts contain only variant-specific
logic.
"""
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import DataConfig, TrainingConfig


def set_global_seed(seed):
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs.

    Seeding NumPy matters here because the HYBRID model draws collocation/boundary
    points with ``np.random``; seeding torch covers weight init and DataLoader
    shuffling. Call this with a different seed per run for multi-seed experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # CUDA uses a separate per-device seed pool; set all at once to cover multi-GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def initialize_weights(model):
    """Apply Kaiming uniform initialization for consistent comparison across models."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            # Kaiming uniform is variance-consistent for ReLU activations and reduces
            # sensitivity to depth when comparing shallow vs. deep MLP configs.
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)


class ShipSpeedPredictor(nn.Module):
    """Shared configurable MLP architecture used by tabular model variants."""

    def __init__(self, input_size, hidden_layers=None):
        super().__init__()
        hidden_layers = hidden_layers or [128, 64, 32, 16]
        if not hidden_layers:
            raise ValueError("hidden_layers must contain at least one hidden size")

        # Build the layer stack dynamically so depth/width is controlled from config
        # without subclassing for each variant.
        layers = []
        in_features = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ReLU())
            in_features = int(hidden_size)
        # Final linear projects to a single scalar (shaft power in kW).
        layers.append(nn.Linear(in_features, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BaseModel:
    """Base class with shared training infrastructure for all model variants."""

    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE',
                 hidden_layers=None):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.loss_function_choice = loss_function_choice
        self.hidden_layers = hidden_layers or [128, 64, 32, 16]
        self.device = self._get_device()

        # Seed here (not at script entry) so subclass constructors that create
        # additional tensors also start from a deterministic state.
        set_global_seed(DataConfig.RANDOM_STATE)

        self.model = ShipSpeedPredictor(input_size, hidden_layers=self.hidden_layers).to(self.device)
        initialize_weights(self.model)

    @staticmethod
    def _get_device():
        # Prefer CUDA, fall back to MPS (Apple Silicon), then CPU.
        # MPS is checked separately because it is unavailable on Linux/Windows
        # and raises no error when simply not present.
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using NVIDIA GPU with CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Silicon GPU with MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
        return device

    def get_optimizer(self):
        # Lazy construction via lambdas avoids instantiating every optimizer
        # when only one is needed.
        optimizers = {
            'Adam': lambda: optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=TrainingConfig.WEIGHT_DECAY
            ),
            'SGD': lambda: optim.SGD(
                self.model.parameters(), lr=self.lr, momentum=0.9,
                weight_decay=TrainingConfig.WEIGHT_DECAY
            ),
            'RMSprop': lambda: optim.RMSprop(
                self.model.parameters(), lr=self.lr, weight_decay=TrainingConfig.WEIGHT_DECAY
            ),
            # LBFGS is a full-batch second-order method; strong_wolfe line search
            # improves convergence for the physics residual loss used in PI-NODE.
            'LBFGS': lambda: optim.LBFGS(
                self.model.parameters(), lr=self.lr,
                max_iter=20, history_size=10, line_search_fn="strong_wolfe"
            ),
        }
        if self.optimizer_choice not in optimizers:
            raise ValueError(f"Optimizer '{self.optimizer_choice}' not recognized. "
                             f"Available: {list(optimizers.keys())}")
        return optimizers[self.optimizer_choice]()

    def get_loss_function(self):
        functions = {
            'MSE': nn.MSELoss,
            'MAE': nn.L1Loss,
        }
        if self.loss_function_choice not in functions:
            raise ValueError(f"Loss function '{self.loss_function_choice}' not recognized. "
                             f"Available: {list(functions.keys())}")
        return functions[self.loss_function_choice]()

    def _effective_batch_size(self, data_length):
        # LBFGS requires the full dataset per step to compute accurate curvature;
        # mini-batches break its line search guarantees.
        if self.optimizer_choice == 'LBFGS':
            return data_length
        return self.batch_size

    def prepare_dataloader(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        # shuffle=False preserves the chronological order of ship voyage data,
        # which matters for time-series integrity during evaluation.
        return DataLoader(
            dataset,
            batch_size=self._effective_batch_size(len(X)),
            shuffle=False, num_workers=0
        )

    def prepare_unscaled_dataloader(self, X_unscaled):
        # Unscaled features are fed to the physics branch (propeller law, resistance
        # curves) which require dimensional inputs in SI / original units.
        X_tensor = torch.tensor(X_unscaled.values, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor)
        return DataLoader(
            dataset,
            batch_size=self._effective_batch_size(len(X_unscaled)),
            shuffle=False, num_workers=0
        )

    def evaluate(self, X_eval, y_eval, dataset_type="Validation", data_processor=None):
        self.model.eval()
        X_eval_tensor = torch.tensor(X_eval.values, dtype=torch.float32).to(self.device)
        y_eval_tensor = torch.tensor(y_eval.values, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_eval_tensor)
            loss = loss_function(outputs, y_eval_tensor)
            print(f"\n{dataset_type} Loss: {loss.item():.8f}")

            # Inverse-transform to kW so the printed RMSE is interpretable
            # without knowing the scaling parameters.
            if data_processor:
                outputs_original = data_processor.inverse_transform_y(outputs.cpu().numpy())
                y_eval_original = data_processor.inverse_transform_y(y_eval_tensor.cpu().numpy())
                rmse = np.sqrt(np.mean((outputs_original - y_eval_original) ** 2))
                print(f"{dataset_type} RMSE: {rmse:.4f}")

        return loss.item()

    @staticmethod
    def reset_weights(m):
        if isinstance(m, nn.Linear):
            m.reset_parameters()
