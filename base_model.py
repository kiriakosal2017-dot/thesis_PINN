import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from config import DataConfig, TrainingConfig


def initialize_weights(model):
    """Apply Kaiming uniform initialization for consistent comparison across models."""
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
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

        layers = []
        in_features = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, int(hidden_size)))
            layers.append(nn.ReLU())
            in_features = int(hidden_size)
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

        torch.manual_seed(DataConfig.RANDOM_STATE)

        self.model = ShipSpeedPredictor(input_size, hidden_layers=self.hidden_layers).to(self.device)
        initialize_weights(self.model)

    @staticmethod
    def _get_device():
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
        if self.optimizer_choice == 'LBFGS':
            return data_length
        return self.batch_size

    def prepare_dataloader(self, X, y):
        X_tensor = torch.tensor(X.values, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self._effective_batch_size(len(X)),
            shuffle=False, num_workers=0
        )

    def prepare_unscaled_dataloader(self, X_unscaled):
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
