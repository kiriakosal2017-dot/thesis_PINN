import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DataConfig, ColumnConfig, ShipConfig, SequenceConfig, TrainingConfig
from read_data import DataProcessor, create_sequences
from base_model import initialize_weights


class ShipLSTMPredictor(nn.Module):
    """LSTM-based architecture for time-series power prediction."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]  # take output at last time step
        out = torch.relu(self.fc1(last_step))
        out = torch.relu(self.fc2(out))
        return self.fc3(out)


class LSTMPINNModel:
    """Physics-Informed LSTM: recurrent model + Newton's surge equation."""

    KNOTS_TO_MS = 0.51444

    def __init__(self, input_size, hidden_size=128, num_layers=2,
                 lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE',
                 alpha=1.0, beta=0.1):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.loss_function_choice = loss_function_choice
        self.alpha = alpha
        self.beta = beta
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = self._get_device()

        torch.manual_seed(DataConfig.RANDOM_STATE)

        self.model = ShipLSTMPredictor(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        ).to(self.device)

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
            'Adam': lambda: torch.optim.Adam(self.model.parameters(), lr=self.lr),
            'SGD': lambda: torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9),
            'RMSprop': lambda: torch.optim.RMSprop(self.model.parameters(), lr=self.lr),
        }
        if self.optimizer_choice not in optimizers:
            raise ValueError(f"Optimizer '{self.optimizer_choice}' not recognized.")
        return optimizers[self.optimizer_choice]()

    def get_loss_function(self):
        functions = {'MSE': nn.MSELoss, 'MAE': nn.L1Loss}
        if self.loss_function_choice not in functions:
            raise ValueError(f"Loss function '{self.loss_function_choice}' not recognized.")
        return functions[self.loss_function_choice]()

    def prepare_sequence_dataloader(self, X_seq, X_unscaled_seq, y_seq):
        """Create DataLoader from sequence arrays.

        Args:
            X_seq: (N, seq_len, features) scaled
            X_unscaled_seq: (N, seq_len, features) unscaled
            y_seq: (N,) scaled targets
        """
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        X_uns_t = torch.tensor(X_unscaled_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1).to(self.device)

        dataset = TensorDataset(X_t, X_uns_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def calculate_physics_loss(self, X_unscaled_last, predicted_power_scaled,
                               feature_indices, data_processor):
        """Compute surge equation residual at the last time step.

        Newton's law: (M * (1 + k_added)) * a = T_prop - R_total
        Where T_prop = P_pred * eta_D / V
        """
        ship = ShipConfig
        M_eff = ship.MASS * (1.0 + ship.ADDED_MASS_COEFF)

        V_knots = X_unscaled_last[:, feature_indices[ColumnConfig.SPEED]]
        V = V_knots * self.KNOTS_TO_MS
        V = torch.clamp(V, min=0.1)

        trim = (X_unscaled_last[:, feature_indices[ColumnConfig.DRAFT_FORE]]
                - X_unscaled_last[:, feature_indices[ColumnConfig.DRAFT_AFT]])

        accel = X_unscaled_last[:, feature_indices['acceleration']]

        H_s_idx = feature_indices.get(ColumnConfig.WAVE_HEIGHT)
        heading_idx = feature_indices.get(ColumnConfig.HEADING)
        wave_dir_idx = feature_indices.get(ColumnConfig.WAVE_DIRECTION)

        # --- Compute R_total (same as PGNN) ---
        g = torch.tensor(ship.G, device=V.device, dtype=V.dtype)
        L_t = torch.tensor(ship.L_T, device=V.device, dtype=V.dtype)
        L = torch.tensor(ship.L, device=V.device, dtype=V.dtype)
        nu = torch.tensor(ship.NU, device=V.device, dtype=V.dtype)

        Re = torch.clamp(V * L / nu, min=1e-5)
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2

        R_F = 0.5 * ship.RHO * V**2 * ship.S * C_f
        STWAVE2 = 1 + ship.ALPHA_TRIM * trim
        R_W = 0.5 * ship.RHO * V**2 * ship.S * ship.STWAVE1 * STWAVE2
        R_APP = 0.5 * ship.RHO * V**2 * ship.S_APP * C_f
        F_nt = V / torch.sqrt(g * L_t)
        R_TR = 0.5 * ship.RHO * V**2 * ship.A_T * (1 - F_nt)
        R_C = 0.5 * ship.RHO * V**2 * ship.S * ship.C_A

        R_total = R_F * (1 + ship.K) + R_W + R_APP + R_TR + R_C

        if H_s_idx is not None and heading_idx is not None and wave_dir_idx is not None:
            H_s = torch.clamp(X_unscaled_last[:, H_s_idx], min=0.0)
            theta_ship = X_unscaled_last[:, heading_idx]
            theta_wave = X_unscaled_last[:, wave_dir_idx]
            theta_rel = torch.abs(theta_wave - theta_ship) % 360
            theta_rel = torch.where(theta_rel > 180, 360 - theta_rel, theta_rel)
            theta_rel_rad = theta_rel * np.pi / 180
            k_wave = 1e-7
            R_AW = 0.5 * ship.RHO * V**2 * ship.S * k_wave * H_s**2 * (1 + torch.cos(theta_rel_rad))
            R_total = R_total + R_AW

        # --- Inverse-transform predicted power to real scale (kW) ---
        P_pred_np = predicted_power_scaled.detach().cpu().numpy()
        P_pred_kW = data_processor.inverse_transform_y(P_pred_np)
        P_pred_real = torch.tensor(P_pred_kW, dtype=V.dtype, device=V.device)
        P_pred_W = P_pred_real * 1000.0  # kW -> W

        # T_prop = P * eta_D / V
        T_prop = (P_pred_W * ship.ETA_D) / V

        # LHS: (M + m_added) * acceleration
        LHS = M_eff * accel

        # RHS: T_prop - R_total
        RHS = T_prop.squeeze() - R_total

        physics_residual = LHS - RHS
        physics_loss = torch.mean(physics_residual ** 2)

        # Normalize by scale of forces to keep loss balanced
        force_scale = torch.mean(R_total.detach() ** 2) + 1e-8
        physics_loss_normalized = physics_loss / force_scale

        return physics_loss_normalized

    def train(self, train_loader, feature_indices, data_processor,
              val_loader=None, live_plot=False):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        train_losses = []
        val_losses = []

        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            running_data_loss = 0.0
            running_physics_loss = 0.0
            total_batches = len(train_loader)

            progress_bar = tqdm(
                enumerate(train_loader),
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=True,
                total=total_batches,
            )

            for batch_idx, (X_batch, X_uns_batch, y_batch) in progress_bar:
                optimizer.zero_grad()

                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)

                # Physics loss uses unscaled features at the LAST time step
                X_uns_last = X_uns_batch[:, -1, :]

                physics_loss = self.calculate_physics_loss(
                    X_uns_last, outputs, feature_indices, data_processor)

                total_loss = self.alpha * data_loss + self.beta * physics_loss
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()
                running_data_loss += data_loss.item()
                running_physics_loss += physics_loss.item()

                progress_bar.set_postfix({
                    "Total": f"{running_loss / (batch_idx + 1):.6f}",
                    "Data": f"{running_data_loss / (batch_idx + 1):.6f}",
                    "Physics": f"{running_physics_loss / (batch_idx + 1):.6f}",
                })

            avg_loss = running_loss / total_batches
            train_losses.append(avg_loss)

            if val_loader is not None:
                val_loss = self._evaluate_on_loader(val_loader)
                val_losses.append(val_loss)
                print(f"Epoch [{epoch+1}/{self.epochs}], Total: {avg_loss:.6f}, Val: {val_loss:.6f}")
            else:
                val_losses.append(None)
                print(f"Epoch [{epoch+1}/{self.epochs}], Total: {avg_loss:.6f}")

            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch + 2), train_losses, label='Training Loss')
                if val_loader is not None:
                    valid = [v for v in val_losses if v is not None]
                    ax.plot(range(1, len(valid) + 1), valid, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('LSTM-PINN: Training and Validation Loss')
                ax.legend()
                plt.pause(0.01)

        if live_plot:
            plt.ioff()
            plt.show()
            fig.savefig('training_validation_loss_plot_LSTM_PINN.png')

    def _evaluate_on_loader(self, data_loader):
        self.model.eval()
        loss_function = self.get_loss_function()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, _, y_batch in data_loader:
                outputs = self.model(X_batch)
                loss = loss_function(outputs, y_batch)
                running_loss += loss.item()
        return running_loss / len(data_loader)

    def evaluate(self, X_seq, y_seq, dataset_type="Test", data_processor=None):
        self.model.eval()
        X_t = torch.tensor(X_seq, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1).to(self.device)

        loss_function = self.get_loss_function()
        with torch.no_grad():
            outputs = self.model(X_t)
            loss = loss_function(outputs, y_t)
            print(f"\n{dataset_type} Loss: {loss.item():.8f}")

            if data_processor:
                outputs_orig = data_processor.inverse_transform_y(outputs.cpu().numpy())
                y_orig = data_processor.inverse_transform_y(y_t.cpu().numpy())
                rmse = np.sqrt(np.mean((outputs_orig - y_orig) ** 2))
                print(f"{dataset_type} RMSE: {rmse:.4f}")

        return loss.item()

    def cross_validate(self, X_seq, X_uns_seq, y_seq, feature_indices,
                       data_processor, k_folds=5):
        kfold = KFold(n_splits=k_folds, shuffle=False)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_seq)):
            print(f"\nFold {fold+1}/{k_folds}")

            train_loader = self.prepare_sequence_dataloader(
                X_seq[train_idx], X_uns_seq[train_idx], y_seq[train_idx])
            val_loader = self.prepare_sequence_dataloader(
                X_seq[val_idx], X_uns_seq[val_idx], y_seq[val_idx])

            self._reinit_model(X_seq.shape[2])

            self.train(train_loader, feature_indices, data_processor,
                       val_loader=val_loader, live_plot=False)

            val_loss = self._evaluate_on_loader(val_loader)
            fold_results.append(val_loss)
            print(f"Fold {fold+1} Validation Loss: {val_loss:.8f}")

        avg = np.mean(fold_results)
        print(f"\nCross-validation Average Loss: {avg:.8f}")
        return avg

    def _reinit_model(self, input_size):
        self.model = ShipLSTMPredictor(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
        ).to(self.device)

    @staticmethod
    def hyperparameter_search(X_seq, X_uns_seq, y_seq, feature_indices,
                              param_grid, data_processor, k_folds=5):
        best_params = None
        best_loss = float('inf')

        combos = list(product(
            param_grid['lr'],
            param_grid['batch_size'],
            param_grid['alpha'],
            param_grid['beta'],
        ))

        for lr, batch_size, alpha, beta in combos:
            print(f"\nTesting: lr={lr}, batch={batch_size}, alpha={alpha}, beta={beta}")

            model = LSTMPINNModel(
                input_size=X_seq.shape[2],
                lr=lr,
                epochs=TrainingConfig.EPOCHS_CV,
                batch_size=batch_size,
                optimizer_choice=TrainingConfig.OPTIMIZER,
                loss_function_choice=TrainingConfig.LOSS_FUNCTION,
                alpha=alpha,
                beta=beta,
            )

            avg_loss = model.cross_validate(
                X_seq, X_uns_seq, y_seq, feature_indices, data_processor, k_folds=k_folds)

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_params = {'lr': lr, 'batch_size': batch_size,
                               'alpha': alpha, 'beta': beta}

        print(f"\nBest params: {best_params}, loss: {best_loss:.8f}")

        with open("best_hyperparameters_LSTM_PINN.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss


def _build_feature_indices(columns):
    """Build and validate feature name -> column index mapping."""
    feature_indices = {col: idx for idx, col in enumerate(columns)}
    required = [
        ColumnConfig.SPEED, ColumnConfig.DRAFT_FORE, ColumnConfig.DRAFT_AFT,
        'acceleration', 'dt',
    ]
    for col in required:
        if col not in feature_indices:
            raise ValueError(f"Required column '{col}' not found in data")
    return feature_indices


if __name__ == "__main__":
    # --- Load temporal data ---
    data_processor = DataProcessor()
    result = data_processor.load_and_prepare_temporal_data()

    if result is None:
        print("Failed to load temporal data.")
        exit(1)

    X_train, X_test, X_train_unscaled, X_test_unscaled, \
        y_train, y_test, y_train_unscaled, y_test_unscaled = result

    print(f"X_train shape: {X_train.shape}")
    print(f"Columns: {list(X_train.columns)}")

    feature_indices = _build_feature_indices(X_train.columns)

    # --- Create sequences ---
    seq_len = SequenceConfig.LENGTH
    print(f"\nCreating sequences with length={seq_len}...")

    X_train_seq, X_train_uns_seq, y_train_seq = create_sequences(
        X_train, X_train_unscaled, y_train, seq_length=seq_len)
    X_test_seq, X_test_uns_seq, y_test_seq = create_sequences(
        X_test, X_test_unscaled, y_test, seq_length=seq_len)

    print(f"Train sequences: {X_train_seq.shape}, Test sequences: {X_test_seq.shape}")

    if len(X_train_seq) == 0:
        print("No valid sequences created. Check data continuity / MAX_TIME_GAP setting.")
        exit(1)

    # --- Hyperparameter search ---
    param_grid = {
        'lr': [0.001, 0.0005],
        'batch_size': [64, 128],
        'alpha': [1.0],
        'beta': [0.01, 0.1],
    }

    best_params, best_loss = LSTMPINNModel.hyperparameter_search(
        X_train_seq, X_train_uns_seq, y_train_seq,
        feature_indices, param_grid, data_processor, k_folds=3,
    )

    # --- Final training ---
    n_val = int(len(X_train_seq) * DataConfig.TEST_SIZE)
    X_final_train = X_train_seq[:-n_val]
    X_final_uns_train = X_train_uns_seq[:-n_val]
    y_final_train = y_train_seq[:-n_val]

    X_final_val = X_train_seq[-n_val:]
    X_final_uns_val = X_train_uns_seq[-n_val:]
    y_final_val = y_train_seq[-n_val:]

    final_model = LSTMPINNModel(
        input_size=X_train_seq.shape[2],
        lr=best_params['lr'],
        epochs=TrainingConfig.EPOCHS_FINAL,
        batch_size=best_params['batch_size'],
        optimizer_choice=TrainingConfig.OPTIMIZER,
        loss_function_choice=TrainingConfig.LOSS_FUNCTION,
        alpha=best_params['alpha'],
        beta=best_params['beta'],
    )

    train_loader = final_model.prepare_sequence_dataloader(
        X_final_train, X_final_uns_train, y_final_train)
    val_loader = final_model.prepare_sequence_dataloader(
        X_final_val, X_final_uns_val, y_final_val)

    final_model.train(
        train_loader, feature_indices, data_processor,
        val_loader=val_loader, live_plot=True,
    )

    # --- Test evaluation ---
    final_model.evaluate(X_test_seq, y_test_seq, dataset_type="Test",
                         data_processor=data_processor)
