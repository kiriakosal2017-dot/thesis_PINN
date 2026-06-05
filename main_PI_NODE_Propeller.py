"""Physics-Informed Neural ODE with Propeller Hydrodynamics (Hard-Physics).

This model introduces a paradigm shift: instead of trying to model the entire
ship resistance (which relies on inaccurate empirical ITTC-78 formulas),
it models the propeller kinematics and dynamics.

The Neural ODE predicts the 'Wake Fraction' (w), which is the unmeasurable
environmental dynamic. The rest of the Power calculation is purely analytical
and based on the Propeller Law and Hydrodynamics.

Architecture:
    Encoder(x_0) → z_0  ─→  ODEIntegrate(dz/dt = f_θ(z; ctx))  ─→  Decoder(z_T) → w (Wake Fraction)
    Then:
    Va = V_ship * (1 - w)
    J = Va / (n * D)
    KQ = c0 + c1*J + c2*J^2 + c3*J^3 (where c_i are trainable parameters)
    Q = KQ * rho * n^2 * D^5
    P = 2 * pi * n * Q
"""

import copy
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from torchdiffeq import odeint as _odeint
from torchdiffeq import odeint_adjoint as _odeint_adjoint

from config import (
    DataConfig, ColumnConfig, ShipConfig, SequenceConfig,
    TrainingConfig, ModelConfig, PropellerConfig
)
from base_model import set_global_seed
from read_data import DataProcessor, create_sequences, split_calm_weather_indices


# ── Neural ODE building blocks ────────────────────────────────────

class _ODEFunc(nn.Module):
    def __init__(self, hidden_size, context_size, num_layers=2):
        super().__init__()
        # Ensure context_size matches what we pass
        in_dim = hidden_size + context_size
        layers = []
        prev = in_dim
        for _ in range(num_layers):
            layers.extend([nn.Linear(prev, hidden_size), nn.Tanh()])
            prev = hidden_size
        self.net = nn.Sequential(*layers)
        self._ctx = None

    def set_context(self, ctx):
        self._ctx = ctx

    def forward(self, t, z):
        return self.net(torch.cat([z, self._ctx], dim=-1))


class _Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3), # Outputs [w, t, eta_R]
        )

    def forward(self, z):
        return self.net(z)


class PropellerNeuralODEPredictor(nn.Module):
    def __init__(self, input_size, calm_water_indices, weather_indices, hidden_size=64, ode_num_layers=2,
                 dropout=0.1, solver='rk4', use_adjoint=False,
                 n_ode_steps=10, encoder_mode='first', use_ode=True, use_weather=True):
        super().__init__()
        self.encoder_mode = encoder_mode
        self.calm_water_indices = calm_water_indices
        self.weather_indices = weather_indices
        # Ablation switches (defaults preserve the full PI-NODE behaviour):
        #   use_ode=False     -> skip the Neural ODE integration (decode z0 directly),
        #                        turning Branch A into a plain encoder baseline.
        #   use_weather=False -> disable the Sea-State residual branch (no [dw,dt,deta_R]).
        self.use_ode = use_ode
        self.use_weather = use_weather
        
        calm_input_size = len(calm_water_indices)
        weather_input_size = len(weather_indices)
        
        # Branch A: Calm-Water ODE
        self.encoder = _Encoder(calm_input_size, hidden_size, dropout)
        self.gru_encoder = nn.GRU(
            input_size=calm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        ) if encoder_mode == 'gru' else None
        self.ode_func = _ODEFunc(hidden_size, calm_input_size, ode_num_layers)
        self.decoder = _Decoder(hidden_size, dropout) # Outputs [w_calm, t_calm, eta_R_calm]
        
        # Branch B: Weather Residual
        self.weather_residual_net = nn.Sequential(
            nn.Linear(weather_input_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3) # Outputs [dw, dt, deta_R]
        )
        
        self.solver = solver
        self.use_adjoint = use_adjoint
        self.n_ode_steps = n_ode_steps
        
        # Trainable coefficients for KQ polynomial
        self.kq_c0 = nn.Parameter(torch.tensor(0.06))
        self.kq_c1 = nn.Parameter(torch.tensor(-0.04))
        self.kq_c2 = nn.Parameter(torch.tensor(-0.01))
        self.kq_c3 = nn.Parameter(torch.tensor(0.00))

        # Trainable coefficients for KT polynomial
        self.kt_b0 = nn.Parameter(torch.tensor(0.60))
        self.kt_b1 = nn.Parameter(torch.tensor(-0.40))
        self.kt_b2 = nn.Parameter(torch.tensor(-0.10))
        self.kt_b3 = nn.Parameter(torch.tensor(0.00))

    def forward(self, x_seq):
        # Split inputs
        x_calm = x_seq[:, :, self.calm_water_indices]
        x_weather_last = x_seq[:, -1, self.weather_indices] # Only use the last timestep for weather residual
        
        # --- Branch A: Calm-Water ODE ---
        ctx = x_calm.mean(dim=1) # Context is the mean of calm features over the sequence
        
        if self.encoder_mode == 'first':
            z0 = self.encoder(x_calm[:, 0, :])
        elif self.encoder_mode == 'last_mean':
            z0 = self.encoder(x_calm[:, -1, :])
        elif self.encoder_mode == 'gru':
            _, h_n = self.gru_encoder(x_calm)
            z0 = torch.tanh(h_n[-1])
        else:
            raise ValueError(
                f"Unknown encoder_mode='{self.encoder_mode}'. "
                "Supported modes: 'first', 'last_mean', 'gru'."
            )

        if self.use_ode:
            self.ode_func.set_context(ctx)

            t_span = torch.linspace(0.0, 1.0, self.n_ode_steps + 1,
                                    device=z0.device, dtype=z0.dtype)

            if self.use_adjoint:
                adjoint_params = tuple(self.ode_func.parameters()) + (ctx,)
                z_traj = _odeint_adjoint(
                    self.ode_func, z0, t_span,
                    method=self.solver,
                    adjoint_params=adjoint_params,
                )
            else:
                z_traj = _odeint(
                    self.ode_func, z0, t_span,
                    method=self.solver,
                )

            z_final = z_traj[-1]
        else:
            # Ablation: no ODE — decode the encoded latent state directly.
            z_final = z0

        out_calm = self.decoder(z_final)
        
        # Transform calm outputs
        w_calm = 0.05 + 0.40 * torch.sigmoid(out_calm[:, 0:1])
        t_calm = 0.05 + 0.25 * torch.sigmoid(out_calm[:, 1:2])
        eta_R_calm = 0.95 + 0.15 * torch.sigmoid(out_calm[:, 2:3])
        
        # --- Branch B: Weather Residual ---
        if self.use_weather:
            out_residual = self.weather_residual_net(x_weather_last)

            # Residuals are unbounded but typically small, initialized around 0
            dw = out_residual[:, 0:1] * 0.1  # scaling factor to keep initial predictions small
            dt = out_residual[:, 1:2] * 0.1
            deta_R = out_residual[:, 2:3] * 0.05
        else:
            # Ablation: no Sea-State residual branch.
            dw = torch.zeros_like(w_calm)
            dt = torch.zeros_like(t_calm)
            deta_R = torch.zeros_like(eta_R_calm)
        
        # --- Total ---
        w_total = w_calm + dw
        t_total = t_calm + dt
        eta_R_total = eta_R_calm + deta_R
        
        # Constrain totals to reasonable physics bounds
        w_total = torch.clamp(w_total, min=0.01, max=0.60)
        t_total = torch.clamp(t_total, min=0.01, max=0.40)
        eta_R_total = torch.clamp(eta_R_total, min=0.80, max=1.20)
        
        return w_total, t_total, eta_R_total


# ── Trainer / Evaluator ───────────────────────────────────────────

class PINODEPropellerModel:
    KNOTS_TO_MS = 0.51444
    # Soft physics regularization defaults (kept intentionally mild).
    KQ_MIN = 0.0
    KQ_MAX = 0.12
    KQ_PRIOR = (0.06, -0.04, -0.01, 0.00)
    LAMBDA_KQ_RANGE = 0.5
    LAMBDA_KQ_CURVATURE = 0.01
    LAMBDA_KQ_PRIOR = 0.001

    def __init__(self, input_size, feature_indices, calm_water_indices, weather_indices, data_processor, hidden_size=64, ode_num_layers=2,
                 dropout=0.1, solver='rk4', use_adjoint=False,
                 n_ode_steps=10, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE',
                 weight_decay=0.0, seed=None, encoder_mode='first',
                 use_ode=True, use_weather=True, freeze_polynomials=False):
        self.input_size = input_size
        self.feature_indices = feature_indices
        self.calm_water_indices = calm_water_indices
        self.weather_indices = weather_indices
        self.data_processor = data_processor
        self.hidden_size = hidden_size
        self.ode_num_layers = ode_num_layers
        self.dropout = dropout
        self.solver = solver
        self.use_adjoint = use_adjoint
        self.n_ode_steps = n_ode_steps
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer_choice = optimizer_choice
        self.loss_function_choice = loss_function_choice
        self.weight_decay = weight_decay
        self.encoder_mode = encoder_mode
        self.use_ode = use_ode
        self.use_weather = use_weather
        self.freeze_polynomials = freeze_polynomials
        self.device = self._get_device()

        self.seed = DataConfig.RANDOM_STATE if seed is None else seed
        set_global_seed(self.seed)

        self.model = PropellerNeuralODEPredictor(
            input_size=input_size,
            calm_water_indices=calm_water_indices,
            weather_indices=weather_indices,
            hidden_size=hidden_size,
            ode_num_layers=ode_num_layers,
            dropout=dropout,
            solver=solver,
            use_adjoint=use_adjoint,
            n_ode_steps=n_ode_steps,
            encoder_mode=encoder_mode,
            use_ode=use_ode,
            use_weather=use_weather,
        )

        # Ablation: optionally freeze the trainable K_T/K_Q polynomial coefficients
        # at their B-series initialization (tests whether learnable physics helps).
        if freeze_polynomials:
            for name, param in self.model.named_parameters():
                if name.startswith('kq_') or name.startswith('kt_'):
                    param.requires_grad_(False)

        self._move_model_to_device_with_fallback()

    def _move_model_to_device_with_fallback(self):
        try:
            self.model = self.model.to(self.device)
        except RuntimeError as exc:
            msg = str(exc).lower()
            if self.device.type == "mps" and "out of memory" in msg:
                print("MPS OOM while moving model – falling back to CPU.")
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
                self.device = torch.device("cpu")
                self.model = self.model.to(self.device)
            else:
                raise

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def get_optimizer(self):
        # Apply a mildly larger LR to KQ and KT polynomial coefficients for stability.
        base_params = [p for n, p in self.model.named_parameters() if not (n.startswith('kq_') or n.startswith('kt_'))]
        poly_params = [p for n, p in self.model.named_parameters() if (n.startswith('kq_') or n.startswith('kt_'))]
        
        param_groups = [
            {'params': base_params},
            {'params': poly_params, 'lr': self.lr * 1.5}
        ]
        
        if self.optimizer_choice == 'Adam':
            return torch.optim.Adam(param_groups, lr=self.lr, weight_decay=self.weight_decay)
        raise ValueError(f"Optimizer '{self.optimizer_choice}' not recognized.")

    def get_loss_function(self):
        if self.loss_function_choice == 'MSE':
            return nn.MSELoss()
        if self.loss_function_choice in ('SmoothL1', 'Huber'):
            return nn.SmoothL1Loss(beta=0.5)
        raise ValueError(f"Loss function '{self.loss_function_choice}' not recognized.")

    def prepare_sequence_dataloader(self, X_seq, X_unscaled_seq, y_seq, shuffle=False):
        X_t = torch.tensor(X_seq, dtype=torch.float32)
        X_uns_t = torch.tensor(X_unscaled_seq, dtype=torch.float32)
        y_t = torch.tensor(y_seq, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_t, X_uns_t, y_t)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=0)

    def compute_analytical_power(self, w_pred, t_pred, eta_R_pred, X_uns_last):
        """
        Computes the analytical predicted power based on wake fraction, thrust deduction,
        relative rotative efficiency, and propeller laws.
        """
        # Fetch required features from the unscaled data
        # We need RPM and Speed
        # Note: If 'Propeller-Shaft-RPM' is not in feature_indices, we need to adapt.
        # But 'Propeller-Shaft-RPM' was passed to data processor. Let's assume it is.
        rpm_idx = self.feature_indices.get('Propeller-Shaft-RPM')
        if rpm_idx is None:
            raise ValueError("Propeller-Shaft-RPM is required for Propeller Physics model")
            
        speed_idx = self.feature_indices[ColumnConfig.SPEED]
        
        rpm = torch.clamp(X_uns_last[:, rpm_idx], min=10.0).view(-1, 1) # Avoid div by zero
        n = rpm / 60.0 # revolutions per second
        
        V_knots = torch.clamp(X_uns_last[:, speed_idx], min=1.0).view(-1, 1)
        V_ship = V_knots * self.KNOTS_TO_MS # m/s
        
        # 1. Advance Speed of water to propeller
        Va = V_ship * (1.0 - w_pred)
        
        # 2. Advance Coefficient J
        D = PropellerConfig.D
        J = Va / (n * D)
        
        # 3. Torque Coefficient KQ & Thrust Coefficient KT using learned polynomials
        c0, c1, c2, c3 = self.model.kq_c0, self.model.kq_c1, self.model.kq_c2, self.model.kq_c3
        KQ = c0 + c1*J + c2*(J**2) + c3*(J**3)
        
        b0, b1, b2, b3 = self.model.kt_b0, self.model.kt_b1, self.model.kt_b2, self.model.kt_b3
        KT = b0 + b1*J + b2*(J**2) + b3*(J**3)
        
        # 4. Open Water Efficiency (eta_0)
        # Avoid division by zero by adding a small epsilon to KQ
        eta_0 = (J * KT) / (2.0 * np.pi * (KQ + 1e-8))
        
        # 5. Torque Q (in Nm)
        rho = ShipConfig.RHO
        Q_pred_Nm = KQ * rho * (n**2) * (D**5)
        
        # 6. Power P (in Watts and kW)
        P_pred_W = 2.0 * np.pi * n * Q_pred_Nm
        P_pred_kW = P_pred_W / 1000.0
        
        return P_pred_kW, KQ, KT, J, eta_0

    def _scale_power_torch(self, P_pred_kW):
        """Scales kW power to the normalized space of the NN targets."""
        scaler = self.data_processor.scaler_y
        y_mean = torch.tensor(scaler.mean_[0], dtype=torch.float32, device=self.device)
        y_std = torch.tensor(scaler.scale_[0], dtype=torch.float32, device=self.device)
        return (P_pred_kW - y_mean) / y_std

    def train(self, train_loader, val_loader=None, live_plot=False,
              metrics_output_path=None, checkpoint_path="best_model_PI_NODE_Propeller.pt",
              history_csv=None):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            min_lr=self.lr * 0.05,
        )

        train_losses, val_losses, val_rmses = [], [], []
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        patience = TrainingConfig.EARLY_STOPPING_PATIENCE
        min_delta = TrainingConfig.EARLY_STOPPING_MIN_DELTA

        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            
            progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}")
            for batch_idx, (X_batch, X_uns_batch, y_batch) in progress_bar:
                optimizer.zero_grad()
                X_batch = X_batch.to(self.device)
                X_uns_batch = X_uns_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # 1. Neural ODE predicts wake fraction w, thrust deduction t, and eta_R
                w_pred, t_pred, eta_R_pred = self.model(X_batch)
                
                # 2. Physics Equations convert to P_pred_kW
                X_uns_last = X_uns_batch[:, -1, :]
                P_pred_kW, KQ, KT, J, eta_0 = self.compute_analytical_power(w_pred, t_pred, eta_R_pred, X_uns_last)
                
                # 3. Scale P_pred_kW to match y_batch (which is scaled)
                P_pred_scaled = self._scale_power_torch(P_pred_kW)
                
                # 4. Pure MSE Loss on the Power
                loss = loss_function(P_pred_scaled, y_batch)
                
                # Soft physics regularization:
                # 1) keep KQ and KT in realistic range,
                # 2) avoid excessive polynomial curvature,
                # 3) keep coefficients near B-series prior (soft, not hard clamp).
                # 4) maintain open water efficiency (eta_0) in a realistic bound [0.4, 0.75]
                kq_range_penalty = torch.mean(
                    torch.relu(self.KQ_MIN - KQ) ** 2 +
                    torch.relu(KQ - self.KQ_MAX) ** 2
                )
                kt_range_penalty = torch.mean(
                    torch.relu(0.0 - KT) ** 2 +  # KT must be positive
                    torch.relu(KT - 1.0) ** 2    # Upper bound
                )

                c0, c1, c2, c3 = self.model.kq_c0, self.model.kq_c1, self.model.kq_c2, self.model.kq_c3
                b0, b1, b2, b3 = self.model.kt_b0, self.model.kt_b1, self.model.kt_b2, self.model.kt_b3
                
                kq_curvature = 2.0 * c2 + 6.0 * c3 * J
                kt_curvature = 2.0 * b2 + 6.0 * b3 * J
                curvature_penalty = torch.mean(kq_curvature ** 2 + kt_curvature ** 2)

                p0, p1, p2, p3 = self.KQ_PRIOR
                kq_prior_penalty = (
                    (c0 - p0) ** 2 + (c1 - p1) ** 2 + (c2 - p2) ** 2 + (c3 - p3) ** 2
                )
                
                # eta_0 should be roughly between 0.4 and 0.75 for a normal propeller
                eta_0_penalty = torch.mean(
                    torch.relu(0.4 - eta_0) ** 2 + 
                    torch.relu(eta_0 - 0.75) ** 2
                )

                total_loss = (
                    loss +
                    self.LAMBDA_KQ_RANGE * (kq_range_penalty + kt_range_penalty) +
                    self.LAMBDA_KQ_CURVATURE * curvature_penalty +
                    self.LAMBDA_KQ_PRIOR * kq_prior_penalty +
                    0.1 * eta_0_penalty # Weight for eta_0 realistic bounds
                )

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()

                running_loss += total_loss.item()
                progress_bar.set_postfix({"Loss": f"{running_loss/(batch_idx+1):.4f}"})

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if val_loader is not None:
                val_loss, val_rmse = self.evaluate_loader(val_loader)
                val_losses.append(val_loss)
                val_rmses.append(val_rmse)
                scheduler.step(val_loss)
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f} kW")
                print(f"         | Current LR = {current_lr:.6f}")
                
                # Print current learned KQ polynomial
                c0, c1, c2, c3 = [p.item() for p in [self.model.kq_c0, self.model.kq_c1, self.model.kq_c2, self.model.kq_c3]]
                b0, b1, b2, b3 = [p.item() for p in [self.model.kt_b0, self.model.kt_b1, self.model.kt_b2, self.model.kt_b3]]
                print(f"         | Learned KQ = {c0:.4f} + ({c1:.4f})J + ({c2:.4f})J^2 + ({c3:.4f})J^3")
                print(f"         | Learned KT = {b0:.4f} + ({b1:.4f})J + ({b2:.4f})J^2 + ({b3:.4f})J^3")

                if (best_val_loss - val_loss) > min_delta:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                    if checkpoint_path:
                        torch.save(best_state, checkpoint_path)
                else:
                    epochs_without_improvement += 1
            else:
                print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch + 2), train_losses, label='Train')
                if val_loader:
                    ax.plot(range(1, len(val_losses) + 1), val_losses, label='Val')
                ax.legend()
                plt.pause(0.01)

            if val_loader and epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}.")
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        if history_csv is not None:
            import csv, os
            os.makedirs(os.path.dirname(history_csv) or ".", exist_ok=True)
            with open(history_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss", "val_rmse"])
                for i, tr in enumerate(train_losses):
                    vl = val_losses[i] if i < len(val_losses) else None
                    vr = val_rmses[i] if i < len(val_rmses) else None
                    w.writerow([i + 1, tr, "" if vl is None else vl,
                                "" if vr is None else vr])
            print(f"Saved training history -> {history_csv}")

        if live_plot:
            plt.ioff()
            plt.close()

    def evaluate_loader(self, loader):
        self.model.eval()
        loss_function = self.get_loss_function()
        running_loss = 0.0
        all_preds = []
        all_true = []
        
        with torch.no_grad():
            for X_batch, X_uns_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                X_uns_batch = X_uns_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                w_pred, t_pred, eta_R_pred = self.model(X_batch)
                X_uns_last = X_uns_batch[:, -1, :]
                P_pred_kW, _, _, _, _ = self.compute_analytical_power(w_pred, t_pred, eta_R_pred, X_uns_last)
                P_pred_scaled = self._scale_power_torch(P_pred_kW)
                
                loss = loss_function(P_pred_scaled, y_batch)
                running_loss += loss.item()
                
                # Collect for RMSE (in kW)
                # True scaled y -> kW
                y_np = y_batch.cpu().numpy()
                y_true_kW = self.data_processor.inverse_transform_y(y_np)
                
                all_preds.append(P_pred_kW.cpu().numpy())
                all_true.append(y_true_kW)

        avg_loss = running_loss / len(loader)
        
        all_preds_cat = np.concatenate(all_preds).reshape(-1, 1)
        all_true_cat = np.concatenate(all_true).reshape(-1, 1)
        rmse = np.sqrt(np.mean((all_preds_cat - all_true_cat)**2))
        
        return avg_loss, rmse


if __name__ == "__main__":
    from read_data import DataProcessor, create_sequences
    
    # Needs to NOT drop Propeller-Shaft-RPM!
    # Update config temporally just for this run to keep RPM
    if 'Propeller-Shaft-RPM' in DataConfig.DROP_COLUMNS:
        DataConfig.DROP_COLUMNS.remove('Propeller-Shaft-RPM')
        
    proc = DataProcessor()
    # We must ensure 'Propeller-Shaft-RPM' is kept in the features!
    res = proc.load_and_prepare_temporal_data()
    if res is None: exit(1)
    
    X_train, X_test, X_train_uns, X_test_uns, y_train, y_test, y_train_uns, y_test_uns = res
    
    feature_indices = {c: i for i, c in enumerate(X_train.columns)}
    
    seq_len = SequenceConfig.LENGTH
    X_tr_seq, X_tr_uns_seq, y_tr_seq = create_sequences(X_train, X_train_uns, y_train, seq_length=seq_len)
    X_te_seq, X_te_uns_seq, y_te_seq = create_sequences(X_test, X_test_uns, y_test, seq_length=seq_len)
    
    n_val = int(len(X_tr_seq) * 0.2)
    
    # Identify Calm-Water vs Weather indices (dt/acceleration excluded from both).
    calm_water_indices, weather_indices = split_calm_weather_indices(X_train.columns)
    print(f"Calm water features: {len(calm_water_indices)}, Weather features: {len(weather_indices)}")
    
    model = PINODEPropellerModel(
        input_size=X_tr_seq.shape[2],
        feature_indices=feature_indices,
        calm_water_indices=calm_water_indices,
        weather_indices=weather_indices,
        data_processor=proc,
        hidden_size=64,
        ode_num_layers=2,
        lr=0.001,
        epochs=50,
        batch_size=64,
        loss_function_choice='SmoothL1',
    )
    
    train_loader = model.prepare_sequence_dataloader(
        X_tr_seq[:-n_val], X_tr_uns_seq[:-n_val], y_tr_seq[:-n_val], shuffle=True
    )
    val_loader = model.prepare_sequence_dataloader(
        X_tr_seq[-n_val:], X_tr_uns_seq[-n_val:], y_tr_seq[-n_val:], shuffle=False
    )
    test_loader = model.prepare_sequence_dataloader(
        X_te_seq, X_te_uns_seq, y_te_seq, shuffle=False
    )
    
    print("Training PI-NODE with Propeller Hard-Physics...")
    model.train(train_loader, val_loader=val_loader, checkpoint_path="best_pinode_propeller.pt")
    
    print("\nEvaluating on Test Set...")
    test_loss, test_rmse = model.evaluate_loader(test_loader)
    print(f"FINAL TEST RMSE: {test_rmse:.2f} kW")
