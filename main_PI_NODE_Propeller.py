"""PI-NODE propeller model: a physics-informed Neural ODE for ship shaft-power prediction.

The architecture splits operational inputs into a calm-water branch (Neural ODE encoder → latent
ODE integration → decoder → w, t, η_R) and a sea-state residual branch (shallow MLP → Δw, Δt, Δη_R).
Combined wake fraction w and thrust-deduction t feed differentiable Wageningen B-series K_T/K_Q cubic
polynomials inside the propeller law P = 2πnQ, making every gradient path physically interpretable.
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
    """Right-hand side of dz/dt = f_θ(z, ctx), the learnable vector field inside the ODE solver.

    The context vector ctx (mean of the calm-water features over the input window) is injected at
    every evaluation of f_θ, so the trajectory shape adapts to the current operating point without
    requiring a separate conditional-ODE architecture.
    """
    def __init__(self, hidden_size, context_size, num_layers=2):
        super().__init__()
        # Ensure context_size matches what we pass
        in_dim = hidden_size + context_size
        layers = []
        prev = in_dim
        for _ in range(num_layers):
            # Tanh keeps the latent dynamics bounded, preventing blow-up during integration.
            layers.extend([nn.Linear(prev, hidden_size), nn.Tanh()])
            prev = hidden_size
        self.net = nn.Sequential(*layers)
        self._ctx = None

    def set_context(self, ctx):
        # Store ctx before each forward pass; torchdiffeq calls forward(t, z) without extra args.
        self._ctx = ctx

    def forward(self, t, z):
        # Concatenate latent state z with the fixed context before computing dz/dt.
        return self.net(torch.cat([z, self._ctx], dim=-1))


class _Encoder(nn.Module):
    """Maps a single calm-water feature vector (one timestep) to the initial latent state z₀."""
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            # Tanh bounds z₀ in (−1, 1), giving the ODE a stable starting region.
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    """Projects the final ODE latent state z_T to the three calm-water propulsive coefficients.

    Outputs are raw (unbounded) logits; the caller applies sigmoid-based squashing to place
    each coefficient inside its physically admissible interval.
    """
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
    """Dual-branch grey-box predictor for w, t, and η_R.

    Branch A (calm water): an encoder maps the first (or summary) timestep of the calm-water
    features to z₀; a Neural ODE integrates dz/dt = f_θ(z; ctx) over a normalised time span
    [0, 1] with n_ode_steps evaluation points; a decoder converts z_T to (w_calm, t_calm, η_R_calm).

    Branch B (sea state): a shallow MLP processes the last-timestep weather features and outputs
    additive corrections (Δw, Δt, Δη_R) that capture wave/wind-induced deviations from calm-water
    performance without contaminating the clean hydrodynamic signal in Branch A.
    """
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
        # The encoder reads from the calm-water feature subset only; weather signals must not leak
        # into the ODE trajectory, which models steady hydrodynamic state evolution.
        self.encoder = _Encoder(calm_input_size, hidden_size, dropout)
        # GRU variant — aggregates the full window sequentially before handing off to the ODE.
        # Only instantiated when encoder_mode='gru' to avoid unnecessary parameters.
        self.gru_encoder = nn.GRU(
            input_size=calm_input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        ) if encoder_mode == 'gru' else None
        # The ODE function takes (z, ctx) as input; context_size must match calm_input_size.
        self.ode_func = _ODEFunc(hidden_size, calm_input_size, ode_num_layers)
        self.decoder = _Decoder(hidden_size, dropout) # Outputs [w_calm, t_calm, eta_R_calm]

        # Branch B: Weather Residual
        # Shallow enough to learn mean sea-state offsets without overfitting; the final layer
        # starts near zero so Branch B does not overwhelm Branch A early in training.
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

        # Trainable K_Q polynomial coefficients initialised from a Wageningen B-series estimate.
        # Starting near the B-series values gives the physics prior a head start; the prior
        # regularisation penalty (LAMBDA_KQ_PRIOR) then keeps drift in check during training.
        self.kq_c0 = nn.Parameter(torch.tensor(0.06))
        self.kq_c1 = nn.Parameter(torch.tensor(-0.04))
        self.kq_c2 = nn.Parameter(torch.tensor(-0.01))
        self.kq_c3 = nn.Parameter(torch.tensor(0.00))

        # Trainable K_T polynomial coefficients — same initialisation philosophy as K_Q.
        self.kt_b0 = nn.Parameter(torch.tensor(0.60))
        self.kt_b1 = nn.Parameter(torch.tensor(-0.40))
        self.kt_b2 = nn.Parameter(torch.tensor(-0.10))
        self.kt_b3 = nn.Parameter(torch.tensor(0.00))

    def forward(self, x_seq):
        # Split the full feature sequence into calm-water and weather subsets.
        x_calm = x_seq[:, :, self.calm_water_indices]
        # Only the final timestep is used for the weather residual; short-term sea-state
        # averages would lag the instantaneous condition that matters for power at time T.
        x_weather_last = x_seq[:, -1, self.weather_indices]

        # --- Branch A: Calm-Water ODE ---
        # Context = time-average of calm features over the window; represents the mean
        # operating point that modulates the shape of the latent ODE trajectory.
        ctx = x_calm.mean(dim=1)

        # Three encoder_mode options offer an ablation axis on how z₀ is initialised:
        #   'first'     — use only the oldest timestep (cheapest; suitable when the window
        #                 captures a near-stationary operating condition).
        #   'last_mean' — use the most recent timestep (biases z₀ toward current state).
        #   'gru'       — full sequential summary; highest capacity but more parameters.
        if self.encoder_mode == 'first':
            z0 = self.encoder(x_calm[:, 0, :])
        elif self.encoder_mode == 'last_mean':
            z0 = self.encoder(x_calm[:, -1, :])
        elif self.encoder_mode == 'gru':
            _, h_n = self.gru_encoder(x_calm)
            # h_n[-1] is the final GRU hidden state; tanh keeps it in the same range as the
            # MLP encoder output so the ODE dynamics are not disrupted at initialisation.
            z0 = torch.tanh(h_n[-1])
        else:
            raise ValueError(
                f"Unknown encoder_mode='{self.encoder_mode}'. "
                "Supported modes: 'first', 'last_mean', 'gru'."
            )

        if self.use_ode:
            self.ode_func.set_context(ctx)

            # Integrate over a unit interval; the number of evaluation points controls the
            # trade-off between trajectory resolution and forward-pass cost.
            t_span = torch.linspace(0.0, 1.0, self.n_ode_steps + 1,
                                    device=z0.device, dtype=z0.dtype)

            if self.use_adjoint:
                # Adjoint method: O(1) memory in depth (backpropagates through ODE as a black-box).
                # adjoint_params must include any tensor that is used inside the ODE function.
                adjoint_params = tuple(self.ode_func.parameters()) + (ctx,)
                z_traj = _odeint_adjoint(
                    self.ode_func, z0, t_span,
                    method=self.solver,
                    adjoint_params=adjoint_params,
                )
            else:
                # Standard autograd through the solver steps — fine for moderate n_ode_steps.
                z_traj = _odeint(
                    self.ode_func, z0, t_span,
                    method=self.solver,
                )

            # Take the final trajectory state; intermediate states are not supervised.
            z_final = z_traj[-1]
        else:
            # Ablation: no ODE — decode the encoded latent state directly.
            z_final = z0

        out_calm = self.decoder(z_final)

        # Sigmoid squashing maps unbounded logits to physically bounded coefficients:
        #   w (wake fraction):    [0.05, 0.45] — typical range for single-screw vessels.
        #   t (thrust deduction): [0.05, 0.30] — tightly bounded; rarely deviates far from 0.15.
        #   η_R (rel-rotative):   [0.95, 1.10] — usually slightly above 1 for fixed-pitch propellers.
        w_calm = 0.05 + 0.40 * torch.sigmoid(out_calm[:, 0:1])
        t_calm = 0.05 + 0.25 * torch.sigmoid(out_calm[:, 1:2])
        eta_R_calm = 0.95 + 0.15 * torch.sigmoid(out_calm[:, 2:3])

        # --- Branch B: Weather Residual ---
        if self.use_weather:
            out_residual = self.weather_residual_net(x_weather_last)

            # Scale down residuals so that early-training corrections are small relative to
            # the calm-water baseline; the network can grow them if the data demands it.
            dw = out_residual[:, 0:1] * 0.1  # scaling factor to keep initial predictions small
            dt = out_residual[:, 1:2] * 0.1
            deta_R = out_residual[:, 2:3] * 0.05
        else:
            # Ablation: no Sea-State residual branch.
            dw = torch.zeros_like(w_calm)
            dt = torch.zeros_like(t_calm)
            deta_R = torch.zeros_like(eta_R_calm)

        # Combine calm-water baseline with weather-induced corrections.
        w_total = w_calm + dw
        t_total = t_calm + dt
        eta_R_total = eta_R_calm + deta_R

        # Hard-clamp the combined outputs to absolute physical limits as a safety net;
        # the soft range penalties in the loss do most of the work, but pathological
        # inputs (extreme sea states or outlier RPM) could otherwise push predictions
        # outside admissible propeller operating space.
        w_total = torch.clamp(w_total, min=0.01, max=0.60)
        t_total = torch.clamp(t_total, min=0.01, max=0.40)
        eta_R_total = torch.clamp(eta_R_total, min=0.80, max=1.20)

        return w_total, t_total, eta_R_total


# ── Trainer / Evaluator ───────────────────────────────────────────

class PINODEPropellerModel:
    KNOTS_TO_MS = 0.51444

    # Soft physics regularization defaults (kept intentionally mild).
    # KQ_MIN/MAX define the admissible torque-coefficient window for this propeller class;
    # values outside this window indicate physically impossible operating points.
    KQ_MIN = 0.0
    KQ_MAX = 0.12
    # Wageningen B-series nominal coefficients used as the soft prior target.
    KQ_PRIOR = (0.06, -0.04, -0.01, 0.00)
    LAMBDA_KQ_RANGE = 0.5       # Weight on the K_Q/K_T out-of-range quadratic penalty.
    LAMBDA_KQ_CURVATURE = 0.01  # Weight on the polynomial second-derivative penalty.
    LAMBDA_KQ_PRIOR = 0.001     # Weight on the soft B-series coefficient prior.

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
            # MPS OOM can surface here on Apple Silicon with large hidden sizes;
            # fall back to CPU rather than crashing the training run.
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
        # Give the K_T/K_Q polynomial coefficients a mildly boosted learning rate (1.5×) relative
        # to the neural network weights.  The polynomial lives in a much lower-dimensional and
        # physically constrained space; a slightly larger step helps it converge without destabilising
        # the encoder/ODE parameters that require more conservative updates.
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
        # SmoothL1 (Huber) with β=0.5 is less sensitive to the power outliers that occur during
        # manoeuvring or port approaches; quadratic below 0.5 kW (normalised) and linear above.
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
        """Evaluate the propeller law given predicted (w, t, η_R) and measured shaft speed / vessel speed.

        The chain of calculations follows ISO 15016 / ITTC-78 propulsion definitions:
            V_a = V_ship · (1 − w)          — advance speed into the propeller disc
            J   = V_a / (n · D)             — advance coefficient
            K_Q = Σ cᵢ Jⁱ                  — torque coefficient (trainable cubic)
            K_T = Σ bᵢ Jⁱ                  — thrust coefficient (trainable cubic)
            Q   = K_Q · ρ · n² · D⁵        — shaft torque [N·m]
            P   = 2π · n · Q                — delivered power [W]
        """
        # Retrieve feature column indices for RPM and ship speed from the unscaled input.
        rpm_idx = self.feature_indices.get('Propeller-Shaft-RPM')
        if rpm_idx is None:
            raise ValueError("Propeller-Shaft-RPM is required for Propeller Physics model")

        speed_idx = self.feature_indices[ColumnConfig.SPEED]

        # Clamp RPM > 10 to avoid division-by-zero in the advance coefficient J = V_a/(nD).
        rpm = torch.clamp(X_uns_last[:, rpm_idx], min=10.0).view(-1, 1)
        n = rpm / 60.0 # revolutions per second

        # Convert ship speed from knots to m/s for dimensional consistency.
        V_knots = torch.clamp(X_uns_last[:, speed_idx], min=1.0).view(-1, 1)
        V_ship = V_knots * self.KNOTS_TO_MS # m/s

        # 1. Advance speed V_a: the effective inflow velocity seen by the propeller disc,
        #    reduced from ship speed by the wake fraction w (Taylor wake fraction convention).
        Va = V_ship * (1.0 - w_pred)

        # 2. Advance coefficient J: non-dimensional propeller loading parameter.
        D = PropellerConfig.D
        J = Va / (n * D)

        # 3. Torque coefficient K_Q and thrust coefficient K_T from the learnable cubic polynomials.
        #    Cubic form matches the Wageningen B-series regression order and is expressive
        #    enough to represent realistic propeller open-water diagrams.
        c0, c1, c2, c3 = self.model.kq_c0, self.model.kq_c1, self.model.kq_c2, self.model.kq_c3
        KQ = c0 + c1*J + c2*(J**2) + c3*(J**3)

        b0, b1, b2, b3 = self.model.kt_b0, self.model.kt_b1, self.model.kt_b2, self.model.kt_b3
        KT = b0 + b1*J + b2*(J**2) + b3*(J**3)

        # 4. Open-water efficiency η₀ = J·K_T / (2π·K_Q): used only for the soft efficiency
        #    regularisation penalty; not fed back into the power calculation.
        #    A small ε on K_Q prevents NaN when the polynomial passes through zero.
        eta_0 = (J * KT) / (2.0 * np.pi * (KQ + 1e-8))

        # 5. Shaft torque Q [N·m] from the dimensional K_Q relation.
        rho = ShipConfig.RHO
        Q_pred_Nm = KQ * rho * (n**2) * (D**5)

        # 6. Delivered power P [W] = 2π · n · Q; convert to kW for comparison with logged data.
        P_pred_W = 2.0 * np.pi * n * Q_pred_Nm
        P_pred_kW = P_pred_W / 1000.0

        return P_pred_kW, KQ, KT, J, eta_0

    def _scale_power_torch(self, P_pred_kW):
        """Normalise predicted kW power into the same standardised space as the training targets."""
        scaler = self.data_processor.scaler_y
        y_mean = torch.tensor(scaler.mean_[0], dtype=torch.float32, device=self.device)
        y_std = torch.tensor(scaler.scale_[0], dtype=torch.float32, device=self.device)
        return (P_pred_kW - y_mean) / y_std

    def train(self, train_loader, val_loader=None, live_plot=False,
              metrics_output_path=None, checkpoint_path="best_model_PI_NODE_Propeller.pt",
              history_csv=None):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()
        # ReduceLROnPlateau halves the LR after 10 epochs of no val-loss improvement;
        # floor at 5 % of initial LR prevents the scheduler from effectively stopping training.
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

                # 4. Primary task loss on predicted vs measured shaft power (normalised space).
                loss = loss_function(P_pred_scaled, y_batch)

                # Soft physics regularisation — four penalty terms that enforce prior knowledge
                # about the propeller operating envelope without hard-constraining gradients:
                #
                #   Range penalty  : quadratic penalty whenever K_Q or K_T leaves its
                #                    admissible interval; steers coefficients back without clamping.
                #   Curvature penalty : penalises large second derivatives of the polynomial
                #                    (2c₂ + 6c₃J); prevents the cubic from oscillating wildly
                #                    between the sparse J values seen in the training data.
                #   Prior penalty  : soft L2 pull toward the Wageningen B-series coefficients,
                #                    so the polynomial cannot drift arbitrarily far from the
                #                    physical baseline even with sufficient data.
                #   η₀ bound       : ensures open-water efficiency stays in [0.4, 0.75],
                #                    the realistic range for fixed-pitch single-screw propellers.
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

                # Polynomial curvature = second derivative d²K/dJ² evaluated at each sample J.
                kq_curvature = 2.0 * c2 + 6.0 * c3 * J
                kt_curvature = 2.0 * b2 + 6.0 * b3 * J
                curvature_penalty = torch.mean(kq_curvature ** 2 + kt_curvature ** 2)

                p0, p1, p2, p3 = self.KQ_PRIOR
                kq_prior_penalty = (
                    (c0 - p0) ** 2 + (c1 - p1) ** 2 + (c2 - p2) ** 2 + (c3 - p3) ** 2
                )

                # η₀ should be roughly between 0.4 and 0.75 for a normal propeller
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
                # Gradient clipping at norm 5.0 guards against ODE-step explosions that can
                # occur when the solver encounters steep regions of the latent vector field.
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

                # Log the current polynomial state so it is easy to track whether the
                # coefficients are drifting away from physically sensible values.
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

        # Restore the best checkpoint at the end of training.
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

                # Convert scaled targets back to kW for a physically meaningful RMSE.
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

    # Propeller-Shaft-RPM must remain in the feature set for the advance coefficient J = V_a/(nD).
    # Remove it from the drop list for this entry-point if it was excluded by default.
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
