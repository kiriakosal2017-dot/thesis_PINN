"""
Physics-hybrid baseline (HYBRID): an MLP trained with a four-term objective that blends
supervised data loss (alpha), a PGNN soft-physics guidance penalty (beta) derived from
ITTC-78 hull resistance, a PINN collocation PDE residual (gamma), and a P(V=0)=0 boundary
condition (delta). The physics terms regularise the network toward physically plausible
shaft-power predictions without replacing the data-driven fit.
"""

import copy
from itertools import product

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from base_model import BaseModel
from config import ColumnConfig, DataConfig, ShipConfig, TrainingConfig
from read_data import DataProcessor


class UnifiedPhysicsHybridModel(BaseModel):
    """Unified objective: DATA + PGNN guidance + PINN PDE/BC."""

    KNOTS_TO_MS = 0.51444

    def __init__(
        self,
        input_size,
        lr=0.001,
        epochs=100,
        batch_size=64,
        optimizer_choice="Adam",
        loss_function_choice="MSE",
        hidden_layers=None,
        # Loss weights; alpha dominates so physics terms act as soft penalties, not hard
        # constraints — the network can deviate where the physics model is inaccurate.
        alpha=1.0,   # data
        beta=0.05,   # PGNN guidance
        gamma=0.05,  # PINN PDE
        delta=0.02,  # PINN BC
    ):
        super().__init__(
            input_size, lr, epochs, batch_size, optimizer_choice, loss_function_choice, hidden_layers=hidden_layers
        )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def prepare_combined_dataloader(self, X, X_unscaled, y, shuffle=True):
        """Build a single dataloader yielding (X_scaled, X_unscaled, y) tuples.

        Bundling the scaled features, unscaled features, and target into one dataset
        keeps the physics inputs row-aligned with their targets after shuffling —
        impossible when zipping two independently-shuffled loaders.
        """
        X_t = torch.tensor(X.values, dtype=torch.float32)
        Xu_t = torch.tensor(X_unscaled.values, dtype=torch.float32)
        y_t = torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
        dataset = TensorDataset(X_t, Xu_t, y_t)
        return DataLoader(
            dataset,
            batch_size=self._effective_batch_size(len(X)),
            shuffle=shuffle,
            num_workers=0,
        )

    @staticmethod
    def _inverse_scale_power_torch(predicted_power_scaled, data_processor):
        # Undo standard-score normalisation so physics and predicted power are
        # compared in the same physical units (kW).
        scaler = data_processor.scaler_y
        y_mean = torch.tensor(
            scaler.mean_[0], dtype=predicted_power_scaled.dtype, device=predicted_power_scaled.device
        )
        y_std = torch.tensor(
            scaler.scale_[0], dtype=predicted_power_scaled.dtype, device=predicted_power_scaled.device
        )
        return predicted_power_scaled * y_std + y_mean

    def _extract_physics_inputs(self, X_unscaled_batch, feature_indices):
        # Speed arrives in knots from the data pipeline; resistance formulae require m/s.
        # Clamping prevents division-by-zero in the Reynolds-number calculation at V≈0.
        V = X_unscaled_batch[:, feature_indices[ColumnConfig.SPEED]] * self.KNOTS_TO_MS
        V = torch.clamp(V, min=1e-4)
        # Positive trim = vessel trimmed by stern (fore draft > aft draft).
        trim = (
            X_unscaled_batch[:, feature_indices[ColumnConfig.DRAFT_FORE]]
            - X_unscaled_batch[:, feature_indices[ColumnConfig.DRAFT_AFT]]
        )

        hs_idx = feature_indices.get(ColumnConfig.WAVE_HEIGHT)
        heading_idx = feature_indices.get(ColumnConfig.HEADING)
        wave_dir_idx = feature_indices.get(ColumnConfig.WAVE_DIRECTION)

        H_s = X_unscaled_batch[:, hs_idx] if hs_idx is not None else None
        theta_ship = X_unscaled_batch[:, heading_idx] if heading_idx is not None else None
        theta_wave = X_unscaled_batch[:, wave_dir_idx] if wave_dir_idx is not None else None
        return V, trim, H_s, theta_ship, theta_wave

    def _compute_resistance(self, V, trim, H_s=None, theta_ship=None, theta_wave=None):
        """Estimate total calm-water + added-wave resistance via ITTC-78 components.

        Components
        ----------
        R_F   : ITTC-57 frictional resistance  0.5 * rho * V^2 * S * C_f
                with C_f = 0.075 / (log10(Re) - 2)^2
        R_W   : wave-making resistance, scaled by a trim factor STWAVE2 = 1 + alpha_trim * trim
        R_APP : appendage resistance (rudder, shaft brackets) using the same C_f on S_APP
        R_TR  : transom resistance, attenuated by the transom Froude number F_nt = V / sqrt(g * L_t)
        R_C   : correlation allowance  0.5 * rho * V^2 * S * C_A  (empirical hull roughness)
        Form-factor (1+K) multiplies R_F to account for viscous pressure drag.
        R_AW  : added wave resistance (optional) — Bretschneider-style quadratic in H_s,
                modulated by relative encounter angle so head seas give maximum resistance.
        """
        ship = ShipConfig

        g = torch.tensor(ship.G, device=V.device, dtype=V.dtype)
        L_t = torch.tensor(ship.L_T, device=V.device, dtype=V.dtype)
        L = torch.tensor(ship.L, device=V.device, dtype=V.dtype)
        nu = torch.tensor(ship.NU, device=V.device, dtype=V.dtype)

        # ITTC-57 friction line; clamp Re away from zero.
        Re = torch.clamp(V * L / nu, min=1e-5)
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2

        # Frictional resistance (form-factor expansion applied below at R_total summation).
        R_F = 0.5 * ship.RHO * V**2 * ship.S * C_f
        # Wave-making resistance; STWAVE2 captures the trim dependency linearly.
        STWAVE2 = 1 + ship.ALPHA_TRIM * trim
        R_W = 0.5 * ship.RHO * V**2 * ship.S * ship.STWAVE1 * STWAVE2
        # Appendage resistance uses hull C_f on a separate wetted area S_APP.
        R_APP = 0.5 * ship.RHO * V**2 * ship.S_APP * C_f
        # Transom resistance vanishes at high speed (ventilated transom, F_nt → 1).
        F_nt = V / torch.sqrt(g * L_t)
        R_TR = 0.5 * ship.RHO * V**2 * ship.A_T * (1 - F_nt)
        # Correlation allowance — lumps together hull-roughness and model-ship corrections.
        R_C = 0.5 * ship.RHO * V**2 * ship.S * ship.C_A

        # Sum: R_F scaled by form factor, plus remaining ITTC-78 components.
        R_total = R_F * (1 + ship.K) + R_W + R_APP + R_TR + R_C

        if H_s is not None and theta_ship is not None and theta_wave is not None:
            # Added wave resistance — proportional to H_s^2 and to (1 + cos(theta_rel)),
            # which peaks at head seas (theta_rel = 0°) and is zero at stern seas (180°).
            H_s = torch.clamp(H_s, min=0.0)
            theta_rel = torch.abs(theta_wave - theta_ship) % 360
            theta_rel = torch.where(theta_rel > 180, 360 - theta_rel, theta_rel)
            theta_rel_rad = theta_rel * np.pi / 180
            k_wave = 1e-7
            R_AW = 0.5 * ship.RHO * V**2 * ship.S * k_wave * H_s**2 * (1 + torch.cos(theta_rel_rad))
            R_total = R_total + R_AW

        return R_total

    def physics_guidance_loss(self, X_unscaled_batch, predicted_power_scaled, feature_indices, data_processor):
        """PGNN soft-guidance loss: penalise deviation from the ITTC-78 power estimate.

        Shaft power follows P = R_total * V / eta_D (cubic-law in V for calm water).
        The loss is normalised by the mean squared physics estimate so that its magnitude
        stays roughly O(1) across different operating conditions and training stages,
        keeping it commensurable with the unit-normalised data loss.
        """
        V, trim, H_s, theta_ship, theta_wave = self._extract_physics_inputs(X_unscaled_batch, feature_indices)
        R_total = self._compute_resistance(V, trim, H_s, theta_ship, theta_wave)

        # Cubic-law power: P = R * V / eta_D, converted to kW.
        P_phys_kW = ((V * R_total) / ShipConfig.ETA_D) / 1000.0
        P_pred_kW = self._inverse_scale_power_torch(predicted_power_scaled, data_processor).squeeze()

        loss_raw = (P_pred_kW - P_phys_kW) ** 2
        # Relative (normalised) MSE — detach the physics estimate from the graph
        # so gradients only flow through the network prediction.
        scale = torch.mean(P_phys_kW.detach() ** 2) + 1e-8
        return torch.mean(loss_raw) / scale

    def sample_collocation_points(self, num_points, X_train_unscaled, data_processor):
        # Draw uniformly from the observed feature range, then rescale to the
        # network's input space — collocation points must live in the same
        # normalised domain as training data.
        x_min = X_train_unscaled.min()
        x_max = X_train_unscaled.max()
        x_collocation_dict = {
            col: np.random.uniform(low=x_min[col], high=x_max[col], size=num_points)
            for col in X_train_unscaled.columns
        }
        x_collocation_unscaled = pd.DataFrame(x_collocation_dict)
        x_collocation_unscaled = torch.tensor(
            data_processor.scaler_X.transform(x_collocation_unscaled),
            dtype=torch.float32,
            device=self.device,
        )
        # requires_grad enables autograd differentiation w.r.t. inputs inside compute_pde_residual.
        x_collocation_unscaled.requires_grad = True
        return x_collocation_unscaled

    def compute_pde_residual(self, x_collocation, feature_indices):
        """Evaluate the surrogate PDE residual: dP/dV + a*P - b*V^2 = 0.

        This ODE approximates the known physical relationship that shaft power
        grows roughly as V^3. The residual is driven to zero at unseen collocation
        points, enforcing monotonicity and the correct speed-power scaling beyond
        the training distribution.
        """
        x_collocation.requires_grad_(True)
        outputs = self.model(x_collocation)
        # Compute the full input-Jacobian via a single backward pass.
        outputs_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]
        V_idx = feature_indices[ColumnConfig.SPEED]
        V = x_collocation[:, V_idx].view(-1, 1)
        dP_dV = outputs_x[:, V_idx].view(-1, 1)
        # Surrogate ODE coefficients; a and b are calibrated to match the
        # approximate V^3 power curve shape in normalised space.
        a = torch.tensor(0.1, device=self.device)
        b = torch.tensor(0.2, device=self.device)
        return dP_dV + a * outputs - b * V**2

    def sample_boundary_points(self, num_points, X_train_unscaled, feature_indices, data_processor):
        # Boundary condition: speed = 0 while all other features are randomised
        # across their observed range. Setting speed to zero in unscaled space
        # before transforming ensures the scaled speed column lands at the correct
        # normalised value for V=0.
        x_min = X_train_unscaled.min()
        x_max = X_train_unscaled.max()
        cols = list(X_train_unscaled.columns)
        speed_col = cols[feature_indices[ColumnConfig.SPEED]]
        rows = {}
        for col in cols:
            if col == speed_col:
                rows[col] = np.zeros(num_points)
            else:
                rows[col] = np.random.uniform(low=x_min[col], high=x_max[col], size=num_points)
        x_boundary = pd.DataFrame(rows, columns=cols)
        x_boundary_scaled = data_processor.scaler_X.transform(x_boundary)
        x_boundary_t = torch.tensor(x_boundary_scaled, dtype=torch.float32, device=self.device)
        x_boundary_t.requires_grad_(True)
        return x_boundary_t

    def compute_boundary_loss(self, x_boundary, data_processor):
        # Enforce P(V=0) = 0 kW. In the normalised target space, 0 kW maps to
        # scaled_zero = (0 - y_mean) / y_std, NOT 0 — pushing the scaled output to 0
        # would wrongly enforce P = mean(power).
        out = self.model(x_boundary)
        scaler = data_processor.scaler_y
        scaled_zero = torch.tensor(
            (0.0 - scaler.mean_[0]) / scaler.scale_[0],
            dtype=out.dtype, device=out.device,
        )
        return torch.mean((out - scaled_zero) ** 2)

    def train(self, train_loader, X_train_unscaled, feature_indices, data_processor,
              val_loader=None, checkpoint_path=None, history_csv=None):
        optimizer = self.get_optimizer()
        loss_fn = self.get_loss_function()

        best_state = None
        best_val = float("inf")
        # Early stopping monitors the pure data loss on the validation set so that
        # physics term fluctuations do not trigger premature termination.
        patience = TrainingConfig.EARLY_STOPPING_PATIENCE
        min_delta = TrainingConfig.EARLY_STOPPING_MIN_DELTA
        epochs_wo = 0
        train_losses, val_losses = [], []

        for epoch in range(self.epochs):
            self.model.train()
            run_total = run_data = run_pg = run_pde = run_bc = 0.0
            total_batches = len(train_loader)
            bar = tqdm(train_loader, total=total_batches, desc=f"Epoch {epoch+1}/{self.epochs}")

            for i, (Xb, Xub, yb) in enumerate(bar):
                Xb, Xub, yb = Xb.to(self.device), Xub.to(self.device), yb.to(self.device)
                optimizer.zero_grad()

                pred = self.model(Xb)
                # Supervised data loss on this mini-batch.
                data_loss = loss_fn(pred, yb)
                # PGNN guidance: penalise departure from ITTC-78 power, using
                # unscaled features so physical units are correct inside the formula.
                pg_loss = self.physics_guidance_loss(Xub, pred, feature_indices, data_processor)

                # Fresh collocation points each step so the PDE constraint covers
                # the full input domain stochastically rather than a fixed grid.
                x_col = self.sample_collocation_points(self.batch_size, X_train_unscaled, data_processor)
                pde_loss = torch.mean(self.compute_pde_residual(x_col, feature_indices) ** 2)

                # Boundary points with V=0; bc_loss enforces P(V=0)=0 kW.
                x_bc = self.sample_boundary_points(self.batch_size, X_train_unscaled, feature_indices, data_processor)
                bc_loss = self.compute_boundary_loss(x_bc, data_processor)

                # Four-term weighted objective: alpha*data + beta*PGNN + gamma*PDE + delta*BC.
                total_loss = (
                    self.alpha * data_loss
                    + self.beta * pg_loss
                    + self.gamma * pde_loss
                    + self.delta * bc_loss
                )
                total_loss.backward()
                optimizer.step()

                run_total += total_loss.item()
                run_data += data_loss.item()
                run_pg += pg_loss.item()
                run_pde += pde_loss.item()
                run_bc += bc_loss.item()
                bar.set_postfix({
                    "Total": f"{run_total/(i+1):.6f}",
                    "Data": f"{run_data/(i+1):.6f}",
                    "PG": f"{run_pg/(i+1):.6f}",
                    "PDE": f"{run_pde/(i+1):.6f}",
                    "BC": f"{run_bc/(i+1):.6f}",
                })

            val_cur = None
            if val_loader is not None:
                # Validation uses data loss only — physics terms are not evaluated
                # on held-out data because collocation points are sampled independently.
                self.model.eval()
                v = 0.0
                with torch.no_grad():
                    for Xv, yv in val_loader:
                        Xv, yv = Xv.to(self.device), yv.to(self.device)
                        v += loss_fn(self.model(Xv), yv).item()
                val_cur = v / len(val_loader)
                print(f"Epoch {epoch+1}: train_total={run_total/max(total_batches,1):.6f}, val_data={val_cur:.6f}")

            train_losses.append(run_total / max(total_batches, 1))
            val_losses.append(val_cur)

            if val_cur is not None:
                if (best_val - val_cur) > min_delta:
                    best_val = val_cur
                    epochs_wo = 0
                    best_state = copy.deepcopy(self.model.state_dict())
                    if checkpoint_path:
                        torch.save(best_state, checkpoint_path)
                else:
                    epochs_wo += 1
                    if epochs_wo >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break

        # Restore the best-validation checkpoint before returning.
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best HYBRID model (val_data={best_val:.6f})")

        if history_csv is not None:
            import csv, os
            os.makedirs(os.path.dirname(history_csv) or ".", exist_ok=True)
            with open(history_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["epoch", "train_loss", "val_loss"])
                for i, tr in enumerate(train_losses):
                    vl = val_losses[i] if i < len(val_losses) else None
                    w.writerow([i + 1, tr, "" if vl is None else vl])
            print(f"Saved training history -> {history_csv}")

    def cross_validate(self, X, X_unscaled, y, feature_indices, data_processor, k_folds=5):
        # The combined dataloader for training must carry unscaled inputs for the
        # physics terms; the validation dataloader uses scaled inputs only (data loss).
        kf = KFold(n_splits=k_folds, shuffle=True, random_state=DataConfig.RANDOM_STATE)
        scores = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
            print(f"\nFold {fold}/{k_folds}")
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            Xtr_un = X_unscaled.iloc[tr_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            train_loader = self.prepare_combined_dataloader(Xtr, Xtr_un, ytr, shuffle=True)
            val_loader = self.prepare_dataloader(Xva, yva)

            # Reset weights so each fold starts from scratch.
            self.model.apply(self.reset_weights)
            self.train(train_loader, Xtr_un, feature_indices, data_processor, val_loader=val_loader)
            val_loss = self.evaluate(Xva, yva, dataset_type="Validation", data_processor=data_processor)
            scores.append(val_loss)
        avg = float(np.mean(scores))
        print(f"\nCross-validation mean validation loss: {avg:.8f}")
        return avg

    @staticmethod
    def hyperparameter_search(X_train, X_train_unscaled, y_train, feature_indices, param_grid, data_processor, k_folds=3):
        # Grid search over lr, batch size, and all four loss weights.
        # k_folds=3 is intentionally smaller than final training to keep wall time manageable.
        best_params = None
        best_loss = float("inf")
        combos = list(product(
            param_grid["lr"],
            param_grid["batch_size"],
            param_grid["alpha"],
            param_grid["beta"],
            param_grid["gamma"],
            param_grid["delta"],
        ))

        for lr, batch_size, alpha, beta, gamma, delta in combos:
            print(
                f"\nTesting HYBRID: lr={lr}, batch={batch_size}, "
                f"alpha={alpha}, beta={beta}, gamma={gamma}, delta={delta}"
            )
            model = UnifiedPhysicsHybridModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=TrainingConfig.EPOCHS_CV,
                batch_size=batch_size,
                optimizer_choice=TrainingConfig.OPTIMIZER,
                loss_function_choice=TrainingConfig.LOSS_FUNCTION,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            avg = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor, k_folds=k_folds
            )
            if avg < best_loss:
                best_loss = avg
                best_params = {
                    "lr": lr, "batch_size": batch_size,
                    "alpha": alpha, "beta": beta, "gamma": gamma, "delta": delta,
                }

        print(f"\nBest HYBRID params: {best_params}, val={best_loss:.8f}")
        return best_params, best_loss


def _build_feature_indices(X_unscaled):
    # Map column names to integer positions so physics methods can index into
    # unscaled-feature tensors by position rather than by name.
    feature_indices = {col: idx for idx, col in enumerate(X_unscaled.columns)}
    required = [ColumnConfig.SPEED, ColumnConfig.DRAFT_FORE, ColumnConfig.DRAFT_AFT]
    for col in required:
        if col not in feature_indices:
            raise ValueError(f"Required column '{col}' not found in data")

    # Wave-related columns are optional; their absence disables the R_AW term silently.
    optional = [ColumnConfig.WAVE_HEIGHT, ColumnConfig.HEADING, ColumnConfig.WAVE_DIRECTION]
    for col in optional:
        if col not in feature_indices:
            print(f"Optional column '{col}' not found. Wave-direction term will be skipped.")
    return feature_indices


if __name__ == "__main__":
    dp = DataProcessor()
    result = dp.load_and_prepare_data()
    if result is None:
        raise RuntimeError("Failed to load data")

    X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, _, _ = result
    feature_indices = _build_feature_indices(X_train_unscaled)

    param_grid = {
        "lr": [1e-3, 5e-4],
        "batch_size": [64, 128],
        # alpha=1.0 is fixed; only the physics penalty weights are tuned.
        "alpha": [1.0],
        "beta": [0.03, 0.05],   # PGNN guidance weight
        "gamma": [0.03, 0.05],  # PINN PDE weight
        "delta": [0.01, 0.02],  # PINN BC weight
    }
    best_params, _ = UnifiedPhysicsHybridModel.hyperparameter_search(
        X_train, X_train_unscaled, y_train, feature_indices, param_grid, dp, k_folds=3
    )

    # Chronological validation split (no shuffle) to match the temporal protocol
    # used everywhere else and avoid leaking future rows into validation.
    X_tr, X_val, X_tr_un, _, y_tr, y_val = train_test_split(
        X_train, X_train_unscaled, y_train, test_size=DataConfig.TEST_SIZE, shuffle=False
    )

    model = UnifiedPhysicsHybridModel(
        input_size=X_train.shape[1],
        lr=best_params["lr"],
        epochs=TrainingConfig.EPOCHS_FINAL,
        batch_size=best_params["batch_size"],
        optimizer_choice=TrainingConfig.OPTIMIZER,
        loss_function_choice=TrainingConfig.LOSS_FUNCTION,
        alpha=best_params["alpha"],
        beta=best_params["beta"],
        gamma=best_params["gamma"],
        delta=best_params["delta"],
    )
    train_loader = model.prepare_combined_dataloader(X_tr, X_tr_un, y_tr, shuffle=True)
    val_loader = model.prepare_dataloader(X_val, y_val)
    model.train(
        train_loader,
        X_tr_un,
        feature_indices,
        dp,
        val_loader=val_loader,
        checkpoint_path="best_model_HYBRID.pt",
    )
    model.evaluate(X_test, y_test, dataset_type="Test", data_processor=dp)
