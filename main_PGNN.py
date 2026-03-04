import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DataConfig, ColumnConfig, ShipConfig, TrainingConfig
from read_data import DataProcessor
from base_model import BaseModel


class PGNNModel(BaseModel):
    """Physics-Guided Neural Network: MLP + ship resistance equations in loss."""

    KNOTS_TO_MS = 0.51444

    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE',
                 alpha=1.0, beta=0.1, k_wave=0.005):
        super().__init__(input_size, lr, epochs, batch_size, optimizer_choice, loss_function_choice)
        self.alpha = alpha
        self.beta = beta
        self.k_wave = k_wave

    def calculate_physics_loss(self, V, trim, predicted_power_scaled,
                               H_s, theta_ship, theta_wave, data_processor):
        """Ship resistance loss using ITTC-1957 and added wave resistance."""
        ship = ShipConfig
        g = torch.tensor(ship.G, device=V.device, dtype=V.dtype)
        L_t = torch.tensor(ship.L_T, device=V.device, dtype=V.dtype)
        L = torch.tensor(ship.L, device=V.device, dtype=V.dtype)
        nu = torch.tensor(ship.NU, device=V.device, dtype=V.dtype)

        V = torch.clamp(V, min=1e-5)
        H_s = torch.clamp(H_s, min=0.0)

        Re = torch.clamp(V * L / nu, min=1e-5)
        C_f = 0.075 / (torch.log10(Re) - 2) ** 2

        R_F = 0.5 * ship.RHO * V**2 * ship.S * C_f

        STWAVE2 = 1 + ship.ALPHA_TRIM * trim
        C_W = ship.STWAVE1 * STWAVE2
        R_W = 0.5 * ship.RHO * V**2 * ship.S * C_W

        R_APP = 0.5 * ship.RHO * V**2 * ship.S_APP * C_f

        F_nt = V / torch.sqrt(g * L_t)
        R_TR = 0.5 * ship.RHO * V**2 * ship.A_T * (1 - F_nt)

        R_C = 0.5 * ship.RHO * V**2 * ship.S * ship.C_A

        theta_rel = torch.abs(theta_wave - theta_ship) % 360
        theta_rel = torch.where(theta_rel > 180, 360 - theta_rel, theta_rel)
        theta_rel_rad = theta_rel * np.pi / 180

        R_AW = 0.5 * ship.RHO * V**2 * ship.S * self.k_wave * H_s**2 * (1 + torch.cos(theta_rel_rad))

        R_T = R_F * (1 + ship.K) + R_W + R_APP + R_TR + R_C + R_AW

        P_S = ((V * R_T) / ship.ETA_D) / 1000

        P_S_df = pd.DataFrame(P_S.cpu().detach().numpy(), columns=[data_processor.target_column])
        P_S_scaled = data_processor.scaler_y.transform(P_S_df).flatten()
        P_S_scaled = torch.tensor(P_S_scaled, dtype=V.dtype, device=V.device)

        physics_loss = (predicted_power_scaled.squeeze() - P_S_scaled) ** 2
        return physics_loss, P_S_scaled

    def _extract_physics_inputs(self, X_unscaled_batch, feature_indices):
        """Extract speed, trim, and weather data from unscaled features."""
        V = X_unscaled_batch[:, feature_indices[ColumnConfig.SPEED]] * self.KNOTS_TO_MS
        trim = (X_unscaled_batch[:, feature_indices[ColumnConfig.DRAFT_FORE]]
                - X_unscaled_batch[:, feature_indices[ColumnConfig.DRAFT_AFT]])
        H_s = X_unscaled_batch[:, feature_indices[ColumnConfig.WAVE_HEIGHT]]
        theta_ship = X_unscaled_batch[:, feature_indices[ColumnConfig.HEADING]]
        theta_wave = X_unscaled_batch[:, feature_indices[ColumnConfig.WAVE_DIRECTION]]
        return V, trim, H_s, theta_ship, theta_wave

    def train(self, train_loader, unscaled_data_loader, feature_indices, data_processor,
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

            if self.optimizer_choice == 'LBFGS':
                X_batch, y_batch = train_loader.dataset.tensors
                X_unscaled_batch = unscaled_data_loader.dataset.tensors[0]

                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    data_loss = loss_function(outputs, y_batch)

                    V, trim, H_s, theta_ship, theta_wave = self._extract_physics_inputs(
                        X_unscaled_batch, feature_indices)

                    physics_loss, _ = self.calculate_physics_loss(
                        V, trim, outputs, H_s, theta_ship, theta_wave, data_processor)

                    total_loss = self.alpha * data_loss + self.beta * torch.mean(physics_loss)
                    total_loss.backward()
                    return total_loss

                optimizer.step(closure)
                with torch.no_grad():
                    total_loss = closure()
                running_loss = total_loss.item()

                progress_bar = tqdm(total=1, desc=f"Epoch {epoch+1}/{self.epochs}", leave=True)
                progress_bar.set_postfix({"Total Loss": f"{running_loss:.8f}"})
                progress_bar.update(1)
                progress_bar.close()
            else:
                total_batches = len(train_loader)

                progress_bar = tqdm(
                    zip(train_loader, unscaled_data_loader),
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    leave=True,
                    total=total_batches,
                )

                for batch_index, ((X_batch, y_batch), (X_unscaled_batch,)) in enumerate(progress_bar):
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    X_unscaled_batch = X_unscaled_batch.to(self.device)

                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    data_loss = loss_function(outputs, y_batch)

                    V, trim, H_s, theta_ship, theta_wave = self._extract_physics_inputs(
                        X_unscaled_batch, feature_indices)

                    physics_loss, _ = self.calculate_physics_loss(
                        V, trim, outputs, H_s, theta_ship, theta_wave, data_processor)

                    total_loss = self.alpha * data_loss + self.beta * torch.mean(physics_loss)
                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item()
                    running_data_loss += data_loss.item()
                    running_physics_loss += physics_loss.mean().item()

                    progress_bar.set_postfix({
                        "Total": f"{running_loss / (batch_index + 1):.8f}",
                        "Data": f"{running_data_loss / (batch_index + 1):.8f}",
                        "Physics": f"{running_physics_loss / (batch_index + 1):.8f}",
                    })

            avg_total_loss = running_loss / max(len(train_loader), 1)
            train_losses.append(avg_total_loss)

            if val_loader is not None:
                self.model.eval()
                val_running_loss = 0.0
                with torch.no_grad():
                    for X_val_batch, y_val_batch in val_loader:
                        X_val_batch, y_val_batch = X_val_batch.to(self.device), y_val_batch.to(self.device)
                        val_outputs = self.model(X_val_batch)
                        val_loss = loss_function(val_outputs, y_val_batch)
                        val_running_loss += val_loss.item()
                avg_val_loss = val_running_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}, "
                      f"Validation Loss: {avg_val_loss:.8f}")
            else:
                val_losses.append(None)
                print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}")

            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch + 2), train_losses, label='Training Loss')
                if val_loader is not None:
                    ax.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('PGNN: Training and Validation Loss')
                ax.legend()
                plt.pause(0.01)

        if live_plot:
            plt.ioff()
            plt.show()
            fig.savefig('training_validation_loss_plot_PGNN.png')

    def cross_validate(self, X, X_unscaled, y, feature_indices, data_processor, k_folds=5):
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=DataConfig.RANDOM_STATE)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")

            X_train_fold = X.iloc[train_idx]
            X_train_unscaled_fold = X_unscaled.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]

            train_loader = self.prepare_dataloader(X_train_fold, y_train_fold)
            unscaled_loader = self.prepare_unscaled_dataloader(X_train_unscaled_fold)
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)

            self.model.apply(self.reset_weights)

            self.train(train_loader, unscaled_loader, feature_indices, data_processor,
                       val_loader=val_loader, live_plot=False)

            val_loss = self.evaluate(X_val_fold, y_val_fold, dataset_type="Validation",
                                     data_processor=data_processor)
            fold_results.append(val_loss)

        avg_val_loss = np.mean(fold_results)
        print(f"\nCross-validation results: Average Validation Loss = {avg_val_loss:.8f}")
        return avg_val_loss

    @staticmethod
    def hyperparameter_search(X_train, X_train_unscaled, y_train, feature_indices,
                              param_grid, data_processor, k_folds=5):
        best_params = None
        best_loss = float('inf')

        hyperparameter_combinations = list(product(
            param_grid['lr'],
            param_grid['batch_size'],
            param_grid['alpha'],
            param_grid['beta'],
            param_grid['k_wave'],
        ))

        for lr, batch_size, alpha, beta, k_wave in hyperparameter_combinations:
            print(f"\nTesting: lr={lr}, batch_size={batch_size}, alpha={alpha}, beta={beta}, k_wave={k_wave}")

            model = PGNNModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=TrainingConfig.EPOCHS_CV,
                optimizer_choice=TrainingConfig.OPTIMIZER,
                loss_function_choice=TrainingConfig.LOSS_FUNCTION,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                k_wave=k_wave,
            )

            avg_val_loss = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor, k_folds=k_folds)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {
                    'lr': lr, 'batch_size': batch_size,
                    'alpha': alpha, 'beta': beta, 'k_wave': k_wave,
                }

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")

        with open("best_hyperparameters_PGNN.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss


def _build_feature_indices(X_unscaled):
    """Build and validate feature name -> column index mapping."""
    feature_indices = {col: idx for idx, col in enumerate(X_unscaled.columns)}
    required = [
        ColumnConfig.SPEED, ColumnConfig.DRAFT_FORE, ColumnConfig.DRAFT_AFT,
        ColumnConfig.WAVE_HEIGHT, ColumnConfig.HEADING, ColumnConfig.WAVE_DIRECTION,
    ]
    for col in required:
        if col not in feature_indices:
            raise ValueError(f"Required column '{col}' not found in data")
    return feature_indices


if __name__ == "__main__":
    data_processor = DataProcessor()
    result = data_processor.load_and_prepare_data()

    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, \
            y_train, y_test, y_train_unscaled, y_test_unscaled = result

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        assert list(X_train.columns) == list(X_train_unscaled.columns), \
            "Column mismatch between scaled and unscaled data"

        feature_indices = _build_feature_indices(X_train_unscaled)

        param_grid = {
            'lr': [0.001, 0.01],
            'batch_size': [64, 128],
            'alpha': [0.8, 1.0],
            'beta': [0.1, 0.05],
            'k_wave': [1e-8, 1e-7, 1e-6],
        }

        best_params, best_loss = PGNNModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, data_processor, k_folds=5,
        )

        X_train_final, X_val_final, X_train_unscaled_final, X_val_unscaled_final, \
            y_train_final, y_val_final = train_test_split(
                X_train, X_train_unscaled, y_train,
                test_size=DataConfig.TEST_SIZE, random_state=DataConfig.RANDOM_STATE,
            )

        final_model = PGNNModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],
            epochs=TrainingConfig.EPOCHS_FINAL,
            optimizer_choice=TrainingConfig.OPTIMIZER,
            loss_function_choice=TrainingConfig.LOSS_FUNCTION,
            batch_size=best_params['batch_size'],
            alpha=best_params['alpha'],
            beta=best_params['beta'],
            k_wave=best_params['k_wave'],
        )

        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_unscaled_loader = final_model.prepare_unscaled_dataloader(X_train_unscaled_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)

        final_model.train(
            final_train_loader, final_unscaled_loader, feature_indices, data_processor,
            val_loader=final_val_loader, live_plot=True,
        )

        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)
