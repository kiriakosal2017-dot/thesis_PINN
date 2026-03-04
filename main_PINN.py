import copy
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DataConfig, ColumnConfig, TrainingConfig
from read_data import DataProcessor
from base_model import BaseModel


class PINNModel(BaseModel):
    """Physics-Informed Neural Network: MLP + PDE residuals + boundary conditions."""

    def __init__(self, input_size, lr=0.001, epochs=100, batch_size=32,
                 optimizer_choice='Adam', loss_function_choice='MSE',
                 alpha=1.0, beta=0.1, gamma=0.1):
        super().__init__(input_size, lr, epochs, batch_size, optimizer_choice, loss_function_choice)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def sample_collocation_points(self, num_points, X_train_unscaled, data_processor):
        """Random collocation points within the domain for PDE residual evaluation."""
        x_min = X_train_unscaled.min()
        x_max = X_train_unscaled.max()

        x_collocation_unscaled = pd.DataFrame({
            col: np.random.uniform(low=x_min[col], high=x_max[col], size=num_points)
            for col in X_train_unscaled.columns
        })

        x_collocation_scaled = data_processor.scaler_X.transform(x_collocation_unscaled)
        x_collocation = torch.tensor(x_collocation_scaled, dtype=torch.float32, device=self.device)
        x_collocation.requires_grad = True
        return x_collocation

    def sample_boundary_points(self, num_points, X_train_unscaled, feature_indices, data_processor):
        """Boundary points where V=0, enforcing P=0."""
        x_min = X_train_unscaled.min()
        x_max = X_train_unscaled.max()

        x_boundary_unscaled = pd.DataFrame(columns=X_train_unscaled.columns)

        V_col = X_train_unscaled.columns[feature_indices[ColumnConfig.SPEED]]
        x_boundary_unscaled[V_col] = np.zeros(num_points)

        for col in X_train_unscaled.columns:
            if col != V_col:
                x_boundary_unscaled[col] = np.random.uniform(
                    low=x_min[col], high=x_max[col], size=num_points)

        x_boundary_scaled = data_processor.scaler_X.transform(x_boundary_unscaled)
        x_boundary = torch.tensor(x_boundary_scaled, dtype=torch.float32, device=self.device)
        x_boundary.requires_grad = True
        return x_boundary

    def compute_pde_residual(self, x_collocation, feature_indices):
        """PDE residual: dP/dV + aP - bV^2 = 0"""
        x_collocation.requires_grad = True
        outputs = self.model(x_collocation)

        outputs_x = torch.autograd.grad(
            outputs=outputs,
            inputs=x_collocation,
            grad_outputs=torch.ones_like(outputs),
            create_graph=True,
            retain_graph=True,
        )[0]

        V_idx = feature_indices[ColumnConfig.SPEED]
        V = x_collocation[:, V_idx].view(-1, 1)
        outputs_V = outputs_x[:, V_idx].view(-1, 1)

        a = torch.tensor(0.1, device=self.device)
        b = torch.tensor(0.2, device=self.device)
        residual = outputs_V + a * outputs - b * V**2

        return residual

    def compute_boundary_loss(self, x_boundary):
        """Enforce P = 0 when V = 0."""
        outputs_boundary = self.model(x_boundary)
        return torch.mean(outputs_boundary**2)

    def train(self, train_loader, X_train_unscaled, feature_indices, data_processor,
              val_loader=None, live_plot=False, checkpoint_path=None):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        train_losses = []
        val_losses = []
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
            running_data_loss = 0.0
            running_pde_loss = 0.0
            running_boundary_loss = 0.0
            val_current = None

            if self.optimizer_choice == 'LBFGS':
                X_batch, y_batch = train_loader.dataset.tensors

                def closure():
                    optimizer.zero_grad()
                    outputs = self.model(X_batch)
                    data_loss = loss_function(outputs, y_batch)

                    x_collocation = self.sample_collocation_points(
                        len(X_batch), X_train_unscaled, data_processor)
                    pde_residual = self.compute_pde_residual(x_collocation, feature_indices)
                    pde_loss = torch.mean(pde_residual**2)

                    x_boundary = self.sample_boundary_points(
                        len(X_batch), X_train_unscaled, feature_indices, data_processor)
                    boundary_loss = self.compute_boundary_loss(x_boundary)

                    total_loss = self.alpha * data_loss + self.beta * pde_loss + self.gamma * boundary_loss
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

                train_losses.append(running_loss)
                if val_loader is not None:
                    val_current = self._evaluate_on_loader(val_loader)
                    val_losses.append(val_current)
            else:
                total_batches = len(train_loader)

                progress_bar = tqdm(
                    enumerate(train_loader),
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    leave=True,
                    total=total_batches,
                )

                for batch_index, (X_batch, y_batch) in progress_bar:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    optimizer.zero_grad()

                    outputs = self.model(X_batch)
                    data_loss = loss_function(outputs, y_batch)

                    x_collocation = self.sample_collocation_points(
                        self.batch_size, X_train_unscaled, data_processor)
                    pde_residual = self.compute_pde_residual(x_collocation, feature_indices)
                    pde_loss = torch.mean(pde_residual**2)

                    x_boundary = self.sample_boundary_points(
                        self.batch_size, X_train_unscaled, feature_indices, data_processor)
                    boundary_loss = self.compute_boundary_loss(x_boundary)

                    total_loss = self.alpha * data_loss + self.beta * pde_loss + self.gamma * boundary_loss
                    total_loss.backward()
                    optimizer.step()

                    running_loss += total_loss.item()
                    running_data_loss += data_loss.item()
                    running_pde_loss += pde_loss.item()
                    running_boundary_loss += boundary_loss.item()

                    progress_bar.set_postfix({
                        "Total": f"{running_loss / (batch_index + 1):.8f}",
                        "Data": f"{running_data_loss / (batch_index + 1):.8f}",
                        "PDE": f"{running_pde_loss / (batch_index + 1):.8f}",
                        "BC": f"{running_boundary_loss / (batch_index + 1):.8f}",
                    })

                avg_total_loss = running_loss / total_batches
                train_losses.append(avg_total_loss)

                if val_loader is not None:
                    val_current = self._evaluate_on_loader(val_loader)
                    val_losses.append(val_current)
                    print(f"Epoch [{epoch+1}/{self.epochs}], Total Loss: {avg_total_loss:.8f}, "
                          f"Validation Loss: {val_current:.8f}")
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
                    ax.set_title('PINN: Training and Validation Loss')
                    ax.legend()
                    plt.pause(0.01)

            if val_current is not None:
                if (best_val_loss - val_current) > min_delta:
                    best_val_loss = val_current
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                    if checkpoint_path is not None:
                        torch.save(best_state, checkpoint_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping at epoch {epoch+1}: "
                        f"no Validation improvement > {min_delta} for {patience} epochs."
                    )
                    break

        if live_plot:
            plt.ioff()
            plt.show()
            fig.savefig('training_validation_loss_plot_PINN.png')

        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model state (Validation Loss: {best_val_loss:.8f})")

    def _evaluate_on_loader(self, data_loader):
        self.model.eval()
        loss_function = self.get_loss_function()
        running_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in data_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = loss_function(outputs, y_batch)
                running_loss += loss.item()
        return running_loss / len(data_loader)

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
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)

            self.model.apply(self.reset_weights)

            self.train(train_loader, X_train_unscaled_fold, feature_indices, data_processor,
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
            param_grid['gamma'],
        ))

        for lr, batch_size, alpha, beta, gamma in hyperparameter_combinations:
            print(f"\nTesting: lr={lr}, batch_size={batch_size}, alpha={alpha}, beta={beta}, gamma={gamma}")

            model = PINNModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=TrainingConfig.EPOCHS_CV,
                optimizer_choice=TrainingConfig.OPTIMIZER,
                loss_function_choice=TrainingConfig.LOSS_FUNCTION,
                batch_size=batch_size,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
            )

            avg_val_loss = model.cross_validate(
                X_train, X_train_unscaled, y_train, feature_indices, data_processor, k_folds=k_folds)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {
                    'lr': lr, 'batch_size': batch_size,
                    'alpha': alpha, 'beta': beta, 'gamma': gamma,
                }

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")

        with open("best_hyperparameters_PINN.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss


def _build_feature_indices(X_unscaled):
    """Build and validate feature name -> column index mapping."""
    feature_indices = {col: idx for idx, col in enumerate(X_unscaled.columns)}
    required = [ColumnConfig.SPEED, ColumnConfig.DRAFT_FORE, ColumnConfig.DRAFT_AFT]
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
            'beta': [0.1, 0.2],
            'gamma': [0.1, 0.2],
        }

        best_params, best_loss = PINNModel.hyperparameter_search(
            X_train, X_train_unscaled, y_train, feature_indices,
            param_grid, data_processor, k_folds=5,
        )

        X_train_final, X_val_final, X_train_unscaled_final, X_val_unscaled_final, \
            y_train_final, y_val_final = train_test_split(
                X_train, X_train_unscaled, y_train,
                test_size=DataConfig.TEST_SIZE, random_state=DataConfig.RANDOM_STATE,
            )

        final_model = PINNModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],
            epochs=TrainingConfig.EPOCHS_FINAL,
            optimizer_choice=TrainingConfig.OPTIMIZER,
            loss_function_choice=TrainingConfig.LOSS_FUNCTION,
            batch_size=best_params['batch_size'],
            alpha=best_params['alpha'],
            beta=best_params['beta'],
            gamma=best_params['gamma'],
        )

        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)

        final_model.train(
            final_train_loader, X_train_unscaled_final, feature_indices, data_processor,
            val_loader=final_val_loader, live_plot=True, checkpoint_path="best_model_PINN.pt",
        )

        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)
