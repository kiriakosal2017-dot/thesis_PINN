"""
Pure data-driven baseline (DATA): a standard MLP trained solely on observed shaft-power
measurements. No physics terms are included, making this the weakest prior but also the
most flexible benchmark against which physics-informed variants are compared.
"""

import copy
import torch
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from config import DataConfig, TrainingConfig
from read_data import DataProcessor
from base_model import BaseModel


class DataDrivenModel(BaseModel):
    """Pure data-driven model: standard MLP with MSE/MAE loss, no physics."""

    def train(self, train_loader, val_loader=None, live_plot=False, checkpoint_path=None,
              history_csv=None):
        optimizer = self.get_optimizer()
        loss_function = self.get_loss_function()

        train_losses = []
        val_losses = []
        best_state = None
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        # Early stopping guards against overfitting on ship operational data,
        # which is often noisy and auto-correlated.
        patience = TrainingConfig.EARLY_STOPPING_PATIENCE
        min_delta = TrainingConfig.EARLY_STOPPING_MIN_DELTA

        if live_plot:
            plt.ion()
            fig, ax = plt.subplots()

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0

            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{self.epochs}",
                leave=False,
            )

            # Standard supervised forward-backward pass; no physics terms.
            for X_batch, y_batch in progress_bar:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                data_loss = loss_function(outputs, y_batch)
                data_loss.backward()
                optimizer.step()

                running_loss += data_loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            if val_loader is not None:
                # Evaluate on validation set without gradient tracking.
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
                print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {avg_train_loss:.8f}, "
                      f"Validation Loss: {avg_val_loss:.8f}")

                # Keep the snapshot with the lowest validation loss seen so far;
                # improvement must exceed min_delta to avoid saving on noise.
                if (best_val_loss - avg_val_loss) > min_delta:
                    best_val_loss = avg_val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_without_improvement = 0
                    if checkpoint_path is not None:
                        torch.save(best_state, checkpoint_path)
                else:
                    epochs_without_improvement += 1
            else:
                val_losses.append(None)
                print(f"Epoch [{epoch+1}/{self.epochs}], Training Loss: {avg_train_loss:.8f}")

            if live_plot:
                ax.clear()
                ax.plot(range(1, epoch + 2), train_losses, label='Training Loss')
                if val_loader is not None:
                    ax.plot(range(1, epoch + 2), val_losses, label='Validation Loss')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title('Training and Validation Loss over Epochs')
                ax.legend()
                plt.pause(0.01)

            if val_loader is not None and epochs_without_improvement >= patience:
                print(
                    f"Early stopping at epoch {epoch+1}: "
                    f"no Validation improvement > {min_delta} for {patience} epochs."
                )
                break

        if live_plot:
            plt.ioff()
            plt.show()
            fig.savefig('training_validation_loss_plot_DATA.png')

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

        # Restore the best checkpoint so the caller gets the lowest-val-loss weights.
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"Restored best model state (Validation Loss: {best_val_loss:.8f})")

    def cross_validate(self, X, y, data_processor, k_folds=5):
        # Stratified k-fold is not used because the target is continuous;
        # a fixed random_state ensures reproducibility across hyperparameter trials.
        kfold = KFold(n_splits=k_folds, shuffle=True, random_state=DataConfig.RANDOM_STATE)
        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold {fold+1}/{k_folds}")

            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            train_loader = self.prepare_dataloader(X_train_fold, y_train_fold)
            val_loader = self.prepare_dataloader(X_val_fold, y_val_fold)

            # Reset weights before each fold so folds are independent.
            self.model.apply(self.reset_weights)
            self.train(train_loader, val_loader=val_loader, live_plot=False)

            val_loss = self.evaluate(X_val_fold, y_val_fold, dataset_type="Validation",
                                     data_processor=data_processor)
            fold_results.append(val_loss)

        avg_val_loss = np.mean(fold_results)
        print(f"\nCross-validation results: Average Validation Loss = {avg_val_loss:.8f}")
        return avg_val_loss

    @staticmethod
    def hyperparameter_search(X_train, y_train, param_grid, data_processor, k_folds=5):
        best_params = None
        best_loss = float('inf')

        # Exhaustive grid search over lr × batch_size; small grid kept tractable
        # by using a reduced epoch budget (EPOCHS_CV) during cross-validation.
        hyperparameter_combinations = list(product(
            param_grid['lr'],
            param_grid['batch_size'],
        ))

        for lr, batch_size in hyperparameter_combinations:
            print(f"\nTesting combination: lr={lr}, batch_size={batch_size}")

            model = DataDrivenModel(
                input_size=X_train.shape[1],
                lr=lr,
                epochs=TrainingConfig.EPOCHS_CV,
                optimizer_choice=TrainingConfig.OPTIMIZER,
                loss_function_choice=TrainingConfig.LOSS_FUNCTION,
                batch_size=batch_size,
            )

            avg_val_loss = model.cross_validate(X_train, y_train, data_processor, k_folds=k_folds)

            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                best_params = {'lr': lr, 'batch_size': batch_size}

        print(f"\nBest parameters: {best_params}, with average validation loss: {best_loss:.8f}")

        with open("best_hyperparameters_DATA.txt", "w") as f:
            f.write(f"Best parameters: {best_params}\n")
            f.write(f"Best average validation loss: {best_loss:.8f}\n")

        return best_params, best_loss


if __name__ == "__main__":
    data_processor = DataProcessor()
    result = data_processor.load_and_prepare_data()

    if result is not None:
        X_train, X_test, X_train_unscaled, X_test_unscaled, \
            y_train, y_test, y_train_unscaled, y_test_unscaled = result

        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")

        param_grid = {
            'lr': [0.001, 0.01],
            'batch_size': [64, 128],
        }

        best_params, best_loss = DataDrivenModel.hyperparameter_search(
            X_train, y_train, param_grid, data_processor, k_folds=5
        )

        # Chronological validation split (no shuffle) to match the temporal protocol
        # used everywhere else and avoid leaking future rows into validation.
        X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
            X_train, y_train, test_size=DataConfig.TEST_SIZE, shuffle=False
        )

        final_model = DataDrivenModel(
            input_size=X_train.shape[1],
            lr=best_params['lr'],
            epochs=TrainingConfig.EPOCHS_FINAL,
            optimizer_choice=TrainingConfig.OPTIMIZER,
            loss_function_choice=TrainingConfig.LOSS_FUNCTION,
            batch_size=best_params['batch_size'],
        )

        final_train_loader = final_model.prepare_dataloader(X_train_final, y_train_final)
        final_val_loader = final_model.prepare_dataloader(X_val_final, y_val_final)

        final_model.train(
            final_train_loader,
            val_loader=final_val_loader,
            live_plot=True,
            checkpoint_path="best_model_DATA.pt",
        )
        final_model.evaluate(X_test, y_test, dataset_type="Test", data_processor=data_processor)
