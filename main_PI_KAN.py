"""PI-KAN baseline: the HYBRID physics-informed model with a KAN backbone.

PIKANModel subclasses UnifiedPhysicsHybridModel and replaces ONLY the backbone
(MLP -> KAN). All physics, training, dataloaders, and evaluation are inherited unchanged,
isolating MLP-vs-KAN as the single variable in the comparison with PI-NODE.
"""
from sklearn.model_selection import train_test_split

from config import DataConfig, TrainingConfig
from kan_layer import KAN
from main_HYBRID import UnifiedPhysicsHybridModel, _build_feature_indices
from read_data import DataProcessor


class PIKANModel(UnifiedPhysicsHybridModel):
    def __init__(self, input_size, kan_width=None, grid_size=5, spline_order=3,
                 lr=0.001, epochs=1000, batch_size=64,
                 optimizer_choice="Adam", loss_function_choice="MSE",
                 alpha=1.0, beta=0.05, gamma=0.05, delta=0.02):
        # Build HYBRID (which builds an MLP we immediately discard) to inherit all
        # physics + training state, then swap in the KAN backbone.
        super().__init__(
            input_size=input_size, lr=lr, epochs=epochs, batch_size=batch_size,
            optimizer_choice=optimizer_choice, loss_function_choice=loss_function_choice,
            alpha=alpha, beta=beta, gamma=gamma, delta=delta,
        )
        self.kan_width = kan_width or [input_size, 64, 32, 1]
        if self.kan_width[0] != input_size or self.kan_width[-1] != 1:
            raise ValueError(
                f"kan_width must start with input_size={input_size} and end with 1; "
                f"got {self.kan_width}"
            )
        self.model = KAN(self.kan_width, grid_size=grid_size, spline_order=spline_order).to(
            self.device
        )

    def n_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def cross_validate(self, *args, **kwargs):
        # Inherited cross_validate relies on BaseModel.reset_weights, which only
        # resets nn.Linear layers and would NOT reset the KAN backbone between
        # folds. The PI-KAN comparison uses a single chronological train/val split
        # (see evaluate_pikan.py / train_multiseed_pikan.py), not k-fold CV.
        raise NotImplementedError(
            "cross_validate is not supported for PIKANModel (reset_weights does not "
            "reset KAN layers). Use a single chronological train/val split instead."
        )


if __name__ == "__main__":
    dp = DataProcessor()
    result = dp.load_and_prepare_data()
    if result is None:
        raise RuntimeError("Failed to load data")
    X_train, X_test, X_train_unscaled, X_test_unscaled, y_train, y_test, _, _ = result
    feature_indices = _build_feature_indices(X_train_unscaled)

    in_size = X_train.shape[1]
    model = PIKANModel(
        input_size=in_size,
        kan_width=[in_size, 64, 32, 1],
        lr=1e-3,
        epochs=TrainingConfig.EPOCHS_FINAL,
        batch_size=64,
    )
    print(f"PI-KAN trainable params: {model.n_params()}")

    X_tr, X_val, X_tr_un, _, y_tr, y_val = train_test_split(
        X_train, X_train_unscaled, y_train, test_size=DataConfig.TEST_SIZE, shuffle=False
    )
    train_loader = model.prepare_combined_dataloader(X_tr, X_tr_un, y_tr, shuffle=True)
    val_loader = model.prepare_dataloader(X_val, y_val)
    model.train(
        train_loader, X_tr_un, feature_indices, dp,
        val_loader=val_loader, checkpoint_path="best_model_PI_KAN_danae.pt",
    )
    model.evaluate(X_test, y_test, dataset_type="Test", data_processor=dp)
