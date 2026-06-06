"""Smoke test: PIKANModel reuses HYBRID physics with a KAN backbone (no pytest)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kan_layer import KAN
from main_PI_KAN import PIKANModel


def test_backbone_is_kan():
    m = PIKANModel(input_size=7, kan_width=[7, 16, 1], epochs=1, batch_size=8)
    assert isinstance(m.model, KAN), f"backbone is {type(m.model)}, expected KAN"


def test_forward_and_physics_losses_backprop():
    import pandas as pd
    import numpy as np
    from config import ColumnConfig
    # Minimal synthetic batch matching the physics feature columns HYBRID needs.
    cols = [ColumnConfig.SPEED, ColumnConfig.DRAFT_FORE, ColumnConfig.DRAFT_AFT]
    n = 16
    X = pd.DataFrame(np.random.rand(n, len(cols)), columns=cols)
    m = PIKANModel(input_size=len(cols), kan_width=[len(cols), 16, 1],
                   epochs=1, batch_size=8)
    fi = {c: i for i, c in enumerate(cols)}
    xb = torch.tensor(X.values, dtype=torch.float32, device=m.device)
    pred = m.model(xb)
    assert pred.shape == (n, 1)
    # PDE residual exercises the 2nd-order autograd path through the KAN.
    x_col = m.sample_collocation_points(8, X, _FakeProc())
    res = m.compute_pde_residual(x_col, fi)
    res.pow(2).mean().backward()
    print("forward + PDE backward OK")


class _FakeProc:
    """Stand-in scaler_X that is identity, just for collocation sampling shape."""
    class _S:
        def transform(self, df):
            return df.values
    scaler_X = _S()


if __name__ == "__main__":
    test_backbone_is_kan()
    test_forward_and_physics_losses_backprop()
    print("ALL PIKAN MODEL SMOKE TESTS PASSED")
