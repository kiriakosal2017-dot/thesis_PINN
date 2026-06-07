"""Regression tests for the PI-KAN collapse/seed fixes.

Guards against two bugs:
  #1 PI-KAN trained on MPS collapsed to a constant (mean-power) predictor because the
     Apple-MPS backend miscomputes the KAN B-spline double-backward. force_cpu=True
     makes CPU the default device. (No-collapse itself is verified by the integration
     re-run; here we only assert the device default that fixes it.)
  #2 BaseModel.__init__ called set_global_seed(RANDOM_STATE), silently overriding the
     caller seed, so every "multi-seed" run was really seed 42. The seed= constructor
     arg is now applied AFTER super().__init__, so seeds genuinely differ.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from main_PI_KAN import PIKANModel


def _flat_weights(m):
    return torch.cat([p.detach().flatten() for p in m.model.parameters()])


def test_force_cpu_is_default():
    # MPS miscomputes KAN double-backward; CPU must be the default device to avoid silent collapse.
    m = PIKANModel(input_size=7, kan_width=[7, 16, 1])
    assert m.device.type == "cpu", f"expected cpu, got {m.device}"


def test_different_seeds_give_different_init():
    # Different seeds must produce different initial weights; identical weights indicate the seed override bug has regressed.
    a = _flat_weights(PIKANModel(input_size=7, kan_width=[7, 16, 1], seed=0))
    b = _flat_weights(PIKANModel(input_size=7, kan_width=[7, 16, 1], seed=1))
    assert a.shape == b.shape
    assert not torch.allclose(a, b), \
        "seed=0 and seed=1 produced identical init -> seed not applied (bug #2 regressed)"


def test_same_seed_is_reproducible():
    # The same seed must yield identical initial weights across separate instantiations.
    a = _flat_weights(PIKANModel(input_size=7, kan_width=[7, 16, 1], seed=3))
    b = _flat_weights(PIKANModel(input_size=7, kan_width=[7, 16, 1], seed=3))
    assert torch.allclose(a, b), "same seed gave different init (non-reproducible)"


if __name__ == "__main__":
    test_force_cpu_is_default()
    test_different_seeds_give_different_init()
    test_same_seed_is_reproducible()
    print("all PI-KAN fix regression tests passed")
