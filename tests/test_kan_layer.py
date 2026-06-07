"""Unit checks for the self-contained B-spline KAN layer and network."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kan_layer import KAN, KANLinear


def test_forward_shape():
    # Output tensor must match the declared output width for a typical batch.
    net = KAN([7, 16, 1], grid_size=5, spline_order=3)
    x = torch.randn(32, 7)
    y = net(x)
    assert y.shape == (32, 1), f"expected (32,1), got {tuple(y.shape)}"


def test_first_order_grad_flows():
    # First-order gradients must be finite; required for any gradient-based training.
    net = KAN([7, 16, 1])
    x = torch.randn(8, 7, requires_grad=True)
    y = net(x).sum()
    g = torch.autograd.grad(y, x, create_graph=True)[0]
    assert g.shape == x.shape
    assert torch.isfinite(g).all(), "non-finite first-order grad"


def test_second_order_grad_flows():
    # compute_pde_residual takes d/dx with create_graph=True then backprops again;
    # spline weights must receive gradients through the second-order path.
    net = KAN([7, 16, 1])
    x = torch.randn(8, 7, requires_grad=True)
    out = net(x)
    g = torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out),
                            create_graph=True)[0]
    loss = (g ** 2).mean()
    loss.backward()  # 2nd-order path; must not raise
    assert net.layers[0].spline_weight.grad is not None, "no grad on spline_weight"


def test_params_present_and_trainable():
    # A layer with no trainable parameters would be unusable; guard against silent misconfiguration.
    layer = KANLinear(7, 16)
    n = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    assert n > 0


def test_out_of_range_inputs_finite():
    # Design relies on far-out-of-grid inputs collapsing to the SiLU base path;
    # verify outputs and grads stay finite for inputs well outside [-4, 4].
    net = KAN([7, 16, 1])
    x = torch.randn(8, 7, requires_grad=True) * 10.0
    out = net(x)
    assert torch.isfinite(out).all(), "non-finite output for out-of-range input"
    g = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    assert torch.isfinite(g).all(), "non-finite grad for out-of-range input"


if __name__ == "__main__":
    test_forward_shape()
    test_first_order_grad_flows()
    test_second_order_grad_flows()
    test_params_present_and_trainable()
    test_out_of_range_inputs_finite()
    print("all KAN layer tests passed")
