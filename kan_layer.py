"""Self-contained B-spline Kolmogorov-Arnold Network (efficient-kan formulation).

Pure PyTorch, no external dependency (pykan/efficient_kan are not installed). Used as
the backbone of the PI-KAN baseline (main_PI_KAN.py) so it must support higher-order
autograd: PINODE-style PDE residuals take d(output)/d(input) with create_graph=True and
then backpropagate through it.

grid_range defaults to (-4, 4) because inputs are StandardScaler-normalized (roughly
[-4, 4]); a tighter range would push most samples outside the spline support, collapsing
the KAN to its base (SiLU) path. We do NOT adapt the grid during training (no update_grid)
to keep the model deterministic across seeds.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=nn.SiLU, grid_range=(-4.0, 4.0)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = base_activation()
        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * scale_noise
                / self.grid_size
            )
            coeff = self._curve2coeff(
                self.grid.T[self.spline_order : -self.spline_order], noise
            )
            self.spline_weight.data.copy_(self.scale_spline * coeff)

    def b_splines(self, x):
        """x: (batch, in_features) -> (batch, in_features, grid_size + spline_order)."""
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )
        return bases.contiguous()

    def _curve2coeff(self, x, y):
        """Least-squares fit of spline coefficients to (x, y). Used only at init."""
        A = self.b_splines(x).transpose(0, 1)          # (in, batch, grid+order)
        B = y.transpose(0, 1)                           # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution    # (in, grid+order, out)
        return solution.permute(2, 0, 1).contiguous()   # (out, in, grid+order)

    def forward(self, x):
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output


class KAN(nn.Module):
    """Stack of KANLinear layers. layers_hidden = [in, h1, ..., out]."""

    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        self.layers = nn.ModuleList(
            KANLinear(in_f, out_f, grid_size=grid_size, spline_order=spline_order)
            for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
