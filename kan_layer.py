"""Self-contained B-spline Kolmogorov–Arnold Network (efficient-kan formulation).

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
    """Single KAN layer: maps (batch, in_features) -> (batch, out_features).

    Each output unit is a sum over all input dimensions, where each univariate
    contribution is base_activation(x) + learnable_spline(x) — the "base + spline"
    decomposition from Liu et al. 2024 (efficient-kan variant).
    """

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

        # Build the uniform knot vector. A B-spline of order k defined on grid_size
        # intervals needs grid_size + 2*k + 1 knots total: k "ghost" knots are
        # extended on each side so that the basis is non-zero at the boundary.
        # Each input feature gets its own row, so grid is (in_features, n_knots).
        h = (grid_range[1] - grid_range[0]) / grid_size
        if h <= 0:
            raise ValueError(
                f"grid spacing must be positive; got grid_range={grid_range}, grid_size={grid_size}"
            )
        grid = (
            (torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0])
            .expand(in_features, -1)
            .contiguous()
        )
        # Register as a buffer so it moves with .to(device) but is not a parameter.
        self.register_buffer("grid", grid)

        # base_weight: standard linear projection applied to base_activation(x).
        # spline_weight: B-spline coefficients, shape (out, in, grid_size + spline_order).
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        self.base_activation = base_activation()
        self.reset_parameters(scale_noise)

    def reset_parameters(self, scale_noise):
        # Kaiming init for the base path keeps the variance-preserving property
        # across the linear projection despite the nonlinear base_activation.
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            # Initialise spline weights by fitting a small random function on the
            # interior knots so the spline path is non-trivial from the first step.
            # interior knots, shape (grid_size + 1, in_features)
            interior_grid = self.grid.T[self.spline_order : -self.spline_order]
            noise = (
                (torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                * scale_noise
                / self.grid_size
            ).to(self.grid.device)
            coeff = self._curve2coeff(interior_grid, noise)
            self.spline_weight.data.copy_(self.scale_spline * coeff)

    def b_splines(self, x):
        """Evaluate the B-spline basis at x via the de Boor recurrence.

        x: (batch, in_features) -> returns (batch, in_features, grid_size + spline_order).

        The recurrence builds from order-0 indicator bases (piecewise constant) up to
        order spline_order using the standard Cox–de Boor formula:
            B_{i,k}(x) = (x - t_i)/(t_{i+k} - t_i) * B_{i,k-1}(x)
                        + (t_{i+k+1} - x)/(t_{i+k+1} - t_{i+1}) * B_{i+1,k-1}(x)
        Each iteration reduces the number of basis functions by 1 (from N+1 to N),
        ending at grid_size + spline_order functions — exactly spline_weight's last dim.
        The result must retain the computation graph (no detach) so that second-order
        autograd through the physics residuals can differentiate through the basis.
        """
        grid = self.grid
        x = x.unsqueeze(-1)
        # Order-0: indicator function — 1 if x falls in interval [t_i, t_{i+1}).
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        # Iteratively elevate the spline order using the de Boor recurrence.
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
        # contiguous() ensures the tensor layout is compact for subsequent matmuls.
        return bases.contiguous()

    def _curve2coeff(self, x, y):
        """Least-squares fit of spline coefficients to (x, y). Used only at init."""
        A = self.b_splines(x).transpose(0, 1)          # (in, batch, grid+order)
        B = y.transpose(0, 1)                           # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution    # (in, grid+order, out)
        return solution.permute(2, 0, 1).contiguous()   # (out, in, grid+order)

    def forward(self, x):
        # Base path: SiLU-activated linear projection provides a smooth, globally
        # defined residual that stabilises training when x falls near knot boundaries.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        # Spline path: flatten the (in_features, n_basis) basis matrix to a vector
        # per sample so that a single F.linear call computes all (out, in) dot products.
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1),
        )
        # The additive combination lets the base handle regions outside the spline
        # support while the spline refines the fit inside.
        return base_output + spline_output


class KAN(nn.Module):
    """Stack of KANLinear layers forming a full Kolmogorov–Arnold Network.

    layers_hidden lists the width at each stage: [in, h1, h2, ..., out].
    All layers share the same grid_size and spline_order.
    """

    def __init__(self, layers_hidden, grid_size=5, spline_order=3):
        super().__init__()
        if len(layers_hidden) < 2:
            raise ValueError(
                f"layers_hidden must list at least [in, out]; got {layers_hidden}"
            )
        self.layers = nn.ModuleList(
            KANLinear(in_f, out_f, grid_size=grid_size, spline_order=spline_order)
            for in_f, out_f in zip(layers_hidden, layers_hidden[1:])
        )

    def forward(self, x):
        # Sequential pass through KANLinear layers; no skip connections.
        for layer in self.layers:
            x = layer(x)
        return x
