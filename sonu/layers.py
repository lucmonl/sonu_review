"""LoRA realization of SONU's two-sided sketches, paper Appendix C.

``Theta = W_base + B @ A`` with ``B = P^L`` of shape ``(I, d)`` and
``A = (P^R).T`` of shape ``(d, J)``. The LoRA scaling coefficient is fixed to 1
as Appendix C assumes, and there is no adapter dropout: both are required for
the backward pass to yield exactly the sketches of Eq. (1).
"""

from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .optimizer import solve_dtype


class SketchLinear(nn.Module):
    """A frozen Linear plus the two sketching factors, which carry gradients.

    Eq. (16)-(17): one ordinary backward pass through this module produces
    ``grad_B = G @ A.T`` and ``grad_A = B.T @ G``, i.e. the two sketches
    ``S^L = G P^R`` and ``S^R = (G^T P^L).T``, without materializing the dense
    gradient ``G`` on the worker.
    """

    def __init__(self, base: nn.Linear, rank: int, gamma: float = 1.0,
                 initialize: bool = True):
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("SketchLinear supports torch.nn.Linear only")
        if not 0 < rank <= min(base.weight.shape):
            raise ValueError("rank must be positive and at most min(weight.shape)")
        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError("gamma must be finite and positive")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.rank, self.gamma = rank, gamma
        self.A = nn.Parameter(base.weight.new_zeros(rank, self.in_features))
        self.B = nn.Parameter(base.weight.new_zeros(self.out_features, rank))
        if initialize:
            self.initialize()

    @torch.no_grad()
    def initialize(self):
        """Eqs. (13), (18)-(19): uniform-SV sketches, compensated base weight.

        The leading singular subspaces of the pretrained weight are taken with
        a *uniform* scale gamma rather than the original singular values, which
        keeps the Gram matrices well conditioned for the pseudo-inverses in
        Eqs. (8) and (11). Both factors must be nonzero for the sketches to be
        informative at t=1, so the base weight absorbs the initial product and
        the forward function is unchanged.
        """
        dtype = solve_dtype(self.base.weight.dtype)  # SVD is lifted off BF16
        weight = self.base.weight.to(dtype)
        u, _, vh = torch.linalg.svd(weight, full_matrices=False)
        self.B.copy_(self.gamma * u[:, : self.rank])
        self.A.copy_(self.gamma * vh[: self.rank])
        # Compensate the factors as actually stored, including their rounding.
        self.base.weight.copy_(weight - self.B.to(dtype) @ self.A.to(dtype))

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)

    @torch.no_grad()
    def effective_weight(self):
        """Exact readout of Theta for export and tests, not a training buffer.

        FP64 here only makes the sum of the stored tensors exact; it does not
        add precision the optimizer did not already have.
        """
        return self.base.weight.double() + self.B.double() @ self.A.double()


def attach_sketches(
    model,
    rank=16,
    gamma=1.0,
    targets=("q_proj", "k_proj", "v_proj", "o_proj"),
    initialize=True,
):
    """Freeze every parameter and wrap the requested attention projections.

    Section 3.1 applies two-sided sketching to the self-attention linear
    projections; all other parameters, including embeddings, norms, MLPs and
    the output head, stay frozen.
    """
    selected = [
        (name, module)
        for name, module in model.named_modules()
        if name.rsplit(".", 1)[-1] in targets
    ]
    if not selected:
        raise ValueError(f"No target modules found: {targets}")
    missing = set(targets) - {n.rsplit(".", 1)[-1] for n, _ in selected}
    if missing:
        raise ValueError(f"Target modules missing: {sorted(missing)}")
    for name, module in selected:
        if not isinstance(module, nn.Linear) or rank > min(module.weight.shape):
            raise ValueError(f"Unsupported target or rank: {name}")
    model.requires_grad_(False)
    layers = {}
    for name, module in selected:
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        layer = SketchLinear(module, rank, gamma, initialize)
        setattr(parent, child, layer)
        layers[name] = layer
    return layers


@torch.no_grad()
def merge_sketches(model):
    """Fold the factors into the base weights for standard full-model export.

    SONU updates base weights, so an adapter-only export would be incomplete.
    """
    for name, layer in list(model.named_modules()):
        if not isinstance(layer, SketchLinear):
            continue
        layer.base.weight.copy_(layer.effective_weight())
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child, layer.base)
    return model
