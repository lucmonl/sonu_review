"""SONU: Sketched Orthonormalized Updates (Algorithm 1).

Notation follows the paper. The implementation stores ``B`` with shape
``(I, d)`` and ``A`` with shape ``(d, J)``, so

    paper P^L = B,   P^R = A.T,   M^L = mB,   M^R = mA.T

and the LoRA scaling coefficient is fixed to 1 as in Appendix C, giving
``Theta = W_base + B @ A``.

Equation map
------------
(8)        ``paired_directions``  -- Nystrom momentum surrogate
(10)-(12)  ``update_momenta`` / ``_transport`` -- aligned momentum, residuals
(14), (21) ``apply_update`` -- progressive orthonormalization, compensation
(13), (19) ``layers.SketchLinear.initialize``
(1), (17)  ``layers.SketchLinear.forward`` -- both sketches from one backward

Precision
---------
Persistent optimizer state -- the projected momenta ``M^L, M^R``, the ambient
residuals ``E^L, E^R`` and the sketch snapshots -- is stored in
``SONUConfig.dtype``, **BF16 by default**. Storage dominates optimizer memory
because each residual is ambient ``I x J``.

BF16 cannot carry a pseudo-inverse. Every value is therefore *computed* at the
**solve dtype**, ``max(config.dtype, FP32)``, and rounded once when it is
stored. Eq. (11)'s transport and Eq. (8)'s Nystrom core are compositions with
``(.)^+``, so this promotion covers each pinv/QR/SVD together with the matmuls
feeding and consuming it. One rounding per stored value per round, not one per
intermediate operation. See docs/ALGORITHM.md for the measured cost.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import math
import torch


DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "fp64": torch.float64}


def solve_dtype(dtype):
    """Promotion floor for inverses, decompositions and their matmul chains."""
    return dtype if dtype in (torch.float32, torch.float64) else torch.float32


@dataclass(frozen=True)
class SONUConfig:
    """beta, alpha, S and the optimizer-state dtype of Algorithm 1."""

    momentum: float = 0.9
    evolution_rate: float = 10.0
    switch_interval: int = 50
    dtype: str = "bf16"

    def __post_init__(self):
        if not math.isfinite(self.momentum) or not 0 <= self.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if not math.isfinite(self.evolution_rate) or self.evolution_rate < 0:
            raise ValueError("evolution_rate must be finite and nonnegative")
        if not isinstance(self.switch_interval, int) or self.switch_interval <= 0:
            raise ValueError("switch_interval must be a positive integer")
        if self.dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {sorted(DTYPES)}")

    @property
    def store(self):
        """Dtype of persistent momentum, residual and snapshot tensors."""
        return DTYPES[self.dtype]

    @property
    def solve(self):
        """Dtype every step's linear algebra is evaluated in."""
        return solve_dtype(self.store)


def update_B_at(step, config):
    """Algorithm 1 line 11: evolve P^L when floor(t/S) is even, else P^R.

    ``step`` is one-based, so with S=50 this updates B on rounds 1-49, A on
    50-99, B on 100-149, and so on.
    """
    return (step // config.switch_interval) % 2 == 0


def _promote(solve, *tensors):
    return [None if x is None else x.to(solve) for x in tensors]


def _store(tensor, dtype):
    """Detached private copy at ``dtype``.

    ``Tensor.to`` returns *self* when the dtype already matches, so a plain
    cast would alias a live parameter and be mutated by the in-place factor
    update later in the same round. ``copy=True`` is required, not defensive.
    """
    return tensor.detach().to(dtype, copy=True)


def _gram_inverse(frame):
    """(F^T F)^+ on a frame already promoted to the solve dtype."""
    return torch.linalg.pinv(frame.T @ frame, rcond=1e-6)


def _transport(momentum, previous, current, error, beta):
    """Eqs. (11)-(12). Returns the aligned momentum and the next residual.

    ``momentum`` is (n, d) and both frames are (m, d), so the ambient estimate
    is (n, m). For M^L, n = I and m = J; the M^R call passes the transposes.

        T = M P_previous^+ + E,   M_aligned = beta T P,   E' = beta (T - T P P^+)

    The dense gradient is never reconstructed: T is an ambient *residual*
    estimate, built only from the communicated sketches. Both outputs are at
    the solve dtype; the caller rounds once when storing.
    """
    solve = solve_dtype(momentum.dtype)
    momentum, previous, current, error = _promote(
        solve, momentum, previous, current, error
    )
    previous_inverse = _gram_inverse(previous) @ previous.T
    current_inverse = _gram_inverse(current) @ current.T
    ambient = momentum @ previous_inverse
    if error is not None:
        ambient.add_(error)
    projected = ambient @ current
    residual = ambient.sub_(projected @ current_inverse)
    return beta * projected, beta * residual


def update_momenta(B, A, grad_B, grad_A, state, config):
    """Eq. (10): transport onto the current sketch, then add the new sketch.

    The alignment always uses the factor that moved in the *previous* round,
    before the next factor is chosen; this ordering matters at a switch.
    ``B``/``A`` and the gradients arrive at the solve dtype; the returned
    momenta are the stored tensors, rounded once to ``config.store``.
    """
    beta, store, solve = config.momentum, config.store, config.solve
    if not state:
        # Eq. (7) with M_0 = 0 and E_0 = 0.
        state.update(
            mB=_store(grad_B, store), mA=_store(grad_A, store), eB=None, eA=None
        )
    else:
        mB, mA = beta * state["mB"].to(solve), beta * state["mA"].to(solve)
        if state["last_update_B"]:
            # P^L moved, so M^R is transported and E^R takes the new residual
            # while E^L only decays. Eq. (12) in transposed coordinates.
            error = None if state["eA"] is None else state["eA"].T
            mA, error = _transport(state["mA"].T, state["prev_B"], B, error, beta)
            mA, state["eA"] = mA.T, error.T.to(store)
            if state["eB"] is not None:
                state["eB"] = (beta * state["eB"].to(solve)).to(store)
        else:
            mB, error = _transport(
                state["mB"], state["prev_A"].T, A.T, state["eB"], beta
            )
            state["eB"] = error.to(store)
            if state["eA"] is not None:
                state["eA"] = (beta * state["eA"].to(solve)).to(store)
        state["mB"], state["mA"] = (mB + grad_B).to(store), (mA + grad_A).to(store)
    # Snapshot the frames this round's gradient was taken through, so the next
    # round aligns against them. Exact when config.dtype is the model dtype.
    state["prev_B"], state["prev_A"] = _store(B, store), _store(A, store)
    return state["mB"].to(solve), state["mA"].to(solve)


def paired_directions(B, mB, mA):
    """Eq. (8): polar factors of M^L (B^T M^L)^dagger M^R via two thin QRs.

    The small inverse is absorbed into the right factor before QR, which is
    algebraically the same surrogate as QR-ing both sketches. Only d-by-d
    matrices are inverted and decomposed; the dense surrogate is never formed.
    U and V are paired, and their joint sign convention matters because the
    sketch evolution in Eq. (14) consumes them.
    """
    solve = solve_dtype(mB.dtype)
    B, mB, mA = B.to(solve), mB.to(solve), mA.to(solve)
    cross_inverse = torch.linalg.pinv(B.T @ mB)
    left_q, left_r = torch.linalg.qr(mB, mode="reduced")
    right_q, right_r = torch.linalg.qr((cross_inverse @ mA).T, mode="reduced")
    core = left_r @ right_r.T
    u, singular_values, vh = torch.linalg.svd(core, full_matrices=False)
    U, Vh = left_q @ u, (right_q @ vh.T).T
    # mSign(0) = 0: do not move along arbitrary null-space completions.
    cutoff = max(core.shape) * torch.finfo(core.dtype).eps * singular_values.max()
    keep = (singular_values > cutoff).to(U.dtype)
    return U * keep, Vh * keep[:, None]


@torch.no_grad()
def apply_update(layer, U, Vh, lr, config, update_B):
    """Eqs. (14) and (25)-(28): evolve one sketch, compensate the base weight.

    The effective weight moves by exactly ``-lr * U @ Vh`` (Eq. 20); the extra
    rank-d term cancels the change in the auxiliary product B @ A.
    """
    solve = config.solve
    B, A = layer.B.to(solve).clone(), layer.A.to(solve).clone()
    U, Vh = U.to(solve), Vh.to(solve)
    alpha = config.evolution_rate
    if update_B:
        moved = B - alpha * lr * U
        B_new, A_new = layer.gamma * torch.linalg.qr(moved, mode="reduced")[0], A
    else:
        moved = A - alpha * lr * Vh
        A_new, B_new = layer.gamma * torch.linalg.qr(moved.T, mode="reduced")[0].T, B
    layer.B.copy_(B_new)
    layer.A.copy_(A_new)
    # Re-read the factors as actually stored, so Eq. (21) holds up to the base
    # weight's own rounding rather than up to the pre-rounding values.
    B_new, A_new = layer.B.to(solve), layer.A.to(solve)
    weight = layer.base.weight
    for start in range(0, B.shape[0], 256):  # chunked: never form a dense I x J
        rows = slice(start, start + 256)
        delta = -lr * (U[rows] @ Vh)
        if update_B:
            delta.add_((B[rows] - B_new[rows]) @ A)
        else:
            delta.add_(B[rows] @ (A - A_new))
        weight[rows].copy_(weight[rows].to(solve) + delta)


class SONU:
    """Server-side state for Algorithm 1, one entry per sketched layer.

    Deliberately not a ``torch.optim.Optimizer``: the standard
    ``load_state_dict`` casts optimizer state to the parameter dtype, which
    would silently retype the ambient residual buffers carrying Eq. (12).
    """

    def __init__(self, layers, config=SONUConfig()):
        if not layers:
            raise ValueError("At least one sketch layer is required")
        self.layers, self.config = dict(layers), config
        self.state = {name: {} for name in layers}
        self.steps = 0

    @torch.no_grad()
    def step(self, lr, gradients=None):
        """One global round: average already done, Algorithm 1 lines 8-12."""
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        if gradients is None:
            gradients = {n: (l.B.grad, l.A.grad) for n, l in self.layers.items()}
        if set(gradients) != set(self.layers):
            raise ValueError("Gradient names do not match layer names")
        # Validate every layer before mutating any optimizer or model state.
        for name, layer in self.layers.items():
            for gradient, parameter in zip(gradients[name], (layer.B, layer.A)):
                if gradient is None or gradient.shape != parameter.shape:
                    raise ValueError(f"Missing or invalid sketch gradient: {name}")
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError(f"Non-finite sketch gradient: {name}")
        step = self.steps + 1
        solve = self.config.solve
        update_B = update_B_at(step, self.config)
        for name, layer in self.layers.items():
            B, A = layer.B.to(solve), layer.A.to(solve)
            gB, gA = [g.to(device=B.device, dtype=solve) for g in gradients[name]]
            state = self.state[name]
            mB, mA = update_momenta(B, A, gB, gA, state, self.config)
            U, Vh = paired_directions(B, mB, mA)
            apply_update(layer, U, Vh, lr, self.config, update_B)
            state["last_update_B"] = update_B
        self.steps = step

    def zero_grad(self):
        for layer in self.layers.values():
            layer.A.grad = layer.B.grad = None

    def state_dict(self):
        layers = {
            name: {
                key: value.detach().cpu().clone() if torch.is_tensor(value) else value
                for key, value in state.items()
            }
            for name, state in self.state.items()
        }
        return {"config": asdict(self.config), "steps": self.steps, "layers": layers}

    def load_state_dict(self, saved):
        if saved["config"] != asdict(self.config) or set(saved["layers"]) != set(
            self.layers
        ):
            raise ValueError(
                "Checkpoint optimizer configuration or layers do not match"
            )
        restored = copy.deepcopy(saved["layers"])
        for name, state in restored.items():
            device = self.layers[name].A.device
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    # Move only: the saved dtype is the configured storage
                    # dtype, which the config check above already matched.
                    state[key] = value.to(device=device)
        self.state, self.steps = restored, int(saved["steps"])
