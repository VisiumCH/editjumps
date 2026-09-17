"""The Edit Flows training loss (arXiv 2506.09018 appendix; EvoFlows arXiv 2603.11703, eq. 6)."""

# Annotations reference names imported only under TYPE_CHECKING below.
from __future__ import annotations

from typing import TYPE_CHECKING

from editjumps.core.edit_flows.targets import edit_targets

if TYPE_CHECKING:
    import torch


def edit_flow_loss(
    z_t: list[int],
    z_1: list[int],
    kappa: float,
    dkappa: float,
    insert_lambda: "torch.Tensor",
    insert_q: "torch.Tensor",
    delete_lambda: "torch.Tensor",
    substitute_lambda: "torch.Tensor",
    substitute_q: "torch.Tensor",
    *,
    supervision: str = "eq23",
) -> "torch.Tensor":
    """Return the scalar loss for one aligned example. ``supervision`` selects which reading of the."""
    import torch

    # Term 1: Total exit rate sum across all positions
    loss = insert_lambda.sum() + delete_lambda.sum() + substitute_lambda.sum()

    # Term 2: Weighted cross-entropy on target transition rates (weight = dκ / (1 - κ))
    weight = dkappa / (1.0 - kappa)
    for target in edit_targets(z_t, z_1, supervision):
        i = target.index
        if target.op == "insert":
            rate = insert_lambda[i] * insert_q[i, target.token]
        elif target.op == "delete":
            rate = delete_lambda[i]
        else:  # substitute
            rate = substitute_lambda[i] * substitute_q[i, target.token]
        loss = loss - weight * torch.log(rate.clamp_min(1e-30))
    return loss
