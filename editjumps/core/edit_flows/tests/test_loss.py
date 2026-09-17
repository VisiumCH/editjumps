"""Eq. 6 on a toy model: the supervised edits are the ones whose rate goes up."""

import pytest


def test_edit_flow_loss_trains_a_toy_model_to_recover_edits() -> None:
    """The loss is differentiable and, minimized, raises the rate of each correct edit."""
    torch = pytest.importorskip("torch")
    from editjumps.core.edit_flows.loss import edit_flow_loss
    from editjumps.core.edit_flows.path import strip_epsilon

    eps = -1
    vocab = 6
    # x_t of length 4 (a BOS + 3 tokens); target needs one insert, one sub, one delete.
    z_t = [0, 1, eps, 3, 4]  # x_t = [0, 1, 3, 4]
    z_1 = [0, 1, 2, 5, eps]  # insert 2 after idx1, substitute idx2 -> 5, delete idx3
    length = len(strip_epsilon(z_t))

    torch.manual_seed(0)
    # Free per-position parameters: rate logits (-> softplus) and token logits (-> softmax).
    ins_logit = torch.zeros(length, requires_grad=True)
    del_logit = torch.zeros(length, requires_grad=True)
    sub_logit = torch.zeros(length, requires_grad=True)
    ins_qlogit = torch.zeros(length, vocab, requires_grad=True)
    sub_qlogit = torch.zeros(length, vocab, requires_grad=True)
    opt = torch.optim.Adam([ins_logit, del_logit, sub_logit, ins_qlogit, sub_qlogit], lr=0.1)
    softplus = torch.nn.functional.softplus

    initial = 0.0
    for step in range(300):
        opt.zero_grad()
        loss = edit_flow_loss(
            z_t, z_1, 0.5, 1.0,
            softplus(ins_logit), torch.softmax(ins_qlogit, dim=-1),
            softplus(del_logit), softplus(sub_logit), torch.softmax(sub_qlogit, dim=-1),
        )
        if step == 0:
            initial = loss.item()
        loss.backward()
        opt.step()
    with torch.no_grad():
        final = edit_flow_loss(
            z_t, z_1, 0.5, 1.0,
            softplus(ins_logit), torch.softmax(ins_qlogit, dim=-1),
            softplus(del_logit), softplus(sub_logit), torch.softmax(sub_qlogit, dim=-1),
        ).item()
    assert final < initial - 1.0  # loss dropped substantially
    # the correct insert/substitute tokens became the argmax of their distributions
    assert int(torch.softmax(ins_qlogit, dim=-1)[1].argmax()) == 2
    assert int(torch.softmax(sub_qlogit, dim=-1)[2].argmax()) == 5

def test_edit_flow_loss_clamps_the_log() -> None:
    """A zero rate must not produce -inf: the loss clamps the log's argument."""
    torch = pytest.importorskip("torch")

    from editjumps.core.edit_flows.loss import edit_flow_loss
    from editjumps.core.edit_flows.path import EPS

    n, vocab = 3, 5
    zeros = torch.zeros(n)
    q = torch.full((n, vocab), 1.0 / vocab)
    loss = edit_flow_loss(
        z_t=[1, 2, 3], z_1=[1, 9 % vocab, EPS], kappa=0.5, dkappa=1.0,
        insert_lambda=zeros, insert_q=q, delete_lambda=zeros,
        substitute_lambda=zeros, substitute_q=q,
    )
    assert torch.isfinite(loss), "zero rate produced a non-finite loss; the log is unclamped"
