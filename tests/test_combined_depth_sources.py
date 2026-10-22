"""Uniform samples, edge samples and target maps share one MLP evaluation."""

from unittest.mock import patch
import torch

from modules.point_mlp_depth import PointMLPDepthModule


def test_combined_samples_share_one_mlp_and_target_bank():
    depths = torch.full((2, 4), 3.)
    pixels = torch.tensor([[[0., 0.], [1., 0.], [0., 1.], [1., 1.]]] * 2)
    module = PointMLPDepthModule(
        {"a": 0, "b": 1}, depths, pixels, torch.eye(3).repeat(2, 1, 1),
        torch.ones_like(depths, dtype=torch.bool),
        target_depths=torch.full((2, 3, 3), 3.),
        device="cpu", warmup_steps=1, max_num_iterations=4,
    )
    source, bank = module.get_depths_and_target_bank()
    torch.testing.assert_close(source, depths)
    torch.testing.assert_close(bank, torch.full((2, 3, 3), 3.))
    with patch.object(module.model, "refine", wraps=module.model.refine) as evaluate:
        source, bank = module.get_depths_and_target_bank()
        assert evaluate.call_count == 1
    torch.testing.assert_close(source[:, :2], depths[:, :2])
    torch.testing.assert_close(source[:, 2:], depths[:, 2:])
    torch.testing.assert_close(module.get_parameters(["b"]), source[1:2])
    torch.testing.assert_close(bank, torch.full((2, 3, 3), 3.))
    for loss in (source[:, :2].sum(), source[:, 2:].sum(), bank.sum()):
        module.optimizer.zero_grad()
        loss.backward(retain_graph=True)
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.model.parameters())
