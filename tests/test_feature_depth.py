"""The VGGT-Omega feature correction applied to edge depths and to the depth-residual bank."""

import torch

from depth_refinement.optimization.features import resample_correction
from modules.feature_depth import FeatureDepthModule


def _module(rank=2, target=True):
    torch.manual_seed(0)
    depths = torch.rand(2, 5) + 1
    pixels = torch.rand(2, 5, 2) * torch.tensor([7., 5.])
    features = [torch.randn(4, 3, 4) for _ in range(2)]
    return FeatureDepthModule(
        {"a": 0, "b": 1}, depths, pixels, torch.ones_like(depths, dtype=torch.bool),
        torch.tensor([[6, 8], [6, 8]]), features, rank=rank,
        target_depths=torch.rand(2, 6, 8) + 1 if target else None,
        intrinsics=torch.tensor([[4., 0, 3.5], [0, 4., 2.5], [0, 0, 1]]).repeat(2, 1, 1),
        target_max_resolution=4, device="cpu", warmup_steps=1, max_num_iterations=4,
    ), depths


def test_zero_correction_returns_the_original_depths():
    module, depths = _module()
    edges, bank = module.get_depths_and_target_bank()
    torch.testing.assert_close(edges, depths)
    torch.testing.assert_close(bank, module.target_bank)
    assert module.target_size == (3, 4)
    torch.testing.assert_close(module.query_depth[:, module.edge_count:].reshape(2, 3, 4),
                               module.target_bank)


def test_bank_gets_the_resampled_correction_and_gradients():
    module, _ = _module()
    with torch.no_grad():
        module.model.beta.normal_()
        module.model.log_scale.fill_(0.1)
    edges, bank = module.get_depths_and_target_bank()
    expected = module.target_bank * torch.exp(
        resample_correction(module.model.log_corrections(), module.target_size)
    )
    torch.testing.assert_close(bank, expected)
    torch.testing.assert_close(module.get_parameters(["b"]), edges[1:2])
    _, bank_b = module.get_depths_and_target_bank(["b"])
    torch.testing.assert_close(bank_b, bank[1:2])
    module.optimizer.zero_grad()
    bank.sum().backward()
    assert module.model.beta.grad.abs().sum() > 0 and module.model.log_scale.grad.abs().sum() > 0


def test_rank_zero_is_a_per_image_scale():
    module, depths = _module(rank=0)
    with torch.no_grad():
        module.model.log_scale.copy_(torch.tensor([0.2, -0.1]))
    edges, bank = module.get_depths_and_target_bank()
    scale = torch.exp(torch.tensor([0.2, -0.1]))
    torch.testing.assert_close(edges, depths * scale[:, None])
    torch.testing.assert_close(bank, module.target_bank * scale[:, None, None])


def test_without_targets_there_is_no_bank():
    module, depths = _module(target=False)
    edges, bank = module.get_depths_and_target_bank()
    assert bank is None
    torch.testing.assert_close(edges, depths)
