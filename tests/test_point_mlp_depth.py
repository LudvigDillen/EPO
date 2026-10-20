"""CPU checks for the sparse XYZ depth representation (no EPO run needed)."""

import torch
from unittest.mock import patch

from modules.point_mlp_depth import PointMLPDepthModule


def make_module(**kwargs):
    depth = torch.tensor([[1.0, 2.0, 1000.0], [3.0, 4.0, 1000.0]])
    pixels = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [50.0, 50.0]]] * 2)
    intrinsics = torch.eye(3).expand(2, -1, -1).clone()
    valid = torch.tensor([[True, True, False]] * 2)
    module = PointMLPDepthModule(
        {"first": 0, "second": 1}, depth, pixels, intrinsics, valid,
        warmup_steps=1, max_num_iterations=4, device="cpu",
        **kwargs,
    )
    return module, depth


def test_identity_and_fixed_inputs():
    module, depth = make_module()
    assert torch.equal(module.get_parameters(None), depth)
    assert torch.equal(module.get_parameters(["second"]), depth[1:2])
    assert torch.equal(module.get_all_parameters(), depth)
    assert module.centers[0, 2] < 10  # the padded 1000-depth point was excluded
    fixed_points = module.normalized_points.clone()
    module.get_parameters(None).sum().backward()
    module.optimizer_and_scheduler_step()
    assert torch.equal(module.normalized_points, fixed_points)


def test_bounded_positive_correction_and_padding():
    module, depth = make_module()
    with torch.no_grad():
        module.model.biases[-1][0] = 100
        module.model.biases[-1][1] = -100
    refined = module.get_parameters(None)
    torch.testing.assert_close(refined[0, :2], 2 * depth[0, :2])
    torch.testing.assert_close(refined[1, :2], 0.5 * depth[1, :2])
    assert torch.equal(refined[:, 2], depth[:, 2])


def test_images_have_independent_parameters():
    module, _ = make_module()
    module.get_parameters([0]).sum().backward()
    assert module.model.biases[-1].grad[0].abs().sum() > 0
    assert module.model.biases[-1].grad[1].abs().sum() == 0
    assert len(module.parameters()) == 12


def test_deterministic_initialization():
    first, _ = make_module()
    second, _ = make_module()
    for a, b in zip(first.parameters(), second.parameters()):
        assert torch.equal(a, b)


def test_target_bank_identity_mask_and_single_evaluation():
    targets = torch.full((2, 4, 8), 2.0)
    targets[0, 0, 0] = torch.nan
    mask = torch.ones_like(targets, dtype=torch.bool)
    mask[1, 2:, :2] = False
    module, depth = make_module(target_depths=targets, target_valid_masks=mask,
                                target_max_resolution=4)
    baseline, _ = make_module()
    with patch.object(module.model, "refine", wraps=module.model.refine) as evaluate:
        edges, bank = module.get_depths_and_target_bank()
        assert evaluate.call_count == 1
    assert torch.equal(edges, depth)
    assert bank.shape == (2, 2, 4)
    assert not module.target_valid_masks[0, 0, 0]
    assert not module.target_valid_masks[1, 1, 0]
    assert torch.equal(bank, torch.where(module.target_valid_masks, 2.0, 0.0))
    assert torch.equal(module.normalized_points, baseline.normalized_points)
    # Grid (u,v)=(1,1) lifts to original XYZ=(5,5,2), not coarse-pixel XYZ.
    expected = (torch.tensor([5.0, 5.0, 2.0]) - module.centers[0]) / module.scales[0]
    torch.testing.assert_close(module.query_points[0, module.edge_count + 5], expected)


def test_target_bank_gradients_and_fixed_original_inputs():
    targets = torch.full((2, 3, 5), 2.0, requires_grad=True)
    module, _ = make_module(target_depths=targets)
    fixed_points = module.query_points.clone()
    _, bank = module.get_depths_and_target_bank(["second"])
    bank.sum().backward()
    assert module.model.biases[-1].grad[0].abs().sum() == 0
    assert module.model.biases[-1].grad[1].abs().sum() > 0
    assert targets.grad is None
    assert not module.query_points.requires_grad
    module.optimizer_and_scheduler_step()
    assert torch.equal(module.query_points, fixed_points)
    with torch.no_grad():
        module.model.biases[-1].fill_(0.2)
    _, first = module.get_depths_and_target_bank()
    _, second = module.get_depths_and_target_bank()
    assert torch.equal(first, second)  # no recursive accumulation of depth changes
    assert torch.isfinite(first).all() and (first > 0).all()
    assert (first >= 0.5 * targets.detach()).all() and (first <= 2 * targets.detach()).all()


def test_target_invalid_pixels_and_selection_cache():
    targets = torch.ones(2, 2, 3)
    targets[0, 0, 0] = torch.nan
    targets[1, 0, 1] = -1
    module, _ = make_module(target_depths=targets)
    # Nonzero final weights expose stale coordinate encodings hidden by identity init.
    with torch.no_grad():
        module.model.weights[-1].fill_(0.1)
    expected_edges, expected_bank = module.get_depths_and_target_bank()
    assert torch.isfinite(expected_bank).all()
    assert expected_bank[0, 0, 0] == expected_bank[1, 0, 1] == 0
    for ids in ([1], [0], [1, 0]):
        edges, bank = module.get_depths_and_target_bank(ids)
        torch.testing.assert_close(edges, expected_edges[ids])
        torch.testing.assert_close(bank, expected_bank[ids])
    edges, bank = module.get_depths_and_target_bank()
    torch.testing.assert_close(edges, expected_edges)
    torch.testing.assert_close(bank, expected_bank)
