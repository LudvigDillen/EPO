"""CPU checks for the sparse XYZ depth representation (no EPO run needed)."""

import torch

from modules.point_mlp_depth import PointMLPDepthModule


def make_module():
    depth = torch.tensor([[1.0, 2.0, 1000.0], [3.0, 4.0, 1000.0]])
    pixels = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [50.0, 50.0]]] * 2)
    intrinsics = torch.eye(3).expand(2, -1, -1).clone()
    valid = torch.tensor([[True, True, False]] * 2)
    module = PointMLPDepthModule(
        {"first": 0, "second": 1}, depth, pixels, intrinsics, valid,
        warmup_steps=1, max_num_iterations=4, device="cpu",
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
