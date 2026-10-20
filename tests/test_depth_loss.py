"""CPU checks for sampled relative-depth supervision, independent of EPO wiring."""

import torch

from losses.depth_loss import depth_residual_loss, sampled_depth_residuals


def inputs():
    return dict(
        points_world=torch.tensor([[[1.0, 1.0, 2.0]]]),  # projects to (0.5,0.5)
        source_valid=torch.ones(1, 1, dtype=torch.bool),
        target_depths=torch.tensor([[[2.0, 2.2], [2.4, 2.6]]]),
        target_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        target_intrinsics=torch.eye(3)[None],
        target_world_to_camera=torch.eye(4)[None],
        residual_threshold=0.5,
    )


def test_bilinear_relative_residual_and_identity():
    args = inputs()
    residual, valid = sampled_depth_residuals(**args)
    assert valid.all()
    torch.testing.assert_close(residual, torch.tensor([[0.3 / 2.15]]))
    torch.testing.assert_close(depth_residual_loss(residual, valid), residual[0])
    args["target_depths"] = torch.full((1, 2, 2), 2.0)
    residual, valid = sampled_depth_residuals(**args)
    assert valid.all() and torch.equal(residual, torch.zeros_like(residual))


def test_gradients_through_both_depths_poses_and_intrinsics():
    args = inputs()
    source_depth = torch.tensor(2.0, requires_grad=True)
    args["points_world"] = torch.tensor([[[0.5, 0.5, 1.0]]]) * source_depth
    for key in ("target_depths", "target_intrinsics", "target_world_to_camera"):
        args[key].requires_grad_()
    residual, valid = sampled_depth_residuals(**args)
    depth_residual_loss(residual, valid).sum().backward()
    assert source_depth.grad.isfinite() and source_depth.grad.abs() > 0
    for key in ("target_depths", "target_intrinsics", "target_world_to_camera"):
        grad = args[key].grad
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0, key
    assert (args["target_depths"].grad != 0).all()  # all four interpolation neighbors


def test_invalid_and_empty_support_are_finite():
    args = inputs()
    args["points_world"] = torch.tensor([[
        [1., 1., 2.], [1., 1., -2.], [10., 1., 2.],
        [float("nan"), 1., 2.], [1., 1., 2.],
    ]], requires_grad=True)
    args["source_valid"] = torch.tensor([[True, True, True, True, False]])
    residual, valid = sampled_depth_residuals(**args)
    assert valid.tolist() == [[True, False, False, False, False]]
    for change in ("mask", "nan_depth", "threshold"):
        modified = dict(args)
        if change == "mask":
            modified["target_valid"] = torch.tensor([[[False, True], [True, True]]])
        elif change == "nan_depth":
            modified["target_depths"] = torch.tensor([[[float("nan"), 2.2], [2.4, 2.6]]])
        else:
            modified["residual_threshold"] = 0.01
        residual, valid = sampled_depth_residuals(**modified)
        assert not valid.any() and torch.equal(residual, torch.zeros_like(residual))
        loss = depth_residual_loss(residual, valid).sum()
        assert loss == 0
        loss.backward()
        assert torch.isfinite(args["points_world"].grad).all()
