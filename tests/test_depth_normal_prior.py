"""Normal regularization reuses the current depth bank and shared loss code."""

from unittest.mock import patch
from pathlib import Path
import os
import subprocess

import torch

from depth_refinement.optimization.normals import (
    depth_normals, normal_depth_continuity_mask, normal_preservation_loss,
)
from modules.depth_normal_prior import DepthNormalPrior
from modules.point_mlp_depth import PointMLPDepthModule


def test_runner_normal_options():
    root = Path(__file__).resolve().parents[3]
    command = 'conda() { printf "%s\\n" "$*"; }; export -f conda; bash scripts/run_epo_subsampled.sh'
    for method, weight in (("ours", "0.0"), ("ours", "0.01"), ("EPO", "0.01")):
        env = dict(os.environ, METHOD=method, DEPTH_NORMAL_WEIGHT=weight,
                   DEPTH_NORMAL_MAX_RELATIVE_JUMP="0.15", VIS="false")
        result = subprocess.run(["bash", "-c", command], cwd=root, env=env,
                                capture_output=True, text=True, check=True)
        if method == "ours":
            assert f"--depth-normal-weight {weight}" in result.stdout
            assert "--depth-normal-max-relative-jump 0.15" in result.stdout
        else:
            assert "--depth-normal" not in result.stdout


def inputs():
    depths = torch.full((2, 5, 5), 3.)
    masks = torch.ones_like(depths, dtype=torch.bool)
    intrinsics = torch.tensor([[[10., 0., 2.], [0., 10., 2.], [0., 0., 1.]]]).repeat(2, 1, 1)
    return depths, masks, intrinsics


def test_fixed_reference_masks_and_shared_loss_gradients():
    depths, masks, intrinsics = inputs()
    depths[0, 0, 0] = 0
    masks[0, 0, 0] = False
    depths[1, :, 3:] = 6  # fixed discontinuity mask must exclude this boundary
    depths.requires_grad_()
    intrinsics.requires_grad_()
    prior = DepthNormalPrior(depths, masks, intrinsics)
    normals, valid = depth_normals(depths.detach(), masks, prior.rays)
    torch.testing.assert_close(prior.reference_normals, normals)
    assert torch.equal(prior.reference_valid, valid & normal_depth_continuity_mask(depths, masks, 0.15))
    assert all(not buffer.requires_grad for buffer in prior.buffers())
    reference = prior.reference_normals.clone()
    refined = (depths.detach() * torch.linspace(1., 1.1, 5)).requires_grad_()
    with patch('modules.depth_normal_prior.normal_preservation_loss', wraps=normal_preservation_loss) as shared:
        loss = prior(refined)
        assert shared.call_count == 1 and loss > 0
    loss.backward()
    assert torch.isfinite(refined.grad).all() and refined.grad.abs().sum() > 0
    assert depths.grad is None and intrinsics.grad is None
    assert torch.equal(prior.reference_normals, reference)


def test_identity_and_empty_support():
    depths, masks, intrinsics = inputs()
    prior = DepthNormalPrior(depths, masks, intrinsics)
    assert prior(depths).abs() < 1e-6
    empty = DepthNormalPrior(depths, torch.zeros_like(masks), intrinsics)
    current = depths.clone().requires_grad_()
    assert empty(current) == 0
    empty(current).backward()
    assert torch.isfinite(current.grad).all()


def test_normal_loss_reuses_single_mlp_bank_evaluation():
    depths, masks, intrinsics = inputs()
    pixels = torch.tensor([[[1., 1.], [3., 3.]]] * 2)
    module = PointMLPDepthModule({'a': 0, 'b': 1}, depths[:, :1, :2].reshape(2, 2),
                                pixels, intrinsics, torch.ones(2, 2, dtype=torch.bool),
                                target_depths=depths, device='cpu', warmup_steps=1, max_num_iterations=4)
    prior = DepthNormalPrior(depths, masks, intrinsics)
    with torch.no_grad():
        module.model.weights[-1].fill_(0.01)
    with patch.object(module.model, 'refine', wraps=module.model.refine) as evaluate:
        _, current = module.get_depths_and_target_bank()
        prior(current).backward()
        assert evaluate.call_count == 1
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
