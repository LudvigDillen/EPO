"""Sample current target depths at projected source edges (pinhole cameras)."""

import torch
from depth_refinement.optimization.residuals import sampled_camera_depth_residuals


def sampled_depth_residuals(
    points_world: torch.Tensor,  # [B,L,3], refined source edge points
    source_valid: torch.Tensor,  # [B,L], excludes padded/invalid source edges
    target_depths: torch.Tensor,  # [B,H,W], current refined target bank
    target_valid: torch.Tensor,  # [B,H,W], fixed original validity
    target_intrinsics: torch.Tensor,  # [B,3,3], CURRENT K scaled to bank resolution
    target_world_to_camera: torch.Tensor,  # [B,4,4], CURRENT target poses
    *,
    residual_threshold: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return signed relative residuals and validity, both [B,L].

    Each batch row is one directed pair. Bilinear sampling uses pixel centers
    and requires all contributing target pixels to be valid. Outputs keep a
    fixed shape for compilation/CUDA capture; invalid residuals are zero.
    Gradients flow through source geometry, target depths, K and target poses.
    Depth-discontinuity filtering is deliberately not included yet.
    """
    finite_source = torch.isfinite(points_world).all(-1)  # [B,L]
    points = torch.where(finite_source[..., None], points_world, 0)  # [B,L,3]
    camera = points @ target_world_to_camera[:, :3, :3].transpose(1, 2)
    camera = camera + target_world_to_camera[:, None, :3, 3]  # [B,L,3]
    return sampled_camera_depth_residuals(
        camera, source_valid & finite_source, target_depths, target_valid,
        target_intrinsics, residual_threshold=residual_threshold,
    )


def depth_residual_loss(residuals: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """L1 mean per direction [B]; empty directions return differentiable zero."""
    return torch.where(valid, residuals.abs(), 0).sum(-1) / valid.sum(-1).clamp_min(1)
