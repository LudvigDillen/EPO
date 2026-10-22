"""Fixed depth-normal reference using sc3r's shared normal-preservation loss."""

from torch import nn

from depth_refinement.optimization.normals import (
    camera_rays, depth_normals, normal_depth_continuity_mask, normal_preservation_loss,
)


class DepthNormalPrior(nn.Module):
    def __init__(self, initial_depths, valid_masks, intrinsics, max_relative_jump=0.15):
        super().__init__()
        if min(initial_depths.shape[-2:]) < 3:
            raise ValueError("Normal supervision needs a depth bank of at least 3x3 pixels")
        initial_depths = initial_depths.detach()  # [N,H,W], original bank, never refined
        masks = valid_masks.detach().bool()  # [N,H,W]
        rays = camera_rays(intrinsics.detach(), *initial_depths.shape[-2:])  # [N,H,W,3]
        reference, valid = depth_normals(initial_depths, masks, rays)  # [N,H-2,W-2,3], [N,H-2,W-2]
        valid = valid & normal_depth_continuity_mask(initial_depths, masks, max_relative_jump)
        self.register_buffer("valid_masks", masks)
        self.register_buffer("rays", rays.detach())
        self.register_buffer("reference_normals", reference.detach())
        self.register_buffer("reference_valid", valid.detach())

    def forward(self, current_depths):
        """Regularize the already-evaluated bank; no extra MLP or pose/K evaluation."""
        return normal_preservation_loss(
            current_depths, self.valid_masks, self.rays,
            self.reference_normals, self.reference_valid,
        )
