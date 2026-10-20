"""Adapt sc3r's shared XYZ MLP to edge depths and an optional target depth bank."""

import torch
import torch.nn as nn

from depth_refinement.optimization.mlp import PerImageDepthMLP, normalize_camera_points
from depth_refinement.optimization.normals import camera_rays
from depth_refinement.optimization.residuals import downsample_depth_bank
from modules.base_module import BaseModule


class PointMLPDepthModule(BaseModule):
    """Correct each image's edge depths using fixed original camera-space XYZ."""

    def __init__(
        self,
        image_id_map: dict[str, int],
        depth: torch.Tensor,
        pixels: torch.Tensor,
        intrinsics: torch.Tensor,
        valid_masks: torch.Tensor,
        lr: float = 5e-3,
        grad: bool = True,
        warmup_steps: int = 25,
        max_num_iterations: int = 1000,
        random_seed: int = 0,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        target_depths: torch.Tensor | None = None,
        target_valid_masks: torch.Tensor | None = None,
        target_max_resolution: int = 128,
    ):
        super().__init__(image_id_map, device=device, dtype=dtype)
        depth = torch.as_tensor(depth, device=self.device, dtype=dtype).detach()
        pixels = torch.as_tensor(pixels, device=self.device, dtype=dtype).detach()
        intrinsics = torch.as_tensor(intrinsics, device=self.device, dtype=dtype).detach()
        valid = torch.as_tensor(valid_masks, device=self.device).bool().detach()
        if depth.ndim != 2 or pixels.shape != (*depth.shape, 2):
            raise ValueError("Expected depth [N,P] and edge pixels [N,P,2]")
        if intrinsics.shape != (depth.shape[0], 3, 3) or valid.shape != depth.shape:
            raise ValueError("Intrinsics must be [N,3,3] and validity [N,P]")
        if len(image_id_map) != depth.shape[0] or set(image_id_map.values()) != set(
            range(depth.shape[0])
        ):
            raise ValueError("image_id_map must cover depth rows in order")

        homogeneous = torch.cat((pixels, torch.ones_like(depth[..., None])), dim=-1)
        rays = torch.linalg.solve(intrinsics, homogeneous.transpose(1, 2)).transpose(1, 2)
        points = rays * depth[..., None]  # [N,P,3], fixed original camera-space XYZ
        valid = valid & torch.isfinite(points).all(-1) & torch.isfinite(depth) & (depth > 0)
        normalized, centers, scales = normalize_camera_points(points, valid)
        self.register_buffer("depth", depth)
        self.register_buffer("normalized_points", normalized)
        self.register_buffer("valid_masks", valid)
        self.register_buffer("centers", centers)
        self.register_buffer("scales", scales)
        self.edge_count = depth.shape[1]
        self.target_size = None
        if target_depths is not None:
            targets = torch.as_tensor(target_depths, device=self.device, dtype=dtype).detach()
            if targets.ndim != 3 or targets.shape[0] != depth.shape[0]:
                raise ValueError("Target depths must be [N,H,W] with one map per image")
            if target_max_resolution < 1:
                raise ValueError("target_max_resolution must be positive")
            height, width = targets.shape[-2:]
            factor = min(1.0, target_max_resolution / max(height, width))
            self.target_size = (max(1, round(height * factor)), max(1, round(width * factor)))
            bank = downsample_depth_bank(
                targets, intrinsics, valid_masks=target_valid_masks, output_size=self.target_size
            )
            grid_depth = torch.where(bank.valid_masks, bank.depths, 0)  # [N,Ht,Wt]
            grid_points = camera_rays(bank.intrinsics, *self.target_size) * grid_depth[..., None]
            grid_points = grid_points.flatten(1, 2)  # [N,G,3], fixed original camera XYZ
            grid_valid = bank.valid_masks.flatten(1)  # [N,G]
            grid_points = (grid_points - centers[:, None]) / scales[:, None, None]
            grid_points = torch.where(grid_valid[..., None], grid_points, 0)
            self.register_buffer("target_intrinsics", bank.intrinsics.detach())
            self.register_buffer("target_valid_masks", bank.valid_masks)
            # One fixed query bank also keeps the shared positional-encoding cache valid.
            self.register_buffer("query_depth", torch.cat((depth, grid_depth.flatten(1)), dim=1))
            self.register_buffer("query_points", torch.cat((normalized, grid_points), dim=1))
            self.register_buffer("query_valid", torch.cat((valid, grid_valid), dim=1))
        self.model = PerImageDepthMLP(
            depth.shape[0], device=self.device, dtype=dtype, random_seed=random_seed
        )
        self.model.requires_grad_(grad)
        if grad:
            self.optimizer = torch.optim.AdamW(
                self.parameters(), lr=lr, weight_decay=1e-2, eps=1e-10,
                fused=self.device.type == "cuda",
            )
            self.init_scheduler(warmup_steps, max_num_iterations)

    def parameters(self, recurse=True):
        """Expose the shared MLP's weights to EPO's optimizer."""
        return [p for p in nn.Module.parameters(self, recurse=recurse) if p.requires_grad]

    def get_parameters(self, ids):
        """Return corrected edge depths in EPO's [N,P] layout."""
        return self.get_depths_and_target_bank(ids)[0]

    def get_depths_and_target_bank(self, ids=None):
        """Evaluate once; return edge depths [N,P] and optional target maps [N,Ht,Wt].

        Call this instead of separate getters when both outputs are needed.
        Every evaluation corrects the original depths with the current weights.
        """
        if self.target_size is None:
            base, points, valid = self.depth, self.normalized_points, self.valid_masks
        else:
            base, points, valid = self.query_depth, self.query_points, self.query_valid
        cached_encoding = self.model.pos_enc_pts
        if ids is None:
            indices = None
        else:
            indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            base, points, valid = base[indices], points[indices], valid[indices]
            # Equal-sized selections may refer to different images/coordinates.
            self.model.pos_enc_pts = None
        refined = self.model.refine(base, points, valid, indices)
        if indices is not None:
            self.model.pos_enc_pts = cached_encoding  # preserve the full-bank encoding
        refined = torch.where(valid, refined, base)
        if self.target_size is None:
            return refined, None
        targets = refined[:, self.edge_count:].reshape(refined.shape[0], *self.target_size)
        return refined[:, :self.edge_count], targets

    def get_all_parameters(self):
        return self.get_parameters(None)

    def __repr__(self):
        return f"PointMLPDepthModule(parameters={sum(p.numel() for p in self.parameters()):,})"
