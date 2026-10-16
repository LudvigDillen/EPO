"""Adapt sc3r's shared XYZ depth MLP to EPO's sampled edge depths."""

import torch
import torch.nn as nn

from depth_refinement.optimization.mlp import PerImageDepthMLP, normalize_camera_points
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
        if ids is None:
            indices = None
            base, points, valid = self.depth, self.normalized_points, self.valid_masks
        else:
            indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            base = self.depth[indices]
            points = self.normalized_points[indices]
            valid = self.valid_masks[indices]
        refined = self.model.refine(base, points, valid, indices)
        return torch.where(valid, refined, base)

    def get_all_parameters(self):
        return self.get_parameters(None)

    def __repr__(self):
        return f"PointMLPDepthModule(parameters={sum(p.numel() for p in self.parameters()):,})"
