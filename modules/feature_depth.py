"""Correct VGGT-Omega edge depths in the span of the network's own depth-head features."""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from depth_refinement.optimization.features import PerImageFeatureDepth, whiten_features
from modules.base_module import BaseModule


def load_letterboxed_features(path, image_names, image_hws, device="cuda"):
    """Load ``features.pth`` and crop each square map to its image, ``[C, h, w]`` per image.

    VGGT-Omega sees every image letterboxed into a square, so its feature grid
    covers the padding too. The crop matches the one EPO applies to the square
    depth maps (``helpers.load._process_single_depth``), leaving a grid that spans
    exactly the image. All maps are then resized to the first one's grid, since
    ``PerImageFeatureDepth`` needs one shared grid.
    """
    features = torch.load(Path(path), map_location="cpu", weights_only=True)
    cropped = []
    for name, (height, width) in zip(image_names, image_hws, strict=True):
        stem = Path(name).with_suffix("").as_posix()
        if stem not in features:
            raise KeyError(f"No VGGT-Omega features for {name} in {path}")
        grid = features[stem].to(device=device, dtype=torch.float32)  # [C, S, S]
        size = grid.shape[-1]
        if grid.shape[-2] == size and height != width:
            if width > height:
                crop = round(size * height / width)
                top = (size - crop) // 2
                grid = grid[:, top:top + crop]
            else:
                crop = round(size * width / height)
                left = (size - crop) // 2
                grid = grid[:, :, left:left + crop]
        if cropped and grid.shape[-2:] != cropped[0].shape[-2:]:
            grid = F.interpolate(grid[None], cropped[0].shape[-2:], mode="bilinear",
                                 align_corners=False, antialias=True)[0]
        cropped.append(grid)
    return cropped


class FeatureDepthModule(BaseModule):
    """Per-image ``log d = log d_0 + beta^T z + log_scale`` evaluated at EPO's edge pixels."""

    def __init__(
        self,
        image_id_map: dict[str, int],
        depth: torch.Tensor,
        pixels: torch.Tensor,
        valid_masks: torch.Tensor,
        image_hws: torch.Tensor,
        features: list[torch.Tensor],
        rank: int = 256,
        eigenvalue_floor: float = 1e-6,
        lr: float = 3e-3,
        grad: bool = True,
        warmup_steps: int = 25,
        max_num_iterations: int = 1000,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__(image_id_map, device=device, dtype=dtype)
        depth = torch.as_tensor(depth, device=self.device, dtype=dtype).detach()
        pixels = torch.as_tensor(pixels, device=self.device, dtype=dtype).detach()
        valid = torch.as_tensor(valid_masks, device=self.device).bool().detach()
        if depth.ndim != 2 or pixels.shape != (*depth.shape, 2) or valid.shape != depth.shape:
            raise ValueError("Expected depth [N,P], edge pixels [N,P,2] and validity [N,P]")
        if len(features) != depth.shape[0]:
            raise ValueError("Need one feature map per image")
        valid = valid & torch.isfinite(depth) & (depth > 0)

        # Edge pixels use EPO's integer-centre convention; the feature grid spans
        # the whole image, so pixel x covers [x, x + 1) of [0, W).
        hw = torch.as_tensor(image_hws, device=self.device, dtype=dtype)  # [N,2] (h, w)
        grid = (pixels + 0.5) / hw.flip(-1)[:, None] * 2 - 1  # [N,P,2] in [-1, 1]
        self.register_buffer("depth", depth)
        self.register_buffer("valid_masks", valid)
        self.register_buffer("sample_grid", grid[:, :, None])  # [N,P,1,2]
        self.model = PerImageFeatureDepth(
            [whiten_features(f, rank, eigenvalue_floor) for f in features]
        ).to(self.device)
        self.model.requires_grad_(grad)
        if grad:
            # Plain Adam: weight decay would shrink beta and log_scale toward the
            # unrefined depth, which only hurt on ETH3D (--depth-identity-weight).
            self.optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, eps=1e-10, fused=self.device.type == "cuda"
            )
            self.init_scheduler(warmup_steps, max_num_iterations)

    def parameters(self, recurse=True):
        """Expose beta and log_scale to EPO's optimizer."""
        return [p for p in nn.Module.parameters(self, recurse=recurse) if p.requires_grad]

    def get_parameters(self, ids):
        """Return corrected edge depths in EPO's [N,P] layout."""
        base, grid, valid = self.depth, self.sample_grid, self.valid_masks
        corrections = self.model.log_corrections()  # [N,h,w]
        if ids is not None:
            indices = self.map_names_to_indices(ids) if isinstance(ids[0], str) else ids
            indices = torch.as_tensor(indices, device=self.device, dtype=torch.long)
            base, grid, valid = base[indices], grid[indices], valid[indices]
            corrections = corrections[indices]
        log_correction = F.grid_sample(
            corrections[:, None], grid, mode="bilinear", padding_mode="border",
            align_corners=False,
        )[:, 0, :, 0]  # [B,P]
        return torch.where(valid, base * torch.exp(log_correction), base)

    def get_all_parameters(self):
        return self.get_parameters(None)

    def __repr__(self):
        ranks = self.model.retained_ranks
        return (f"FeatureDepthModule(images={len(ranks)}, retained directions "
                f"min/max {min(ranks)}/{max(ranks)})")
