"""Write EPO's refined depth as dense maps, for evaluating it like a dense method.

EPO optimizes depth only at the sampled edge pixels. The point-MLP and
vggt-features parameterizations define a correction at every pixel, so their
dense map is exact. The affine parameterization has one (scale, shift) pair
per edge pixel; it is spread to every pixel in two ways, both leaving pixels
with no edge within ``radius`` pixels uncorrected:

* ``affine_nn``: the pair of the nearest edge pixel.
* ``affine_interp``: normalized convolution, a Gaussian-weighted mean of the
  nearby pairs, ``(G * (w a)) / (G * w)`` with ``w`` = 1 at edge pixels.

Maps live on EPO's working grid (the image resized to ``images_size``), so
every arm of one scene shares the same pixels.
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage


def _edge_samples(epo, index):
    """Integer edge pixels ``[P,2]`` (x, y) and their validity, for one image."""
    edges = epo.edges_padded.get_parameters([index]).detach().squeeze(0)
    valid = epo.pad_masks.get_parameters([index]).detach().squeeze(0).bool()
    return edges.round().long(), valid


def _spread_affine(ab, edges, valid, hw, radius, sigma):
    """Nearest-neighbour and normalized-convolution fields of per-edge (a, b). (scale and shift)"""
    height, width = hw
    edges, ab = edges[valid].cpu().numpy(), ab[valid].detach().float().cpu().numpy()
    inside = (edges[:, 0] >= 0) & (edges[:, 0] < width) & (edges[:, 1] >= 0) & (edges[:, 1] < height)
    edges, ab = edges[inside], ab[inside]
    weight = np.zeros((height, width), np.float64)
    values = np.zeros((2, height, width), np.float64)
    np.add.at(weight, (edges[:, 1], edges[:, 0]), 1.0)
    for channel in range(2):
        np.add.at(values[channel], (edges[:, 1], edges[:, 0]), ab[:, channel])

    # Nearest edge pixel; a pixel holding several edges takes their mean.
    distance, (rows, columns) = ndimage.distance_transform_edt(weight == 0, return_indices=True)
    mean = values / np.maximum(weight, 1)
    nearest = np.where(distance <= radius, mean[:, rows, columns], 0.0)

    # Normalized convolution; below the weight of one edge at ``radius`` there is
    # no edge close enough, so the pixel keeps the original depth.
    blurred_weight = ndimage.gaussian_filter(weight, sigma, mode="constant")
    blurred = np.stack([ndimage.gaussian_filter(v, sigma, mode="constant") for v in values])
    floor = np.exp(-0.5 * (radius / sigma) ** 2) / (2 * np.pi * sigma**2)
    interpolated = np.where(blurred_weight >= floor, blurred / np.maximum(blurred_weight, 1e-300), 0.0)
    return nearest, interpolated


def dense_depths(epo, *, radius=50.0, sigma=15.0):
    """Return ``{image_stem: {...}}`` with the base and refined maps of every image.

    Each entry holds ``base`` (the unrefined depth) and either ``dense``
    (point-mlp, vggt-features) or ``affine_nn`` and ``affine_interp`` (affine),
    all ``[h, w]`` float32 with 0 where invalid, plus the refined ``edge_pixels``
    ``[P,2]`` (x, y) and ``edge_depth`` ``[P]``.
    """
    module = epo.sampled_depth
    parameterization = epo.depth_parameterization
    feature_corrections = (
        module.model.log_corrections().detach() if parameterization == "vggt-features" else None
    )
    out = {}
    for name, index in epo.image_id_map.items():
        height, width = epo.images[name]["hw"]
        base = epo.images[name]["depth"].detach().to(epo.device, torch.float32)
        base = base.reshape(base.shape[-2:])[:height, :width]  # drop the common-shape NaN pad
        valid = torch.isfinite(base) & (base > 0)
        base = torch.where(valid, base, 0)
        edges, edge_valid = _edge_samples(epo, index)
        edge_depth = module.get_parameters([index]).detach().reshape(-1)
        entry = {
            "base": base.cpu(),
            "edge_pixels": edges[edge_valid].cpu(),
            "edge_depth": edge_depth[edge_valid].float().cpu(),
        }

        if parameterization == "affine":
            if module.direct_backprop or not module.per_pixel:
                raise ValueError("Dense export supports only EPO's per-pixel (scale, shift) model")
            nearest, interpolated = _spread_affine(
                module.params[index], edges, edge_valid, (height, width), radius, sigma
            )
            for key, field in (("affine_nn", nearest), ("affine_interp", interpolated)):
                a, b = (torch.as_tensor(f, device=epo.device, dtype=torch.float32) for f in field)
                refined = torch.clamp(base * (1 + a) + b, min=module.min_depth)
                entry[key] = torch.where(valid, refined, 0).cpu()
        elif parameterization == "vggt-features":
            from depth_refinement.optimization.features import resample_correction

            correction = resample_correction(feature_corrections[index : index + 1], (height, width))[0]
            entry["dense"] = torch.where(valid, base * torch.exp(correction), 0).cpu()
        elif parameterization == "point-mlp":
            from depth_refinement.optimization.normals import camera_rays

            # The MLP sees each pixel's fixed original camera-space point, normalized
            # with the image's edge statistics, exactly as during optimization.
            rays = camera_rays(module.source_intrinsics[index : index + 1], height, width)
            points = (rays[0] * base[..., None] - module.centers[index]) / module.scales[index]
            points = torch.where(valid[..., None], points, 0).reshape(1, -1, 3)
            cached = module.model.pos_enc_pts
            module.model.pos_enc_pts = None
            refined = module.model.refine(
                base.reshape(1, -1), points, valid.reshape(1, -1),
                torch.tensor([index], device=epo.device),
            ).reshape(height, width)
            module.model.pos_enc_pts = cached
            entry["dense"] = torch.where(valid, refined.float(), 0).cpu()
        else:
            raise ValueError(f"No dense export for {parameterization!r}")
        out[Path(name).with_suffix("").as_posix()] = entry
    return out


@torch.no_grad()
def save_dense_depths(epo, path, *, radius=50.0, sigma=15.0):
    """Write :func:`dense_depths` and the export settings to ``path``."""
    torch.save({
        "parameterization": epo.depth_parameterization,
        "radius_px": float(radius),
        "sigma_px": float(sigma),
        "images": dense_depths(epo, radius=radius, sigma=sigma),
    }, path)
