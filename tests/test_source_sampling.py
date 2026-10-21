"""Uniform samples reuse EPO's padding, depth lookup and shared MLP inputs."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from epo import EPO
from modules.point_mlp_depth import PointMLPDepthModule
from depth_refinement.optimization.sampling import pixel_grid
from helpers.reprojection import create_grid


def sample_sources():
    state = SimpleNamespace(
        images={}, max_edges=1, uniform_samples_per_image=24, use_depth_confidence=False, device="cpu",
        dtype=torch.float32, rng=torch.Generator().manual_seed(42),
    )
    for name in ("a", "b"):
        depth = torch.full((10, 14), torch.nan)  # padded canvas
        depth[:8, :12] = 2
        depth[0, 0], depth[2, 2] = torch.nan, 0
        state.images[name] = dict(hw=(8, 12), depth=depth)
    for method in ("_pad_edges", "_sample_source_depths", "_sample_uniform_sources"):
        setattr(state, method, MethodType(getattr(EPO, method), state))
    state._sample_uniform_sources()
    return state


def test_uniform_coverage_depth_validity_and_repeatability():
    state, repeated = sample_sources(), sample_sources()
    for name, data in state.images.items():
        valid = data["pad_mask"].bool()
        points = data["edges_padded"][valid]
        assert len(points) == 22 and len(points) <= 24
        assert points[:, 0].min() == 0 and points[:, 0].max() == 10
        assert points[:, 1].min() == 0 and points[:, 1].max() == 6
        assert torch.equal(data["sampled_depth"][valid], torch.full((22,), 2.))
        assert torch.equal(points, repeated.images[name]["edges_padded"][valid])
        assert "edges_map" not in data  # no edge detector is needed


def test_mlp_normalization_uses_uniform_source_coordinates():
    state = sample_sources()
    pixels = torch.stack([d["edges_padded"] for d in state.images.values()])
    depths = torch.stack([d["sampled_depth"] for d in state.images.values()])
    valid = torch.stack([d["pad_mask"].bool() for d in state.images.values()])
    module = PointMLPDepthModule(
        {"a": 0, "b": 1}, depths, pixels, torch.eye(3)[None].repeat(2, 1, 1), valid,
        grad=False, device="cpu",
    )
    xyz = torch.cat((pixels, torch.ones_like(depths[..., None])), -1) * depths[..., None]
    torch.testing.assert_close(module.centers, xyz.median(dim=1).values)
    torch.testing.assert_close(module.normalized_points,
                               (xyz - module.centers[:, None]) / module.scales[:, None, None])
    torch.testing.assert_close(module.get_parameters(None), depths)


def test_uniform_uses_existing_sample_cap():
    state = sample_sources()
    state.uniform_samples_per_image = 8
    state._sample_uniform_sources()
    assert state.max_edges == 8
    assert all(int(d["pad_mask"].sum()) == 8 for d in state.images.values())


def test_uniform_rejects_dtf_objective():
    with patch("torch.cuda.is_available", return_value=True):
        try:
            EPO(source_sampling="uniform", residual_type="dtf")
        except ValueError as exc:
            assert "Uniform source sampling requires" in str(exc)
        else:
            raise AssertionError("DTF must keep edge source sampling")


def test_depth_only_skips_edge_error_reporting():
    state = SimpleNamespace(dt_fields=None, poses=Mock(), intrinsics=Mock(), unproject_edges_to_3D=Mock())
    assert EPO.compute_mre(state) is None
    assert not hasattr(state, "mre")
    state.poses.update_all_matrices.assert_called_once()
    state.intrinsics.update_all_matrices.assert_called_once()
    state.unproject_edges_to_3D.assert_called_once()


def test_shared_grid_matches_epo_wrapper():
    expected = torch.tensor([[1., 1.], [3., 1.], [5., 1.], [1., 3.], [3., 3.], [5., 3.]])
    grid = pixel_grid(6, 8, device="cpu", stride=2, border=1)
    torch.testing.assert_close(grid.reshape(-1, 2), expected)
    torch.testing.assert_close(create_grid(torch.zeros(1, 6, 8), sampling_factor=2, border=1), expected)
