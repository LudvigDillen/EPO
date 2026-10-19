"""Camera sharing and averaged intrinsics agree for disk and feed-forward input."""

import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import patch

import pycolmap
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import epo


def check_camera_groups_and_averaged_focals(source, shared):
    state = SimpleNamespace(
        device="cpu", dtype=torch.float32, single_camera_per_folder=shared,
        load_with_pad=False, images_size=8, use_depth_confidence=False,
        k_lr=1e-3, grad_k=True, max_num_iterations=100, warmup_steps=1,
        camera_model="SIMPLE_RADIAL", R_lr=1e-3, t_lr=1e-3,
        grad_R=False, grad_t=False, grad_t_offset=False, mlp_pose_lr=1e-3,
        mlp_hidden_dim=8, use_mlp_pose_refinement=False, use_amp=False,
    )
    for method in ("_camera_params", "_ff_camera_params", "_pad_depths_to_common_shape"):
        setattr(state, method, MethodType(getattr(epo.EPO, method), state))
    names = ["a.jpg", "b.jpg", "left/a.jpg", "left/b.jpg", "right/a.jpg"]
    entries, images = {}, {}
    for index, name in enumerate(names):
        focal = 10.0 + 2 * index
        pose = pycolmap.Rigid3d()
        images[index] = SimpleNamespace(
            name=name, camera_id=index,
            camera=pycolmap.Camera(model="PINHOLE", width=8, height=8,
                                   params=[focal, focal, 4.0, 4.0]),
            cam_from_world=lambda pose=pose: pose,
        )
        entries[name] = {
            "image": torch.zeros(3, 8, 8), "depth": torch.ones(8, 8),
            "pose": torch.eye(4),
            "intrinsic": torch.tensor([[focal, 0, 4], [0, focal, 4], [0, 0, 1]]),
        }
    if source == "ff":
        epo.EPO._populate_from_ff(state, entries)
    else:
        state.recon = SimpleNamespace(images=images)
        state.images = {name: {"hw": (8, 8)} for name in names}
        epo.EPO._read_cameras_from_reconstruction(state)

    groups = [state.images[name]["cam_id"] for name in names]
    if shared:
        assert groups == [".", ".", "left", "left", "right"]
        torch.testing.assert_close(state.intrinsics.k_params[:, 0],
                                   torch.tensor([11.0, 15.0, 18.0]))
    else:
        assert len(set(groups)) == len(names)
        torch.testing.assert_close(state.intrinsics.k_params[:, 0].sort().values,
                                   torch.tensor([10.0, 12.0, 14.0, 16.0, 18.0]))
    assert state.intrinsics.params.shape[0] == len(set(groups))


class TestCameraGrouping(unittest.TestCase):
    def test_disk_and_feed_forward_grouping(self):
        # Camera initialization is under test; pose optimization is unrelated.
        with patch.object(epo, "PoseModule", lambda *args, **kwargs: None):
            for source in ("disk", "ff"):
                for shared in (True, False):
                    with self.subTest(source=source, shared=shared):
                        check_camera_groups_and_averaged_focals(source, shared)


if __name__ == "__main__":
    unittest.main()
