"""GT camera overlays are selected and display-aligned without modifying poses."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pycolmap

from epo_modules.reconstruct_and_viz import ReconstructAndVizModule


def test_gt_camera_overlay_alignment_selection_and_color():
    centers = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [8., 8., 8.]])
    translation = np.array([3., 4., 5.])

    def reconstruction(points):
        images = {}
        for index, center in enumerate(points):
            pose = pycolmap.Rigid3d(pycolmap.Rotation3d(), -center)
            images[index] = SimpleNamespace(name=f'{index}.jpg', cam_from_world=lambda pose=pose: pose)
        return SimpleNamespace(images=images)

    gt = reconstruction(centers)
    reference = reconstruction(2 * centers[:4] + translation)
    state = SimpleNamespace(log_reconstruction_rerun=Mock())
    with patch('helpers.load.read_reconstruction_without_points', side_effect=(gt, reference)):
        ReconstructAndVizModule.log_ground_truth_cameras_rerun(state, 'gt', 'initial')
    arguments = state.log_reconstruction_rerun.call_args.kwargs
    assert arguments['camera_color'] == [0, 255, 0]
    assert arguments['static_cameras'] and not arguments['points3D']
    assert arguments['image_names'] == {f'{index}.jpg' for index in range(4)}
    transform = arguments['world_transform']
    np.testing.assert_allclose(transform.scale, 2)
    np.testing.assert_allclose(transform.translation, translation)
    np.testing.assert_allclose(transform.rotation.matrix(), np.eye(3), atol=1e-10)
    np.testing.assert_allclose(gt.images[1].cam_from_world().inverse().translation, centers[1])
