"""Small integration checks for EPO's selectable depth objective."""

from types import MethodType, SimpleNamespace
from unittest.mock import patch

import torch

from epo import EPO
from helpers.cuda_graph_ops import GeometryHead, capture_forward_backward
from modules.point_mlp_depth import PointMLPDepthModule


def make_state(device="cpu"):
    depths = torch.tensor([[2., 2.], [2.04, 2.04]], device=device)
    pixels = torch.tensor([[[0.5, 0.5], [1.5, 1.5]]] * 2, device=device)
    k = torch.eye(3, device=device)[None].repeat(2, 1, 1).requires_grad_()
    poses = torch.eye(4, device=device)[None].repeat(2, 1, 1).requires_grad_()
    bank = torch.tensor([[[2., 2.02, 2.04]] * 3] * 2, device=device)
    module = PointMLPDepthModule(
        {"a": 0, "b": 1}, depths, pixels, k, torch.ones_like(depths, dtype=torch.bool),
        target_depths=bank, device=device, warmup_steps=1, max_num_iterations=4,
    )
    state = SimpleNamespace(
        residual_type="depth", depth_residual_threshold=0.1, sampled_depth=module,
        poses=SimpleNamespace(poses=None, get_projection_matrix=lambda ids: poses if ids is None else poses[ids]),
        intrinsics=SimpleNamespace(cameras=None, get_intrinsic_matrix=lambda ids: k if ids is None else k[ids],
                                   get_k1=lambda ids: None),
        edges_padded=SimpleNamespace(get_parameters=lambda ids: pixels),
        images_cams_ids=torch.tensor([[0, 0], [1, 1]], device=device),
        _image_ids_are_identity=True, backend="torch", dtype=torch.float32,
        pad_masks=SimpleNamespace(params=torch.ones_like(depths, dtype=torch.bool)),
        images_hw=SimpleNamespace(get_parameters=lambda ids: torch.tensor([[3, 3]], device=device).expand(len(ids), -1)),
        dt_fields=SimpleNamespace(params=torch.zeros(2, 1, 3, 3, device=device)),
        depth_bank_transform=torch.eye(3, device=device)[None].expand(2, -1, -1),
        max_viewgraph_pairs=10, log_granular_time=False, _sync_for_timing=lambda: None,
    )
    for name in ("compute_edges_3D", "create_batched_inputs"):
        setattr(state, name, MethodType(getattr(EPO, name), state))
    return state, k, poses


def test_depth_objective_wiring_and_dtf_reporting():
    state, k, poses = make_state()
    output = GeometryHead(state)()
    assert len(output) == 4
    state.edges_3D = SimpleNamespace(params=output[2])
    pairs = torch.tensor([[0, 1, 0, 1]])
    loss, _ = EPO.compute_forward_step(state, pairs)
    assert loss.shape == (2,) and loss.sum() > 0
    loss.sum().backward()
    for grad in (k.grad, poses.grad, state.sampled_depth.model.biases[-1].grad):
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
    assert (state.sampled_depth.model.biases[-1].grad.abs().sum(-1) > 0).all()
    with patch("epo.project_and_sample_logic", return_value=(torch.ones(2, 2), torch.ones(2, 2, dtype=torch.bool))) as dtf:
        reported, _ = EPO.compute_forward_step(state, pairs, robust=False)
        assert dtf.call_count == 1 and torch.equal(reported, torch.ones(2))
    state.residual_type = "dtf"
    assert len(GeometryHead(state)()) == 3


def test_cuda_geometry_bank_replays_and_backpropagates():
    if not torch.cuda.is_available():
        return
    state, k, poses = make_state("cuda")
    head = GeometryHead(state)
    head.parameters = lambda recurse=True: [*state.sampled_depth.parameters(), k, poses]
    graph = capture_forward_backward(head)
    for _ in range(2):
        state.sampled_depth.optimizer.zero_grad()
        output = graph()
        expected = GeometryHead(state)()
        torch.testing.assert_close(output[2], expected[2])
        torch.testing.assert_close(output[3], expected[3])
        (output[2].sum() + output[3].sum()).backward()
        grad = state.sampled_depth.model.biases[-1].grad
        assert torch.isfinite(grad).all() and (grad.abs().sum(-1) > 0).all()
        state.sampled_depth.optimizer.step()
