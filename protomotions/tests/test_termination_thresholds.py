import torch

from protomotions.envs.component_factories import (
    anchor_height_error_term_factory,
    anchor_pos_error_term_factory,
    tracking_error_term_factory,
)
from protomotions.envs.terminations.tracking import (
    compute_anchor_height_error_term,
    compute_anchor_pos_error_term,
    compute_tracking_error,
)


def test_tracking_error_threshold_is_forwarded_by_factory():
    component = tracking_error_term_factory(threshold=1.5)

    assert component.static_params == {"error_threshold": 1.5}
    current = torch.zeros(1, 2, 3)
    reference = current.clone()
    reference[:, 1, 0] = 1.0
    assert not compute_tracking_error(
        current, reference, **component.static_params
    ).item()


def test_tracking_error_threshold_changes_result():
    current = torch.zeros(1, 2, 3)
    reference = current.clone()
    reference[:, 1, 0] = 1.0

    assert compute_tracking_error(current, reference, error_threshold=0.5).item()
    assert not compute_tracking_error(
        current, reference, error_threshold=1.5
    ).item()


def test_beyond_mimic_termination_factories_use_compute_threshold():
    anchor_pos = anchor_pos_error_term_factory(threshold=1.5)
    anchor_height = anchor_height_error_term_factory(threshold=1.5)
    assert anchor_pos.static_params == {"error_threshold": 1.5}
    assert anchor_height.static_params == {"error_threshold": 1.5}

    current_anchor = torch.tensor([[0.0, 0.0, 1.0]])
    reference = torch.zeros(1, 1, 3)
    assert not compute_anchor_pos_error_term(
        current_anchor,
        reference,
        anchor_idx=0,
        **anchor_pos.static_params,
    ).item()
    assert not compute_anchor_height_error_term(
        current_anchor,
        reference,
        anchor_idx=0,
        **anchor_height.static_params,
    ).item()
