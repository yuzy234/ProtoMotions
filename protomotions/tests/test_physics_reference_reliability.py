import torch

from protomotions.agents.mimic.layerwise_adapter_tracker_actor import (
    demand_conditioned_exploration_std,
    gate_demand_action_residual,
    uncertainty_plasticity_gate,
)
from protomotions.envs.control.mimic_control import (
    calibrate_reference_reliability,
)
from protomotions.envs.obs.target_poses import (
    build_contact_conditioned_takeoff_demand,
    build_max_coords_target_poses,
    build_max_coords_target_poses_future_rel,
    build_reliability_gated_max_coords_target_poses,
    build_reliability_gated_max_coords_target_poses_future_rel,
)


def test_takeoff_demand_requires_support_reliability_and_upward_acceleration():
    current_velocity = torch.zeros(3, 9, 3)
    current_velocity[:, 0, 2] = -0.5
    future_velocity = torch.zeros(3, 3, 3)
    future_velocity[:, :, 2] = torch.tensor([0.0, 1.5, 3.0])
    contacts = torch.zeros(3, 9, dtype=torch.bool)
    contacts[0, 4] = True
    contacts[2, 8] = True
    reliability = torch.ones(3, 3)
    reliability[2] = 0.0

    demand = build_contact_conditioned_takeoff_demand(
        current_ref_body_vel=current_velocity,
        future_ref_root_vel=future_velocity,
        current_ref_contacts=contacts,
        future_reference_reliability=reliability,
        vertical_speed_scale=3.0,
    )

    torch.testing.assert_close(demand, torch.tensor([[1.0], [0.0], [0.0]]))


def test_takeoff_demand_ignores_distant_weighted_out_horizon():
    current_velocity = torch.zeros(1, 9, 3)
    future_velocity = torch.zeros(1, 3, 3)
    future_velocity[0, -1, 2] = 4.0
    contacts = torch.zeros(1, 9, dtype=torch.bool)
    contacts[0, 3] = True

    demand = build_contact_conditioned_takeoff_demand(
        current_ref_body_vel=current_velocity,
        future_ref_root_vel=future_velocity,
        current_ref_contacts=contacts,
        future_reference_reliability=torch.ones(1, 3),
        horizon_weights=[1.0, 1.0, 0.0],
    )

    torch.testing.assert_close(demand, torch.zeros(1, 1))


def test_takeoff_demand_scales_only_selected_exploration_dimensions():
    std = demand_conditioned_exploration_std(
        base_std=torch.tensor([0.1, 0.2, 0.3, 0.4]),
        demand=torch.tensor([[0.0], [0.5], [1.0]]),
        max_multiplier=3.0,
        action_indices=[1, 3],
    )
    expected = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.4, 0.3, 0.8],
            [0.1, 0.6, 0.3, 1.2],
        ]
    )
    torch.testing.assert_close(std, expected)


def test_takeoff_demand_relaxes_regularization_without_disabling_it():
    from protomotions.envs.rewards.regularization import (
        relax_regularization_with_demand,
    )

    penalty = torch.tensor([2.0, 2.0, 2.0])
    relaxed = relax_regularization_with_demand(
        penalty,
        demand=torch.tensor([[0.0], [0.5], [1.0]]),
        minimum_multiplier=0.1,
    )
    torch.testing.assert_close(relaxed, torch.tensor([2.0, 1.1, 0.2]))


def test_takeoff_expert_is_zero_outside_demand_and_non_leg_actions():
    residual = torch.ones(2, 5)
    gated = gate_demand_action_residual(
        residual,
        demand=torch.tensor([[0.0], [0.5]]),
        action_indices=[1, 3],
    )


def test_takeoff_vertical_velocity_error_is_gated_timed_and_non_saturating():
    from protomotions.envs.rewards.tracking import (
        compute_contact_transition_vertical_velocity_error,
    )

    current = torch.zeros(3, 9, 3)
    reference = torch.zeros(3, 9, 3)
    reference[:, 0, 2] = torch.tensor([3.0, 3.0, 3.0])
    future = torch.zeros(3, 2, 3)
    future[:, -1, 2] = 3.0
    contacts = torch.zeros(3, 9, dtype=torch.bool)
    contacts[0, 4] = True
    contacts[1, 4] = True
    reliability = torch.ones(3, 2)
    reliability[1] = 0.0

    error = compute_contact_transition_vertical_velocity_error(
        current,
        reference,
        future,
        contacts,
        reliability,
        support_body_ids=[3, 4, 7, 8],
        huber_delta=0.5,
    )
    # Smooth-L1(3 m/s, beta=.5) = 2.75.  Unsupported or unreliable phases
    # are exactly zero, so ordinary motions are unaffected.
    torch.testing.assert_close(error, torch.tensor([2.75, 0.0, 0.0]))

    current[0, 0, 2] = 2.0
    closer = compute_contact_transition_vertical_velocity_error(
        current,
        reference,
        future,
        contacts,
        reliability,
        support_body_ids=[3, 4, 7, 8],
        huber_delta=0.5,
    )
    assert 0.0 < closer[0].item() < error[0].item()
    torch.testing.assert_close(
        gated,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.5, 0.0, 0.5, 0.0],
            ]
        ),
    )
from protomotions.envs.rewards.tracking import (
    compute_reliability_blended_position_rew,
    compute_reliability_blended_velocity_rew,
)
from protomotions.envs.terminations.tracking import (
    compute_reliability_blended_tracking_error,
    mean_body_pos_error,
    reliability_blended_mean_body_pos_error,
)


def _target_pose_inputs():
    torch.manual_seed(17)
    envs, future, bodies = 2, 3, 4
    current_pos = torch.randn(envs, bodies, 3)
    current_rot = torch.zeros(envs, bodies, 4)
    current_rot[..., 3] = 1.0
    current_vel = torch.randn(envs, bodies, 3)
    current_ang_vel = torch.randn(envs, bodies, 3)
    ref_pos = torch.randn(envs, future, bodies, 3)
    ref_rot = torch.zeros(envs, future, bodies, 4)
    ref_rot[..., 3] = 1.0
    ref_vel = torch.randn(envs, future, bodies, 3)
    ref_ang_vel = torch.randn(envs, future, bodies, 3)
    return {
        "current_state_body_pos": current_pos,
        "current_state_body_rot": current_rot,
        "current_state_body_vel": current_vel,
        "current_state_body_ang_vel": current_ang_vel,
        "mimic_ref_pos": ref_pos,
        "mimic_ref_rot": ref_rot,
        "mimic_ref_vel": ref_vel,
        "mimic_ref_ang_vel": ref_ang_vel,
        "with_velocities": True,
        "w_last": True,
    }


def test_reliability_control_gate_has_trusted_and_untrusted_plateaus():
    confidence = torch.tensor([0.0, 0.35, 0.55, 0.75, 1.0])
    weight = calibrate_reference_reliability(
        confidence, low=0.35, high=0.75, smoothstep=True
    )
    torch.testing.assert_close(
        weight, torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0])
    )


def test_default_reliability_control_gate_preserves_confidence_exactly():
    confidence = torch.tensor([-0.1, 0.2, 0.7, 1.1])
    weight = calibrate_reference_reliability(confidence)
    torch.testing.assert_close(weight, torch.tensor([0.0, 0.2, 0.7, 1.0]))


def test_adapter_plasticity_increases_with_reference_uncertainty():
    reliability = torch.tensor([0.0, 0.25, 0.75, 1.0])
    gate = uncertainty_plasticity_gate(reliability, floor=0.05)
    torch.testing.assert_close(gate, torch.tensor([1.0, 0.7625, 0.2875, 0.05]))


def test_reliable_target_encoding_exactly_matches_pretrained_encoding():
    inputs = _target_pose_inputs()
    expected = build_max_coords_target_poses(**inputs)
    actual = build_reliability_gated_max_coords_target_poses(
        **inputs,
        reference_reliability=torch.ones(2, 1),
        minimum_global_weight=0.0,
    )
    torch.testing.assert_close(actual, expected)
    gravity_axis_actual = build_reliability_gated_max_coords_target_poses(
        **inputs,
        reference_reliability=torch.ones(2, 1),
        minimum_global_weight=0.0,
        gravity_axis_only=True,
    )
    torch.testing.assert_close(gravity_axis_actual, expected)


def test_reliable_future_context_preserves_original_temporal_command():
    inputs = _target_pose_inputs()
    expected = build_max_coords_target_poses_future_rel(
        current_state_body_pos=inputs["current_state_body_pos"],
        current_state_body_rot=inputs["current_state_body_rot"],
        mimic_ref_pos=inputs["mimic_ref_pos"],
        mimic_ref_rot=inputs["mimic_ref_rot"],
        w_last=True,
    )
    actual = build_reliability_gated_max_coords_target_poses_future_rel(
        current_state_body_pos=inputs["current_state_body_pos"],
        current_state_body_rot=inputs["current_state_body_rot"],
        mimic_ref_pos=inputs["mimic_ref_pos"],
        mimic_ref_rot=inputs["mimic_ref_rot"],
        future_reference_reliability=torch.ones(2, 3),
        w_last=True,
        include_reliability=False,
    )
    torch.testing.assert_close(actual, expected)


def test_future_context_rejects_untrusted_vertical_root_increments_only():
    inputs = _target_pose_inputs()
    kwargs = {
        "current_state_body_pos": inputs["current_state_body_pos"],
        "current_state_body_rot": inputs["current_state_body_rot"],
        "mimic_ref_rot": inputs["mimic_ref_rot"],
        "future_reference_reliability": torch.zeros(2, 3),
        "w_last": True,
        "minimum_global_weight": 0.0,
        "gravity_axis_only": True,
        "include_reliability": False,
    }
    baseline = build_reliability_gated_max_coords_target_poses_future_rel(
        mimic_ref_pos=inputs["mimic_ref_pos"], **kwargs
    )

    vertical = inputs["mimic_ref_pos"].clone()
    vertical[:, :, :, 2] += torch.tensor([0.3, 0.9, 1.7])[None, :, None]
    vertical_result = build_reliability_gated_max_coords_target_poses_future_rel(
        mimic_ref_pos=vertical, **kwargs
    )
    torch.testing.assert_close(vertical_result, baseline)

    horizontal = inputs["mimic_ref_pos"].clone()
    horizontal[:, :, :, 0] += torch.tensor([0.2, 0.5, 0.9])[None, :, None]
    horizontal_result = build_reliability_gated_max_coords_target_poses_future_rel(
        mimic_ref_pos=horizontal, **kwargs
    )
    assert not torch.allclose(horizontal_result, baseline)


def test_future_context_exposes_only_reliable_vertical_anchors():
    inputs = _target_pose_inputs()
    reliability = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]])
    context_without = build_reliability_gated_max_coords_target_poses_future_rel(
        current_state_body_pos=inputs["current_state_body_pos"],
        current_state_body_rot=inputs["current_state_body_rot"],
        mimic_ref_pos=inputs["mimic_ref_pos"],
        mimic_ref_rot=inputs["mimic_ref_rot"],
        future_reference_reliability=reliability,
        w_last=True,
        include_reliability=True,
        include_trusted_vertical_anchors=False,
    )
    context_with = build_reliability_gated_max_coords_target_poses_future_rel(
        current_state_body_pos=inputs["current_state_body_pos"],
        current_state_body_rot=inputs["current_state_body_rot"],
        mimic_ref_pos=inputs["mimic_ref_pos"],
        mimic_ref_rot=inputs["mimic_ref_rot"],
        future_reference_reliability=reliability,
        w_last=True,
        include_reliability=True,
        include_trusted_vertical_anchors=True,
    )
    anchors = context_with[:, context_without.shape[1] :]
    expected = reliability * (
        inputs["mimic_ref_pos"][:, :, 0, 2]
        - inputs["current_state_body_pos"][:, None, 0, 2]
    )
    torch.testing.assert_close(anchors, expected)
    torch.testing.assert_close(anchors[reliability == 0], torch.zeros(2))


def test_future_context_exposes_reliable_vertical_launch_velocity():
    inputs = _target_pose_inputs()
    reliability = torch.tensor([[0.0, 0.5, 1.0], [1.0, 0.0, 0.25]])
    context_without = build_reliability_gated_max_coords_target_poses_future_rel(
        current_state_body_pos=inputs["current_state_body_pos"],
        current_state_body_rot=inputs["current_state_body_rot"],
        mimic_ref_pos=inputs["mimic_ref_pos"],
        mimic_ref_rot=inputs["mimic_ref_rot"],
        future_reference_reliability=reliability,
        w_last=True,
        include_reliability=True,
        include_trusted_vertical_anchors=True,
        include_trusted_vertical_velocity=False,
    )
    context_with = build_reliability_gated_max_coords_target_poses_future_rel(
        current_state_body_pos=inputs["current_state_body_pos"],
        current_state_body_rot=inputs["current_state_body_rot"],
        mimic_ref_pos=inputs["mimic_ref_pos"],
        mimic_ref_rot=inputs["mimic_ref_rot"],
        future_reference_reliability=reliability,
        w_last=True,
        include_reliability=True,
        include_trusted_vertical_anchors=True,
        mimic_ref_vel=inputs["mimic_ref_vel"],
        include_trusted_vertical_velocity=True,
    )
    velocity = context_with[:, context_without.shape[1] :]
    expected = reliability * inputs["mimic_ref_vel"][:, :, 0, 2]
    torch.testing.assert_close(velocity, expected)
    torch.testing.assert_close(velocity[reliability == 0], torch.zeros(2))


def test_gravity_axis_target_relaxes_vertical_but_preserves_horizontal_path():
    inputs = _target_pose_inputs()
    reliability = torch.zeros(2, 1)
    baseline = build_reliability_gated_max_coords_target_poses(
        **inputs,
        reference_reliability=reliability,
        minimum_global_weight=0.0,
        gravity_axis_only=True,
    )

    vertical = dict(inputs)
    vertical["mimic_ref_pos"] = inputs["mimic_ref_pos"] + torch.tensor(
        [0.0, 0.0, 0.8]
    )
    vertical["mimic_ref_vel"] = inputs["mimic_ref_vel"] + torch.tensor(
        [0.0, 0.0, -1.2]
    )
    vertical_result = build_reliability_gated_max_coords_target_poses(
        **vertical,
        reference_reliability=reliability,
        minimum_global_weight=0.0,
        gravity_axis_only=True,
    )
    torch.testing.assert_close(vertical_result, baseline)

    horizontal = dict(inputs)
    horizontal["mimic_ref_pos"] = inputs["mimic_ref_pos"] + torch.tensor(
        [0.8, -0.4, 0.0]
    )
    horizontal_result = build_reliability_gated_max_coords_target_poses(
        **horizontal,
        reference_reliability=reliability,
        minimum_global_weight=0.0,
        gravity_axis_only=True,
    )
    assert not torch.allclose(horizontal_result, baseline)


def test_unreliable_target_encoding_ignores_shared_root_translation_noise():
    inputs = _target_pose_inputs()
    shifted = dict(inputs)
    position_shift = torch.tensor(
        [[0.8, -0.3, 1.1], [-0.4, 0.7, -0.9]], dtype=torch.float32
    )
    velocity_shift = torch.tensor(
        [[1.2, -0.2, 0.6], [-0.5, 0.4, -1.3]], dtype=torch.float32
    )
    shifted["mimic_ref_pos"] = (
        inputs["mimic_ref_pos"] + position_shift[:, None, None, :]
    )
    shifted["mimic_ref_vel"] = (
        inputs["mimic_ref_vel"] + velocity_shift[:, None, None, :]
    )

    reliability = torch.zeros(2, 1)
    baseline = build_reliability_gated_max_coords_target_poses(
        **inputs,
        reference_reliability=reliability,
        minimum_global_weight=0.0,
    )
    noisy = build_reliability_gated_max_coords_target_poses(
        **shifted,
        reference_reliability=reliability,
        minimum_global_weight=0.0,
    )
    torch.testing.assert_close(noisy, baseline)


def test_low_reliability_ignores_shared_root_shift_but_keeps_articulation():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    shifted = reference + torch.tensor([[[0.0, 0.0, 0.5]]])
    low = compute_reliability_blended_position_rew(
        shifted,
        reference,
        torch.tensor([0.0]),
        minimum_absolute_weight=0.0,
    )
    high = compute_reliability_blended_position_rew(
        shifted,
        reference,
        torch.tensor([1.0]),
        minimum_absolute_weight=0.0,
    )
    torch.testing.assert_close(low, torch.ones_like(low))
    assert high.item() < 0.15


def test_gravity_axis_reward_ignores_only_shared_vertical_shift():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    vertical = reference + torch.tensor([[[0.0, 0.0, 0.5]]])
    vertical_reward = compute_reliability_blended_position_rew(
        vertical,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
        gravity_axis_only=True,
    )
    torch.testing.assert_close(vertical_reward, torch.ones_like(vertical_reward))

    horizontal = reference + torch.tensor([[[0.5, 0.0, 0.0]]])
    horizontal_reward = compute_reliability_blended_position_rew(
        horizontal,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
        gravity_axis_only=True,
    )
    assert horizontal_reward.item() < 0.15


def test_low_reliability_still_penalizes_wrong_relative_pose():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    wrong = reference.clone()
    wrong[:, 1, 2] += 0.5
    reward = compute_reliability_blended_position_rew(
        wrong,
        reference,
        torch.tensor([0.0]),
        minimum_absolute_weight=0.0,
    )
    assert reward.item() < 0.4


def test_reliability_metric_matches_global_error_when_trusted():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    shifted = reference + torch.tensor([[[0.0, 0.0, 0.5]]])
    expected = mean_body_pos_error(shifted, reference)
    actual = reliability_blended_mean_body_pos_error(
        shifted,
        reference,
        torch.ones(1),
        minimum_absolute_weight=0.0,
    )
    torch.testing.assert_close(actual, expected)


def test_reliability_metric_ignores_only_shared_shift_when_untrusted():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    shared_shift = reference + torch.tensor([[[0.2, -0.4, 0.5]]])
    shared_error = reliability_blended_mean_body_pos_error(
        shared_shift,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
    )
    torch.testing.assert_close(shared_error, torch.zeros_like(shared_error))

    wrong_articulation = shared_shift.clone()
    wrong_articulation[:, 1, 2] += 0.5
    articulation_error = reliability_blended_mean_body_pos_error(
        wrong_articulation,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
    )
    assert articulation_error.item() > 0.2


def test_gravity_axis_metric_still_measures_horizontal_root_error():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    vertical = reference + torch.tensor([[[0.0, 0.0, 0.5]]])
    vertical_error = reliability_blended_mean_body_pos_error(
        vertical,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
        gravity_axis_only=True,
    )
    torch.testing.assert_close(vertical_error, torch.zeros_like(vertical_error))

    horizontal = reference + torch.tensor([[[0.5, 0.0, 0.0]]])
    horizontal_error = reliability_blended_mean_body_pos_error(
        horizontal,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
        gravity_axis_only=True,
    )
    torch.testing.assert_close(horizontal_error, torch.full_like(horizontal_error, 0.5))


def test_untrusted_velocity_reward_ignores_shared_root_velocity_only():
    reference = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]], dtype=torch.float32
    )
    shared_shift = reference + torch.tensor([[[2.0, -1.0, 0.5]]])
    shared_reward = compute_reliability_blended_velocity_rew(
        shared_shift,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
    )
    torch.testing.assert_close(shared_reward, torch.ones_like(shared_reward))

    wrong_articulation = shared_shift.clone()
    wrong_articulation[:, 1, 2] += 1.0
    articulation_reward = compute_reliability_blended_velocity_rew(
        wrong_articulation,
        reference,
        torch.zeros(1),
        minimum_absolute_weight=0.0,
    )
    assert articulation_reward.item() < shared_reward.item() - 0.05


def test_untrusted_shared_translation_does_not_trigger_tracking_reset():
    reference = torch.tensor(
        [[[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]]], dtype=torch.float32
    )
    shifted = reference + torch.tensor([[[0.0, 0.0, 1.0]]])
    low_confidence = compute_reliability_blended_tracking_error(
        shifted,
        reference,
        torch.zeros(1),
        error_threshold=0.5,
        minimum_absolute_weight=0.0,
    )
    high_confidence = compute_reliability_blended_tracking_error(
        shifted,
        reference,
        torch.ones(1),
        error_threshold=0.5,
        minimum_absolute_weight=0.0,
    )
    assert not low_confidence.item()
    assert high_confidence.item()
