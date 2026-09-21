# SPDX-License-Identifier: Apache-2.0

import torch

from data.scripts.retime_motion_by_physical_reliability import (
    build_retiming_grid,
    build_retiming_grid_from_interval_rates,
    compute_system_center_of_mass,
    enforce_ballistic_com_velocity,
    enforce_ballistic_translation_velocity,
    phase_rate_from_reliability,
    propagate_ballistic_takeoff_phase_rates,
    project_ballistic_root_height,
    project_ballistic_root_translation,
    reachability_aware_interval_phase_rates,
    recover_seed_connected_foot_support,
    solve_surface_support_root_correction,
)


def test_reliable_and_contact_frames_keep_observed_phase_rate():
    reliability = torch.tensor([0.1, 0.1, 0.9, 0.1, 0.1])
    contacts = torch.zeros(5, 2, dtype=torch.bool)
    contacts[3, 0] = True
    rate = phase_rate_from_reliability(
        reliability,
        contacts,
        max_phase_rate=3.0,
        transition_radius=0,
    )

    torch.testing.assert_close(rate[:2], torch.full((2,), 3.0, dtype=torch.float64))
    assert rate[2].item() == 1.0
    assert rate[3].item() == 1.0
    assert rate[4].item() == 3.0


def test_constant_phase_rate_has_exact_endpoints_and_expected_duration():
    coordinates, output_dt, report = build_retiming_grid(
        torch.full((80,), 2.0),
        source_dt=1.0 / 15.0,
        requested_output_fps=30.0,
    )

    assert coordinates[0].item() == 0.0
    assert coordinates[-1].item() == 79.0
    assert torch.all(coordinates[1:] > coordinates[:-1])
    expected_duration = 79.0 / 15.0 / 2.0
    assert abs(report["calibrated_duration_s"] - expected_duration) < 1e-10
    assert abs((len(coordinates) - 1) * output_dt - expected_duration) < 1e-10
    assert abs(report["effective_mean_phase_rate"] - 2.0) < 1e-10


def test_identity_rate_preserves_source_frame_coordinates_at_source_fps():
    coordinates, output_dt, report = build_retiming_grid(
        torch.ones(12),
        source_dt=1.0 / 30.0,
        requested_output_fps=30.0,
    )

    torch.testing.assert_close(
        coordinates, torch.arange(12, dtype=torch.float64), atol=1e-10, rtol=0.0
    )
    assert abs(output_dt - 1.0 / 30.0) < 1e-10
    assert report["output_frames"] == 12


def test_latent_support_only_extends_candidate_run_connected_to_real_seed():
    contacts = torch.zeros(12, 6, dtype=torch.bool)
    contacts[5:7, 2] = True
    clearance = torch.full((12, 6), 2.0)
    speed = torch.zeros(12, 6)
    # One coherent stance candidate surrounds the observed seed.  The isolated
    # low foot at frame 10 must not become a hallucinated contact.
    clearance[2:9, 2] = torch.tensor([0.40, 0.30, 0.20, 0.06, 0.05, 0.15, 0.35])
    clearance[10, 2] = 0.10

    augmented, diagnostics = recover_seed_connected_foot_support(
        contacts,
        clearance,
        speed,
        foot_body_ids=[2, 3],
        maximum_clearance_m=0.42,
    )

    assert augmented[2:9, 2].all()
    assert not augmented[10].any()
    assert diagnostics["recovered"][2:5].all()
    assert diagnostics["recovered"][7:9].all()
    assert not diagnostics["recovered"][5:7].any()


def test_surface_support_projection_moves_shared_root_into_clearance_band():
    contacts = torch.zeros(10, 4, dtype=torch.bool)
    contacts[2:8, 1] = True
    original = torch.zeros_like(contacts)
    original[4:6, 1] = True
    clearance = torch.full((10, 4), 1.0)
    clearance[2:8, 1] = 0.38

    correction = solve_surface_support_root_correction(
        contacts,
        original,
        clearance,
        clearance_upper_m=0.08,
        data_weight=0.1,
        observed_support_weight=100.0,
        recovered_support_weight=100.0,
        correction_jerk_weight=1.0,
    )

    corrected = clearance[2:8, 1] + correction[2:8]
    assert float(corrected.max()) < 0.09
    assert float(correction[3:7].median()) < -0.28


def test_reachability_phase_compresses_slow_motion_flight_not_support():
    com = torch.zeros(12, 3)
    # Frames 2-4 and 9-11 are supported.  A slow-motion 0.7 m ascent lies
    # between them and would require excessive launch speed at the source rate.
    contacts = torch.zeros(12, 2, dtype=torch.bool)
    contacts[:5, 0] = True
    contacts[9:, 1] = True
    com[4, 2] = 1.0
    com[9:, 2] = 1.7
    com[5:9, 2] = torch.linspace(1.14, 1.56, 4)

    rates, reports = reachability_aware_interval_phase_rates(
        com,
        contacts,
        source_dt=0.2,
        max_phase_rate=3.0,
    )

    torch.testing.assert_close(rates[2:4], torch.ones(2, dtype=torch.float64))
    assert torch.all(rates[4:9] > 1.0)
    assert len(reports) == 1
    assert reports[0]["within_speed_limits"]
    assert reports[0]["phase_rate"] > 1.0


def test_takeoff_phase_propagation_accelerates_only_preceding_support():
    rates = torch.ones(11, dtype=torch.float64)
    rates[4:9] = 1.7
    com = torch.zeros(12, 3)
    com[2, 2] = 0.0
    com[3, 2] = 0.05
    com[4, 2] = 0.10
    contacts = torch.zeros(12, 2, dtype=torch.bool)
    contacts[:5, 0] = True
    contacts[9:, 1] = True
    reports = [
        {
            "start_frame": 4,
            "end_frame": 9,
            "start_is_support": True,
            "end_is_support": True,
            "initial_vertical_speed_m_s": 3.0,
        }
    ]

    adjusted, adjusted_reports = propagate_ballistic_takeoff_phase_rates(
        rates,
        com,
        contacts,
        reports,
        source_dt=0.1,
        preparation_duration_s=0.3,
        velocity_estimation_duration_s=0.2,
        max_phase_rate=6.0,
    )

    # The last crouch minimum is frame 2. Constant acceleration over the two
    # positive-height intervals accelerates both, with the shorter terminal
    # physical interval receiving the larger phase rate.
    torch.testing.assert_close(adjusted[:2], rates[:2])
    assert adjusted[2] > 1.0
    assert adjusted[3] > adjusted[2]
    # Flight timing and post-landing support are not rewritten by this step.
    torch.testing.assert_close(adjusted[4:9], rates[4:9])
    torch.testing.assert_close(adjusted[9:], rates[9:])
    report = adjusted_reports[0]
    assert report["takeoff_phase_propagated"]
    assert report["takeoff_preparation_start_frame"] == 2
    assert report["takeoff_constant_acceleration_m_s2"] > 0.0
    assert report["predicted_retimed_takeoff_vertical_speed_m_s"] > 2.5


def test_interval_rate_grid_integrates_rates_exactly():
    rates = torch.tensor([1.0, 2.0, 2.0, 1.0], dtype=torch.float64)
    coordinates, output_dt, report = build_retiming_grid_from_interval_rates(
        rates, source_dt=0.1, requested_output_fps=20.0
    )

    expected_duration = 0.1 + 0.05 + 0.05 + 0.1
    assert abs(report["calibrated_duration_s"] - expected_duration) < 1e-12
    assert coordinates[0].item() == 0.0
    assert coordinates[-1].item() == 4.0
    assert abs((len(coordinates) - 1) * output_dt - expected_duration) < 1e-12


def test_ballistic_projection_preserves_support_anchor_and_has_gravity():
    root = torch.zeros(8, 3)
    root[:, 0] = torch.arange(8)
    root[:, 2] = torch.linspace(2.0, 1.3, 8)
    contacts = torch.zeros(8, 2, dtype=torch.bool)
    contacts[5:, 0] = True

    projected, segments = project_ballistic_root_height(
        root, contacts, dt=0.1, gravity=9.81
    )

    torch.testing.assert_close(projected[:, :2], root[:, :2])
    assert projected[0, 2].item() == root[0, 2].item()
    assert projected[5, 2].item() == root[5, 2].item()
    torch.testing.assert_close(projected[5:], root[5:])
    acceleration = (
        projected[2:5, 2] - 2.0 * projected[1:4, 2] + projected[:3, 2]
    ) / 0.01
    torch.testing.assert_close(
        acceleration, torch.full_like(acceleration, -9.81), atol=5e-5, rtol=0.0
    )
    assert len(segments) == 1


def test_ballistic_projection_requires_contact_evidence():
    root = torch.randn(10, 3)
    contacts = torch.zeros(10, 2, dtype=torch.bool)
    projected, segments = project_ballistic_root_height(root, contacts, dt=0.1)
    torch.testing.assert_close(projected, root)
    assert segments == []


def test_ballistic_velocity_is_analytic_and_preserves_relative_motion():
    velocity = torch.randn(8, 3, 3)
    relative_before = velocity[:, 1:] - velocity[:, :1]
    segments = [
        {
            "start_frame": 0,
            "end_frame": 5,
            "end_is_support": True,
            "initial_vertical_velocity_m_s": 2.0,
        }
    ]

    corrected = enforce_ballistic_translation_velocity(
        velocity, segments, dt=0.1, gravity=9.81
    )

    expected = 2.0 - 9.81 * torch.arange(5) * 0.1
    torch.testing.assert_close(corrected[:5, 0, 2], expected)
    # The terminal support and all later frames remain post-impact FK values.
    torch.testing.assert_close(corrected[5:], velocity[5:])
    # A common translation delta must not alter articulated relative velocity.
    torch.testing.assert_close(corrected[:, 1:] - corrected[:, :1], relative_before)


def test_ballistic_velocity_includes_terminal_clip_flight_sample():
    velocity = torch.zeros(5, 2, 3)
    segments = [
        {
            "start_frame": 2,
            "end_frame": 4,
            "end_is_support": False,
            "initial_vertical_velocity_m_s": 1.0,
        }
    ]

    corrected = enforce_ballistic_translation_velocity(
        velocity, segments, dt=0.1, gravity=10.0
    )

    torch.testing.assert_close(corrected[2:, 0, 2], torch.tensor([1.0, 0.0, -1.0]))


def test_3d_ballistic_projection_has_constant_horizontal_momentum():
    root = torch.zeros(8, 3)
    root[:, 0] = torch.tensor([0.0, 0.8, -0.2, 1.4, 0.3, 2.0, 2.1, 2.2])
    root[:, 1] = torch.tensor([1.0, 0.7, 1.5, 0.2, 2.0, 3.0, 3.1, 3.2])
    root[:, 2] = torch.linspace(2.0, 1.3, 8)
    contacts = torch.zeros(8, 2, dtype=torch.bool)
    contacts[5:, 0] = True

    projected, segments = project_ballistic_root_translation(
        root, contacts, dt=0.1, gravity=9.81
    )

    torch.testing.assert_close(projected[0], root[0])
    torch.testing.assert_close(projected[5:], root[5:])
    acceleration = (
        projected[2:5] - 2.0 * projected[1:4] + projected[:3]
    ) / 0.01
    torch.testing.assert_close(
        acceleration[:, :2], torch.zeros_like(acceleration[:, :2]), atol=5e-5, rtol=0.0
    )
    torch.testing.assert_close(
        acceleration[:, 2], torch.full_like(acceleration[:, 2], -9.81), atol=5e-5, rtol=0.0
    )
    assert len(segments) == 1
    assert len(segments[0]["initial_velocity_m_s"]) == 3


def test_3d_ballistic_velocity_is_exact_and_preserves_articulation():
    velocity = torch.randn(7, 4, 3)
    relative_before = velocity[:, 1:] - velocity[:, :1]
    segments = [
        {
            "start_frame": 1,
            "end_frame": 5,
            "end_is_support": True,
            "initial_velocity_m_s": [2.0, -1.0, 3.0],
        }
    ]

    corrected = enforce_ballistic_translation_velocity(
        velocity, segments, dt=0.1, gravity=10.0
    )

    expected = torch.tensor(
        [[2.0, -1.0, 3.0], [2.0, -1.0, 2.0], [2.0, -1.0, 1.0], [2.0, -1.0, 0.0]]
    )
    torch.testing.assert_close(corrected[1:5, 0], expected)
    torch.testing.assert_close(corrected[5:], velocity[5:])
    torch.testing.assert_close(corrected[:, 1:] - corrected[:, :1], relative_before)


def test_system_center_of_mass_uses_inertial_offsets_and_masses():
    positions = torch.tensor(
        [[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=torch.float32
    )
    rotations = torch.tensor(
        [[[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    masses = torch.tensor([1.0, 3.0])
    offsets = torch.tensor([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])

    com = compute_system_center_of_mass(positions, rotations, masses, offsets)

    torch.testing.assert_close(com, torch.tensor([[1.5, -0.5, 0.0]]))


def test_com_velocity_accounts_for_articulation_relative_to_root():
    # The COM moves +0.1 m/frame relative to a stationary pelvis.  To obtain a
    # desired +2 m/s COM velocity, root reset velocity must therefore be +1 m/s.
    velocity = torch.zeros(5, 2, 3)
    root = torch.zeros(5, 3)
    com = torch.zeros(5, 3)
    com[:, 0] = torch.arange(5) * 0.1
    segments = [
        {
            "start_frame": 0,
            "end_frame": 4,
            "end_is_support": False,
            "initial_velocity_m_s": [2.0, 0.0, 1.0],
        }
    ]

    corrected = enforce_ballistic_com_velocity(
        velocity, root, com, segments, dt=0.1, gravity=10.0
    )

    torch.testing.assert_close(corrected[:, 0, 0], torch.ones(5))
    torch.testing.assert_close(
        corrected[:, 0, 2], torch.tensor([1.0, 0.0, -1.0, -2.0, -3.0])
    )
    torch.testing.assert_close(corrected[:, 1] - corrected[:, 0], torch.zeros(5, 3))
