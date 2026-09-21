import torch

from protomotions.utils.root_reference_cem import (
    RootReferenceCEMConfig,
    fit_root_knots,
    interpolate_root_knots,
    sample_cem_population,
    trajectory_regularization,
    update_cem_distribution,
)


def test_knot_interpolation_preserves_endpoints_and_shape():
    knots = torch.tensor([[0.0, 1.0, 0.0], [1.0, 2.0, 3.0]])
    trajectory = interpolate_root_knots(knots, frame_count=9)
    assert trajectory.shape == (2, 9)
    torch.testing.assert_close(trajectory[:, 0], knots[:, 0])
    torch.testing.assert_close(trajectory[:, -1], knots[:, -1])
    torch.testing.assert_close(fit_root_knots(trajectory[0], 3), knots[0])


def test_population_contains_mean_and_zero_and_respects_bounds():
    mean = torch.tensor([0.1, -0.2, 0.0])
    std = torch.full_like(mean, 0.5)
    generator = torch.Generator().manual_seed(7)
    candidates = sample_cem_population(
        mean, std, population=10, lower=-0.3, upper=0.2, generator=generator
    )
    torch.testing.assert_close(candidates[0], mean.clamp(-0.3, 0.2))
    torch.testing.assert_close(candidates[1], torch.zeros_like(mean))
    assert candidates.min() >= -0.3
    assert candidates.max() <= 0.2


def test_cem_update_moves_distribution_toward_best_candidates():
    candidates = torch.tensor([[-1.0], [0.0], [1.0], [2.0]])
    scores = -(candidates[:, 0] - 1.5).square()
    mean, std, elite = update_cem_distribution(
        torch.zeros(1),
        torch.ones(1),
        candidates,
        scores,
        elite_count=2,
        minimum_std=0.01,
        momentum=0.0,
    )
    torch.testing.assert_close(mean, torch.tensor([1.5]))
    torch.testing.assert_close(std, torch.tensor([0.5]))
    assert set(elite.tolist()) == {2, 3}


def test_reliability_tethers_only_trusted_frames_strongly():
    config = RootReferenceCEMConfig(
        reliable_reference_weight=1.0,
        spline_curvature_weight=0.0,
    )
    trajectory = torch.full((1, 6), 0.2)
    knots = torch.full((1, 3), 0.2)
    trusted = trajectory_regularization(
        trajectory, knots, torch.ones(6), config
    )["total"]
    unreliable = trajectory_regularization(
        trajectory, knots, torch.zeros(6), config
    )["total"]
    assert trusted.item() > unreliable.item() * 10.0


def test_observation_anchor_is_normalized_by_metric_uncertainty():
    config = RootReferenceCEMConfig(
        reliable_reference_weight=1.0,
        spline_curvature_weight=0.0,
        observation_sigma_min_m=0.025,
        observation_sigma_max_m=0.20,
    )
    trajectory = torch.full((1, 6), 0.05)
    knots = torch.full((1, 3), 0.05)
    trusted = trajectory_regularization(
        trajectory, knots, torch.ones(6), config
    )["reliable_reference"]
    uncertain = trajectory_regularization(
        trajectory, knots, torch.zeros(6), config
    )["reliable_reference"]
    torch.testing.assert_close(trusted, torch.tensor([4.0]))
    torch.testing.assert_close(uncertain, torch.tensor([0.0625]))
