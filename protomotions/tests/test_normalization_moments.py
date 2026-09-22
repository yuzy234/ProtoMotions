import torch

from protomotions.agents.utils.normalization import combine_moments


def test_combine_moments_preserves_small_batch_at_large_count() -> None:
    old_count = torch.tensor(14_219_018_240, dtype=torch.long)
    batch_count = 512
    old_mean = torch.tensor([0.0], dtype=torch.float64)
    old_var = torch.tensor([1.0], dtype=torch.float64)
    batch_mean = torch.tensor([10.0], dtype=torch.float64)
    batch_var = torch.tensor([4.0], dtype=torch.float64)

    mean, variance, count = combine_moments(
        [old_mean, batch_mean],
        [old_var, batch_var],
        [old_count, batch_count],
    )

    assert count.item() == old_count.item() + batch_count
    expected_mean = (
        old_mean * old_count.item() + batch_mean * batch_count
    ) / count.item()
    torch.testing.assert_close(mean, expected_mean)
    assert torch.isfinite(variance).all()
