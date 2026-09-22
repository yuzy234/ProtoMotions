from types import SimpleNamespace

from protomotions.envs.base_env.env import BaseEnv


def _component(**params):
    return SimpleNamespace(static_params=params)


def test_zero_weight_reward_is_inactive():
    assert not BaseEnv._reward_component_is_active(_component(weight=0.0))


def test_log_only_zero_weight_diagnostic_remains_active():
    assert BaseEnv._reward_component_is_active(
        _component(weight=0.0, log_only=True)
    )


def test_multiplicative_reward_remains_active_without_weight():
    assert BaseEnv._reward_component_is_active(_component(multiplicative=True))
