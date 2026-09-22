from types import SimpleNamespace

import torch

from protomotions.agents.base_agent.agent import BaseAgent
from protomotions.agents.ppo.agent import PPO


class _StateRecorder:
    def __init__(self):
        self.loaded = []

    def load_state_dict(self, state):
        self.loaded.append(state)


def _base_shell(resume: bool):
    agent = object.__new__(BaseAgent)
    agent._resume_training_state_on_load = resume
    agent._load_reward_normalization_on_load = resume
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent._early_stop_best_score = None
    agent._early_stop_bad_evals = 0
    agent.config = SimpleNamespace(normalize_rewards=True)
    agent.running_reward_norm = _StateRecorder()
    agent.model = _StateRecorder()
    return agent


def _checkpoint():
    return {
        "epoch": 6830,
        "step_count": 7_160_000_000,
        "run_start_time": 123.0,
        "best_evaluated_score": 0.8,
        "early_stop_best_score": 0.7,
        "early_stop_bad_evals": 3,
        "running_reward_norm": {"count": 99},
        "model": {"weight": 1},
    }


def test_warm_start_keeps_fresh_progress_and_reward_statistics():
    agent = _base_shell(resume=False)
    BaseAgent.load_parameters(agent, _checkpoint())

    assert agent.current_epoch == 0
    assert agent.step_count == 0
    assert agent.best_evaluated_score is None
    assert agent.running_reward_norm.loaded == []
    assert agent.model.loaded == [{"weight": 1}]


def test_resume_restores_progress_and_early_stopping_state():
    agent = _base_shell(resume=True)
    BaseAgent.load_parameters(agent, _checkpoint())

    assert agent.current_epoch == 6830
    assert agent.step_count == 7_160_000_000
    assert agent.best_evaluated_score == 0.8
    assert agent._early_stop_best_score == 0.7
    assert agent._early_stop_bad_evals == 3
    assert agent.running_reward_norm.loaded == [{"count": 99}]


def test_inference_can_load_reward_statistics_without_training_progress():
    agent = _base_shell(resume=False)
    agent._load_reward_normalization_on_load = True
    BaseAgent.load_parameters(agent, _checkpoint())

    assert agent.current_epoch == 0
    assert agent.step_count == 0
    assert agent.running_reward_norm.loaded == [{"count": 99}]


def _ppo_shell(resume: bool):
    agent = object.__new__(PPO)
    agent._resume_training_state_on_load = resume
    agent.current_epoch = 0
    agent.step_count = 0
    agent.fit_start_time = None
    agent.best_evaluated_score = None
    agent._early_stop_best_score = None
    agent._early_stop_bad_evals = 0
    agent.config = SimpleNamespace(
        normalize_rewards=False,
        model=SimpleNamespace(
            actor=SimpleNamespace(actor_logstd=-2.9, learnable_std=False)
        ),
        adaptive_lr=SimpleNamespace(enabled=False),
        advantage_normalization=SimpleNamespace(enabled=False, use_ema=False),
    )
    agent.model = _StateRecorder()
    agent.actor = SimpleNamespace(logstd=torch.nn.Parameter(torch.tensor([-2.9])))
    agent.actor_optimizer = _StateRecorder()
    agent.critic_optimizer = _StateRecorder()
    return agent


def test_warm_start_does_not_restore_adam_state():
    agent = _ppo_shell(resume=False)
    state = _checkpoint()
    state.update(
        {
            "actor_optimizer": {"state": "actor"},
            "critic_optimizer": {"state": "critic"},
        }
    )
    PPO.load_parameters(agent, state)

    assert agent.actor_optimizer.loaded == []
    assert agent.critic_optimizer.loaded == []


def test_resume_restores_adam_state():
    agent = _ppo_shell(resume=True)
    state = _checkpoint()
    state.update(
        {
            "actor_optimizer": {"state": "actor"},
            "critic_optimizer": {"state": "critic"},
        }
    )
    PPO.load_parameters(agent, state)

    assert agent.actor_optimizer.loaded == [{"state": "actor"}]
    assert agent.critic_optimizer.loaded == [{"state": "critic"}]
