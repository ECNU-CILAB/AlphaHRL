from typing import Optional, Sequence

import gymnasium as gym

from AlphaHRL.HRL.data.calculator import TensorAlphaCalculator
from AlphaHRL.HRL.data.expression import Expression
from AlphaHRL.HRL.rl.env.high_level_core import HighLevelEnvConfig, HighLevelEnvCore


class HighLevelEnvWrapper(gym.Wrapper):
    env: HighLevelEnvCore

    def __init__(self, env: HighLevelEnvCore):
        super().__init__(env)

    @property
    def n_factors(self) -> int:
        return self.env.n_factors

    @property
    def factor_names(self):
        return self.env.factor_names

    @property
    def base_weights(self):
        return self.env.base_weights

    @property
    def n_action_factors(self) -> int:
        return self.env.n_action_factors

    @property
    def action_factor_indices(self):
        return self.env.action_factor_indices

    @property
    def action_factor_names(self):
        return self.env.action_factor_names

    @property
    def action_mode(self) -> str:
        return self.env.action_mode

    @property
    def rebalance_interval_days(self) -> int:
        return self.env.rebalance_interval_days

    @property
    def last_episode_summary(self):
        return self.env.last_episode_summary

    def reward(self, reward: float) -> float:
        return reward

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return obs, self.reward(reward), done, truncated, info


def HighLevelEnv(
    exprs: Sequence[Expression],
    calculator: TensorAlphaCalculator,
    base_weights: Optional[Sequence[float]] = None,
    config: Optional[HighLevelEnvConfig] = None,
):
    return HighLevelEnvWrapper(
        HighLevelEnvCore(
            exprs=exprs,
            calculator=calculator,
            base_weights=base_weights,
            config=config,
        )
    )
