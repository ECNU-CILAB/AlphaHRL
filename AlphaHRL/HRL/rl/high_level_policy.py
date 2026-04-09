import gymnasium as gym
import torch
from torch import Tensor, nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class HighLevelSequenceNet(BaseFeaturesExtractor):
    """
    Sequence encoder for high-level daily allocation observations.

    The environment emits an observation of shape [time_rows, n_factors]. A
    lightweight LSTM is used to summarize the recent factor regime into a fixed
    embedding for PPO.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        d_model: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, d_model)

        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("HighLevelSequenceNet requires a Box observation space")
        if len(observation_space.shape) != 2:
            raise ValueError(
                f"Expected 2D observations, got shape={observation_space.shape}"
            )

        _, n_factors = observation_space.shape
        self._proj = nn.Linear(n_factors, d_model)
        self._norm = nn.LayerNorm(d_model)
        self._lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, obs: Tensor) -> Tensor:
        x = obs.float()
        x = self._norm(self._proj(x))
        _, (hidden, _) = self._lstm(x)
        return hidden[-1]
