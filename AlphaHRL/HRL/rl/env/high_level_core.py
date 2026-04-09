from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from torch import Tensor

from AlphaHRL.HRL.data.calculator import TensorAlphaCalculator
from AlphaHRL.HRL.data.expression import Expression
from AlphaHRL.HRL.utils.correlation import batch_pearsonr, batch_spearmanr


_VALID_ACTION_MODES = {"absolute", "residual", "gate"}
_VALID_REWARD_MODES = {"ic", "rank_ic", "ic_rank_mix"}


@dataclass(frozen=True)
class HighLevelEnvConfig:
    lookback_days: int = 20
    episode_days: Optional[int] = 252
    target_horizon_days: int = 20
    rebalance_interval_days: int = 1
    max_leverage: float = 1.0
    action_scale: float = 0.1
    action_mode: str = "gate"
    action_top_k: Optional[int] = None
    reward_mode: str = "ic"
    weight_l2_penalty: float = 0.0
    base_deviation_penalty: float = 0.0
    smoothness_penalty: float = 0.0
    init_prev_weights_with_base: bool = True
    random_start: bool = True
    clip_observation: float = 1.0

    def __post_init__(self) -> None:
        if self.lookback_days < 1:
            raise ValueError("lookback_days must be >= 1")
        if self.episode_days is not None and self.episode_days < 1:
            raise ValueError("episode_days must be >= 1 when provided")
        if self.target_horizon_days < 1:
            raise ValueError("target_horizon_days must be >= 1")
        if self.rebalance_interval_days < 1:
            raise ValueError("rebalance_interval_days must be >= 1")
        if self.max_leverage <= 0.0:
            raise ValueError("max_leverage must be > 0")
        if self.action_scale < 0.0:
            raise ValueError("action_scale must be >= 0")
        if self.action_mode not in _VALID_ACTION_MODES:
            raise ValueError(
                f"Invalid action_mode: {self.action_mode}. "
                f"Expected one of {sorted(_VALID_ACTION_MODES)}."
            )
        if self.action_top_k is not None and self.action_top_k < 1:
            raise ValueError("action_top_k must be >= 1 when provided")
        if self.reward_mode not in _VALID_REWARD_MODES:
            raise ValueError(
                f"Invalid reward_mode: {self.reward_mode}. "
                f"Expected one of {sorted(_VALID_REWARD_MODES)}."
            )
        if self.weight_l2_penalty < 0.0:
            raise ValueError("weight_l2_penalty must be >= 0")
        if self.base_deviation_penalty < 0.0:
            raise ValueError("base_deviation_penalty must be >= 0")
        if self.smoothness_penalty < 0.0:
            raise ValueError("smoothness_penalty must be >= 0")
        if self.clip_observation <= 0.0:
            raise ValueError("clip_observation must be > 0")


class HighLevelEnvCore(gym.Env):
    """
    Daily high-level portfolio allocator.

    The factor pool is fixed during an episode. The agent only decides a new
    daily weight vector for the factors. Reward is computed from the resulting
    mega-alpha cross-sectional performance on the current day.
    """

    _SUMMARY_ROWS = 5

    def __init__(
        self,
        exprs: Sequence[Expression],
        calculator: TensorAlphaCalculator,
        base_weights: Optional[Sequence[float]] = None,
        config: Optional[HighLevelEnvConfig] = None,
    ) -> None:
        super().__init__()

        if len(exprs) == 0:
            raise ValueError("HighLevelEnvCore requires at least one expression")

        self._exprs: List[Expression] = list(exprs)
        self._factor_names = [str(expr) for expr in self._exprs]
        self._calculator = calculator
        self._config = config or HighLevelEnvConfig()
        self._device = calculator.target.device

        self._factor_values = torch.stack(
            [
                torch.nan_to_num(calculator.evaluate_alpha(expr), nan=0.0, posinf=0.0, neginf=0.0)
                for expr in self._exprs
            ],
            dim=0,
        )
        self._target = torch.nan_to_num(calculator.target, nan=0.0, posinf=0.0, neginf=0.0)
        self._n_factors = len(self._exprs)
        self._n_days = calculator.n_days
        if self._n_days <= self._config.lookback_days:
            raise ValueError(
                f"Not enough days ({self._n_days}) for lookback_days={self._config.lookback_days}"
            )

        self._daily_factor_ic = torch.stack(
            [batch_pearsonr(self._factor_values[i], self._target) for i in range(self._n_factors)],
            dim=1,
        )
        self._daily_factor_rank_ic = torch.stack(
            [batch_spearmanr(self._factor_values[i], self._target) for i in range(self._n_factors)],
            dim=1,
        )
        self._base_weights = self._prepare_base_weights(base_weights)
        self._action_indices = self._select_action_indices()
        self._action_index_tensor = torch.as_tensor(
            self._action_indices,
            dtype=torch.long,
            device=self._device,
        )
        self._n_action_factors = len(self._action_indices)

        obs_rows = self._config.lookback_days + self._SUMMARY_ROWS
        clip = self._config.clip_observation
        self.observation_space = gym.spaces.Box(
            low=-clip,
            high=clip,
            shape=(obs_rows, self._n_factors),
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._n_action_factors,),
            dtype=np.float32,
        )

        self._start_day = 0
        self._end_day = 0
        self._day = 0
        self._episode_step = 0
        self._episode_day_count = 0
        self._prev_weights = self._base_weights.clone()
        self._current_weights = self._base_weights.clone()
        self._reward_sum = 0.0
        self._ic_sum = 0.0
        self._rank_ic_sum = 0.0
        self._base_deviation_sum = 0.0
        self._smoothness_sum = 0.0
        self._last_episode_summary: Dict[str, float] = {}

    @property
    def n_factors(self) -> int:
        return self._n_factors

    @property
    def factor_names(self) -> List[str]:
        return list(self._factor_names)

    @property
    def base_weights(self) -> np.ndarray:
        return self._base_weights.detach().cpu().numpy().copy()

    @property
    def n_action_factors(self) -> int:
        return self._n_action_factors

    @property
    def action_factor_indices(self) -> List[int]:
        return list(self._action_indices)

    @property
    def action_factor_names(self) -> List[str]:
        return [self._factor_names[idx] for idx in self._action_indices]

    @property
    def action_mode(self) -> str:
        return self._config.action_mode

    @property
    def rebalance_interval_days(self) -> int:
        return self._config.rebalance_interval_days

    @property
    def last_episode_summary(self) -> Dict[str, float]:
        return dict(self._last_episode_summary)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}

        episode_days_value = (
            options["episode_days"]
            if "episode_days" in options
            else self._config.episode_days
        )
        episode_days = self._resolve_episode_days(episode_days_value)
        random_start = bool(options.get("random_start", self._config.random_start))
        start_day_opt = options.get("start_day")
        self._start_day = self._resolve_start_day(start_day_opt, episode_days, random_start)
        self._end_day = min(self._n_days, self._start_day + episode_days)
        self._day = self._start_day
        self._episode_step = 0
        self._episode_day_count = 0
        self._prev_weights = (
            self._base_weights.clone()
            if self._config.init_prev_weights_with_base
            else torch.zeros_like(self._base_weights)
        )
        self._current_weights = self._base_weights.clone()
        self._reward_sum = 0.0
        self._ic_sum = 0.0
        self._rank_ic_sum = 0.0
        self._base_deviation_sum = 0.0
        self._smoothness_sum = 0.0
        self._last_episode_summary = {}

        info = {
            "start_day": self._start_day,
            "end_day": self._end_day,
            "day_index": self._day,
        }
        return self._build_observation(), info

    def step(self, action: np.ndarray):
        if self._day >= self._end_day:
            raise RuntimeError("Episode already ended. Call reset() before step().")

        day_index = self._day
        weights = self._action_to_weights(action)
        smoothness_l2 = float((weights - self._prev_weights).pow(2).sum().item())
        hold_days = min(self._config.rebalance_interval_days, self._end_day - self._day)
        weight_l2 = float(weights.pow(2).mean().item())
        base_deviation_l2 = float((weights - self._base_weights).pow(2).sum().item())
        day_indices: List[int] = []
        day_ics: List[float] = []
        day_rank_ics: List[float] = []
        day_primary_rewards: List[float] = []
        day_rewards: List[float] = []
        day_smoothness_l2s: List[float] = []
        day_action_applied: List[bool] = []

        for offset in range(hold_days):
            current_day = self._day + offset
            mega_alpha = torch.nan_to_num(
                (weights[:, None] * self._factor_values[:, current_day, :]).sum(dim=0)
            )
            day_target = self._target[current_day]

            ic = float(batch_pearsonr(mega_alpha[None, :], day_target[None, :])[0].item())
            rank_ic = float(batch_spearmanr(mega_alpha[None, :], day_target[None, :])[0].item())
            primary_reward = self._combine_reward(ic, rank_ic)
            smoothness_term = smoothness_l2 if offset == 0 else 0.0
            daily_reward = (
                primary_reward
                - self._config.weight_l2_penalty * weight_l2
                - self._config.base_deviation_penalty * base_deviation_l2
                - self._config.smoothness_penalty * smoothness_term
            )

            day_indices.append(current_day)
            day_ics.append(ic)
            day_rank_ics.append(rank_ic)
            day_primary_rewards.append(primary_reward)
            day_rewards.append(daily_reward)
            day_smoothness_l2s.append(smoothness_term)
            day_action_applied.append(offset == 0)

        reward = float(sum(day_rewards) / hold_days)
        primary_reward = float(sum(day_primary_rewards) / hold_days)
        ic = float(sum(day_ics) / hold_days)
        rank_ic = float(sum(day_rank_ics) / hold_days)

        self._reward_sum += float(sum(day_rewards))
        self._ic_sum += float(sum(day_ics))
        self._rank_ic_sum += float(sum(day_rank_ics))
        self._base_deviation_sum += base_deviation_l2 * hold_days
        self._smoothness_sum += smoothness_l2
        self._episode_day_count += hold_days
        self._prev_weights = weights
        self._current_weights = weights
        self._day += 1
        self._day += hold_days - 1
        self._episode_step += 1

        terminated = self._day >= self._end_day
        info: Dict[str, Any] = {
            "day_index": day_index,
            "day_indices": list(day_indices),
            "weights": weights.detach().cpu().numpy().copy(),
            "daily_weights": [weights.detach().cpu().numpy().copy() for _ in range(hold_days)],
            "ic": ic,
            "daily_ic": list(day_ics),
            "rank_ic": rank_ic,
            "daily_rank_ic": list(day_rank_ics),
            "primary_reward": primary_reward,
            "daily_primary_reward": list(day_primary_rewards),
            "weight_l2": weight_l2,
            "daily_weight_l2": [weight_l2 for _ in range(hold_days)],
            "base_deviation_l2": base_deviation_l2,
            "daily_base_deviation_l2": [base_deviation_l2 for _ in range(hold_days)],
            "smoothness_l2": smoothness_l2,
            "daily_smoothness_l2": list(day_smoothness_l2s),
            "action_applied": True,
            "daily_action_applied": list(day_action_applied),
            "hold_days": hold_days,
            "reward": reward,
            "daily_reward": list(day_rewards),
        }
        if terminated:
            summary = self._episode_summary()
            info.update(summary)
            self._last_episode_summary = summary
            obs = self._terminal_observation()
        else:
            obs = self._build_observation()
        return obs, reward, terminated, False, info

    def render(self):
        return None

    def _prepare_base_weights(self, base_weights: Optional[Sequence[float]]) -> Tensor:
        if base_weights is None:
            weights = torch.full(
                (self._n_factors,),
                fill_value=1.0 / self._n_factors,
                dtype=torch.float32,
                device=self._device,
            )
        else:
            weights = torch.as_tensor(base_weights, dtype=torch.float32, device=self._device)
            if weights.numel() != self._n_factors:
                raise ValueError(
                    f"base_weights length {weights.numel()} does not match factor count {self._n_factors}"
                )
        return self._project_to_leverage(weights)

    def _select_action_indices(self) -> List[int]:
        action_top_k = self._config.action_top_k
        if action_top_k is None or action_top_k >= self._n_factors:
            return list(range(self._n_factors))
        ranked = sorted(
            range(self._n_factors),
            key=lambda idx: (-abs(float(self._base_weights[idx].item())), idx),
        )
        return ranked[:action_top_k]

    def _resolve_episode_days(self, episode_days_opt: Optional[int]) -> int:
        remaining_days = self._n_days - self._config.lookback_days
        episode_days = remaining_days if episode_days_opt is None else int(episode_days_opt)
        return max(1, min(episode_days, remaining_days))

    def _resolve_start_day(
        self,
        start_day_opt: Optional[int],
        episode_days: int,
        random_start: bool,
    ) -> int:
        min_start = self._config.lookback_days
        max_start = max(min_start, self._n_days - episode_days)
        if start_day_opt is not None:
            return int(max(min_start, min(int(start_day_opt), max_start)))
        if random_start and max_start > min_start:
            return int(self.np_random.integers(min_start, max_start + 1))
        return min_start

    def _combine_reward(self, ic: float, rank_ic: float) -> float:
        if self._config.reward_mode == "ic":
            return ic
        if self._config.reward_mode == "rank_ic":
            return rank_ic
        return 0.5 * (ic + rank_ic)

    def _action_to_weights(self, action: np.ndarray) -> Tensor:
        raw_action = torch.as_tensor(action, dtype=torch.float32, device=self._device)
        raw_action = torch.clamp(raw_action.flatten(), -1.0, 1.0)
        if raw_action.numel() != self._n_action_factors:
            raise ValueError(
                f"Action length {raw_action.numel()} does not match "
                f"action factor count {self._n_action_factors}"
            )
        full_action = torch.zeros(self._n_factors, dtype=torch.float32, device=self._device)
        full_action[self._action_index_tensor] = raw_action
        if self._config.action_mode == "absolute":
            weights = self._config.action_scale * full_action
        elif self._config.action_mode == "residual":
            weights = self._base_weights + self._config.action_scale * full_action
        else:
            gates = torch.ones(self._n_factors, dtype=torch.float32, device=self._device)
            gates[self._action_index_tensor] = torch.exp(self._config.action_scale * raw_action)
            weights = self._base_weights * gates
        return self._project_to_leverage(weights)

    def _project_to_leverage(self, weights: Tensor) -> Tensor:
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        leverage = float(weights.abs().sum().item())
        if leverage <= self._config.max_leverage or leverage <= 1e-12:
            return weights
        return weights * (self._config.max_leverage / leverage)

    def _build_observation(self) -> np.ndarray:
        lookback = self._config.lookback_days
        realized_end = max(0, self._day - self._config.target_horizon_days)
        start = max(0, realized_end - lookback)
        window = self._daily_factor_ic[start:realized_end]
        rank_window = self._daily_factor_rank_ic[start:realized_end]
        padded = torch.zeros((lookback, self._n_factors), dtype=torch.float32, device=self._device)
        if window.shape[0] > 0:
            padded[-window.shape[0]:] = window

        if window.shape[0] > 0:
            mean_row = window.mean(dim=0)
            rank_mean_row = rank_window.mean(dim=0)
        else:
            mean_row = torch.zeros(self._n_factors, dtype=torch.float32, device=self._device)
            rank_mean_row = torch.zeros(self._n_factors, dtype=torch.float32, device=self._device)
        realized_history = self._daily_factor_ic[:realized_end]
        if realized_history.shape[0] > 0:
            realized_mean_row = realized_history.mean(dim=0)
        else:
            realized_mean_row = torch.zeros(self._n_factors, dtype=torch.float32, device=self._device)

        obs = torch.cat(
            (
                padded,
                mean_row[None, :],
                rank_mean_row[None, :],
                self._current_weights[None, :],
                self._base_weights[None, :],
                realized_mean_row[None, :],
            ),
            dim=0,
        )
        clip = self._config.clip_observation
        obs = torch.clamp(obs, -clip, clip)
        return obs.detach().cpu().numpy().astype(np.float32, copy=False)

    def _terminal_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _episode_summary(self) -> Dict[str, float]:
        days = max(1, self._episode_day_count)
        return {
            "episode_mean_reward": self._reward_sum / days,
            "episode_mean_ic": self._ic_sum / days,
            "episode_mean_rank_ic": self._rank_ic_sum / days,
            "episode_mean_base_deviation_l2": self._base_deviation_sum / days,
            "episode_mean_smoothness_l2": self._smoothness_sum / days,
            "episode_length": float(self._episode_step),
            "episode_days": float(self._episode_day_count),
            "start_day": float(self._start_day),
            "end_day": float(self._end_day),
        }


HighLevelAllocationEnv = HighLevelEnvCore
