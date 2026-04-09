from dataclasses import dataclass
import math
from typing import Dict, Optional

import torch

from AlphaHRL.HRL.data.expression import Expression, OutOfDataRangeError
from AlphaHRL.HRL.models.alpha_pool import AlphaPoolBase
from AlphaHRL.HRL.utils.pytorch_utils import normalize_by_day
from AlphaHRL.alphahrl_qlib.stock_data import StockData


@dataclass(frozen=True)
class SubtreeProxyRewardConfig:
    enabled: bool = False
    proxy_days: int = 64
    validity_reward: float = 0.001
    stability_reward: float = 0.002
    stable_ratio_threshold: float = 0.95
    per_day_finite_threshold: float = 0.95
    min_valid_count_per_day: int = 8
    min_std_ratio: float = 1e-3
    max_spike_ratio: float = 1e3
    max_abs_value: float = 1e12


@dataclass(frozen=True)
class SubtreeProxyRewardResult:
    total_reward: float = 0.0
    validity_reward: float = 0.0
    stability_reward: float = 0.0
    finite_ratio: float = 0.0
    stability_ratio: float = 0.0
    valid_days: int = 0
    stable_days: int = 0
    total_days: int = 0
    evaluated: bool = False


class SubtreeProxyRewarder:
    def __init__(
        self,
        pool: AlphaPoolBase,
        config: Optional[SubtreeProxyRewardConfig] = None
    ) -> None:
        self._config = config or SubtreeProxyRewardConfig()
        self._cache: Dict[str, SubtreeProxyRewardResult] = {}
        self._proxy_data = self._build_proxy_data(pool)

    def evaluate(self, expr: Expression) -> SubtreeProxyRewardResult:
        if not self._config.enabled or self._proxy_data is None or not expr.is_featured:
            return SubtreeProxyRewardResult()

        expr_key = str(expr)
        if expr_key in self._cache:
            return self._cache[expr_key]

        try:
            with torch.no_grad():
                raw_value = expr.evaluate(self._proxy_data)
                # Match the task-side calculator, which evaluates alphas after
                # day-wise normalization instead of on raw expression values.
                value = normalize_by_day(raw_value)
        except (OutOfDataRangeError, RuntimeError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            result = SubtreeProxyRewardResult()
            self._cache[expr_key] = result
            return result

        if raw_value.ndim != 2 or raw_value.numel() == 0 or value.ndim != 2 or value.numel() == 0:
            result = SubtreeProxyRewardResult()
            self._cache[expr_key] = result
            return result

        finite_ratio = torch.isfinite(raw_value).float().mean().item()
        total_days = int(raw_value.shape[0])
        valid_day_mask = [self._is_valid_day(raw_value[day_idx]) for day_idx in range(total_days)]
        valid_days = sum(valid_day_mask)

        validity_ratio = valid_days / total_days if total_days != 0 else 0.0
        validity_reward = self._config.validity_reward * validity_ratio

        stable_days = 0
        if valid_days == 0:
            stability_ratio = 0.0
            stability_reward = 0.0
        else:
            for day_idx, is_valid_day in enumerate(valid_day_mask):
                if not is_valid_day:
                    continue
                if self._is_stable_day(value[day_idx]):
                    stable_days += 1

            stability_ratio = stable_days / valid_days
            stability_reward = self._score_stability(stability_ratio)
        result = SubtreeProxyRewardResult(
            total_reward=validity_reward + stability_reward,
            validity_reward=validity_reward,
            stability_reward=stability_reward,
            finite_ratio=finite_ratio,
            stability_ratio=stability_ratio,
            valid_days=valid_days,
            stable_days=stable_days,
            total_days=total_days,
            evaluated=True
        )
        self._cache[expr_key] = result
        return result

    def _build_proxy_data(self, pool: AlphaPoolBase) -> Optional[StockData]:
        if not self._config.enabled:
            return None

        calculator = getattr(pool, "calculator", None)
        data = getattr(calculator, "data", None)
        if not isinstance(data, StockData):
            return None

        proxy_days = min(max(1, self._config.proxy_days), data.n_days)
        if proxy_days >= data.n_days:
            return data
        return data[:proxy_days]

    def _is_valid_day(self, day_value: torch.Tensor) -> bool:
        cfg = self._config
        finite_mask = torch.isfinite(day_value)
        total_count = int(day_value.numel())
        finite_count = int(finite_mask.sum().item())
        if total_count == 0 or finite_count < max(2, cfg.min_valid_count_per_day):
            return False

        finite_ratio = finite_count / total_count
        if finite_ratio < cfg.per_day_finite_threshold:
            return False

        finite_values = day_value[finite_mask]
        abs_values = finite_values.abs()
        max_abs = float(abs_values.max().item())
        if not math.isfinite(max_abs) or max_abs > cfg.max_abs_value:
            return False

        return True

    def _is_stable_day(self, day_value: torch.Tensor) -> bool:
        cfg = self._config
        finite_mask = torch.isfinite(day_value)
        finite_count = int(finite_mask.sum().item())
        if finite_count < 2:
            return False

        # Stability only focuses on distributional shape after day-wise normalization.
        finite_values = day_value[finite_mask]
        abs_values = finite_values.abs()

        mean_abs = float(abs_values.mean().item())
        median_abs = float(abs_values.median().item())
        robust_scale = max(mean_abs, median_abs, 1e-6)
        if finite_values.numel() == 1:
            q95_abs = float(abs_values.max().item())
        else:
            q95_abs = float(torch.quantile(abs_values, 0.95).item())
        spike_ratio = q95_abs / robust_scale
        if spike_ratio > cfg.max_spike_ratio:
            return False

        std = float(finite_values.std(unbiased=False).item())
        std_ratio = std / (mean_abs + 1e-6)
        if std_ratio < cfg.min_std_ratio:
            return False

        return True

    def _score_stability(self, stability_ratio: float) -> float:
        threshold = self._config.stable_ratio_threshold
        scale = self._config.stability_reward
        if threshold <= 0.0:
            return scale
        if stability_ratio >= threshold:
            return scale * (stability_ratio - threshold) / max(1e-6, 1.0 - threshold)
        return -scale * (threshold - stability_ratio) / threshold
