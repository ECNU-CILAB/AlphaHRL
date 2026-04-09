import json
import math
import os
from pathlib import Path
import re
import shutil
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import fire
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger as sb3_logger

from AlphaHRL.HRL.data.expression import Expression, Feature, Greater, Less, Operators, Ref, Sub
from AlphaHRL.HRL.data.parser import ExpressionParser
from AlphaHRL.HRL.rl.env.high_level_core import HighLevelEnvConfig, HighLevelEnvCore
from AlphaHRL.HRL.rl.env.high_level_wrapper import HighLevelEnv, HighLevelEnvWrapper
from AlphaHRL.HRL.rl.high_level_policy import HighLevelSequenceNet
from AlphaHRL.HRL.utils import reseed_everything
from AlphaHRL.HRL.utils.logging import (
    format_block,
    format_high_level_iteration_summary,
    format_kv_pairs,
    format_scalar,
    shorten_middle,
)
from AlphaHRL.alphahrl_qlib.calculator import QLibStockDataCalculator
from AlphaHRL.alphahrl_qlib.stock_data import FeatureType, StockData, initialize_qlib

_VALID_POOL_SELECTION_METRICS = {"ic", "icir", "rank_ic", "rank_icir"}
_POOL_CHECKPOINT_RE = re.compile(r"(?P<step>\d+)_steps_pool\.json$")
_BEST_MODEL_SELECTION_METRIC = "train_eval.mean_ic"
_HIGH_LEVEL_SB3_TAG_ORDER = {
    "env/": 0,
    "train_eval/": 1,
    "test_1/": 2,
    "test_2/": 3,
    "test_3/": 4,
    "test_mean/": 5,
    "time/": 6,
}
_HIGH_LEVEL_SB3_METRIC_ORDER = {
    "n_factors": 0,
    "n_action_factors": 1,
    "rebalance_interval_days": 2,
    "mean_ic": 0,
    "mean_rank_ic": 1,
    "icir": 2,
    "rank_icir": 3,
    "length": 4,
    "fps": 0,
    "iterations": 1,
    "time_elapsed": 2,
    "total_timesteps": 3,
}
_SB3_HUMAN_OUTPUT_FOR_HIGH_LEVEL_PATCHED = False


def _split_sb3_log_key(key: str) -> Tuple[str, str]:
    if "/" not in key:
        return "", key
    tag, metric = key.split("/", 1)
    return f"{tag}/", metric


def _high_level_sb3_sort_key(key: str) -> Tuple[int, int, str, str]:
    tag, metric = _split_sb3_log_key(key)
    return (
        _HIGH_LEVEL_SB3_TAG_ORDER.get(tag, 1000),
        _HIGH_LEVEL_SB3_METRIC_ORDER.get(metric, 1000),
        tag,
        metric,
    )


def patch_sb3_human_output_for_high_level() -> None:
    global _SB3_HUMAN_OUTPUT_FOR_HIGH_LEVEL_PATCHED
    if _SB3_HUMAN_OUTPUT_FOR_HIGH_LEVEL_PATCHED:
        return
    _SB3_HUMAN_OUTPUT_FOR_HIGH_LEVEL_PATCHED = True

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Tuple[str, ...]],
        step: int = 0,
    ) -> None:
        key2str: Dict[Tuple[str, str], str] = {}
        for raw_key in sorted(key_values.keys(), key=_high_level_sb3_sort_key):
            value = key_values[raw_key]
            excluded = key_excluded.get(raw_key)
            if excluded is not None and ("stdout" in excluded or "log" in excluded):
                continue

            if isinstance(value, sb3_logger.Video):
                raise sb3_logger.FormatUnsupportedError(["stdout", "log"], "video")
            if isinstance(value, sb3_logger.Figure):
                raise sb3_logger.FormatUnsupportedError(["stdout", "log"], "figure")
            if isinstance(value, sb3_logger.Image):
                raise sb3_logger.FormatUnsupportedError(["stdout", "log"], "image")
            if isinstance(value, sb3_logger.HParam):
                raise sb3_logger.FormatUnsupportedError(["stdout", "log"], "hparam")

            if isinstance(value, float):
                value_str = f"{value:<8.3g}"
            else:
                value_str = str(value)

            tag = ""
            key = raw_key
            if "/" in raw_key:
                tag, metric = _split_sb3_log_key(raw_key)
                key2str[(tag, self._truncate(tag))] = ""
                key = f"{'':3}{metric}"

            truncated_key = self._truncate(key)
            if (tag, truncated_key) in key2str:
                raise ValueError(
                    f"Key '{key}' truncated to '{truncated_key}' that already exists. "
                    "Consider increasing `max_length`."
                )
            key2str[(tag, truncated_key)] = self._truncate(value_str)

        if len(key2str) == 0:
            sb3_logger.warnings.warn("Tried to write empty key-value dict")
            return

        tagless_keys = map(lambda item: item[1], key2str.keys())
        key_width = max(map(len, tagless_keys))
        val_width = max(map(len, key2str.values()))

        dashes = "-" * (key_width + val_width + 7)
        lines = [dashes]
        for (_, key), value in key2str.items():
            key_space = " " * (key_width - len(key))
            val_space = " " * (val_width - len(value))
            lines.append(f"| {key}{key_space} | {value}{val_space} |")
        lines.append(dashes)

        if sb3_logger.tqdm is not None and hasattr(self.file, "name") and self.file.name == "<stdout>":
            sb3_logger.tqdm.write("\n".join(lines) + "\n", file=sys.stdout, end="")
        else:
            self.file.write("\n".join(lines) + "\n")
        self.file.flush()

    sb3_logger.HumanOutputFormat.write = write


def normalize_target_horizon_days(target_horizon_days: int) -> int:
    horizon = int(target_horizon_days)
    if horizon < 1:
        raise ValueError("target_horizon_days must be >= 1")
    return horizon


def build_forward_return_target(base_expr: Expression, target_horizon_days: int) -> Expression:
    horizon = normalize_target_horizon_days(target_horizon_days)
    return Ref(base_expr, -horizon) / base_expr - 1


def resolve_dataset_future_days(
    target_horizon_days: int,
    minimum_future_days: int = 30,
) -> int:
    horizon = normalize_target_horizon_days(target_horizon_days)
    return max(int(minimum_future_days), horizon)


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub],
        },
    )


def load_pool_state(pool_path: str) -> Tuple[List[Expression], Optional[List[float]]]:
    with open(pool_path, encoding="utf-8") as f:
        raw = json.load(f)

    expr_strings = raw.get("exprs")
    if not isinstance(expr_strings, list) or len(expr_strings) == 0:
        raise ValueError(f"Pool file {pool_path} does not contain a non-empty 'exprs' list")

    parser = build_parser()
    exprs = [parser.parse(expr) for expr in expr_strings]
    weights = raw.get("weights")
    if weights is None:
        return exprs, None
    if not isinstance(weights, list):
        raise ValueError(f"Pool file {pool_path} contains a non-list 'weights' field")
    if len(weights) != len(exprs):
        raise ValueError(
            f"Pool file {pool_path} has {len(weights)} weights for {len(exprs)} expressions"
        )
    return exprs, [float(weight) for weight in weights]


def save_pool_state(
    save_path: str,
    exprs: Sequence[Expression],
    weights: Optional[Sequence[float]],
) -> None:
    payload = {
        "exprs": [str(expr) for expr in exprs],
        "weights": None if weights is None else [float(weight) for weight in weights],
    }
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def default_pool_weights(exprs: Sequence[Expression], weights: Optional[Sequence[float]]) -> List[float]:
    if weights is not None:
        return [float(weight) for weight in weights]
    if len(exprs) == 0:
        raise ValueError("Cannot build default weights for an empty expression list")
    fill = 1.0 / len(exprs)
    return [fill for _ in exprs]


def score_pool_state(
    exprs: Sequence[Expression],
    weights: Optional[Sequence[float]],
    calculator: QLibStockDataCalculator,
) -> Dict[str, float]:
    resolved_weights = default_pool_weights(exprs, weights)
    try:
        ic, icir, rank_ic, rank_icir = calculator.calc_pool_all_ret_with_ir(exprs, resolved_weights)
    except ZeroDivisionError:
        ic, rank_ic = calculator.calc_pool_all_ret(exprs, resolved_weights)
        icir = float("inf") if ic > 0.0 else float("-inf") if ic < 0.0 else 0.0
        rank_icir = (
            float("inf") if rank_ic > 0.0 else
            float("-inf") if rank_ic < 0.0 else
            0.0
        )
    return {
        "ic": float(ic),
        "icir": float(icir),
        "rank_ic": float(rank_ic),
        "rank_icir": float(rank_icir),
    }


def list_low_level_pool_paths(low_level_result_dir: str) -> List[Tuple[int, str]]:
    result_dir = Path(low_level_result_dir)
    if not result_dir.is_dir():
        raise ValueError(f"low_level_result_dir does not exist or is not a directory: {low_level_result_dir}")

    candidates: List[Tuple[int, str]] = []
    for path in result_dir.iterdir():
        match = _POOL_CHECKPOINT_RE.fullmatch(path.name)
        if match is None:
            continue
        candidates.append((int(match.group("step")), str(path)))
    candidates.sort()
    if len(candidates) == 0:
        raise ValueError(f"No *_steps_pool.json files found under {low_level_result_dir}")
    return candidates


def select_best_pool_from_low_level_dir(
    low_level_result_dir: str,
    calculator: QLibStockDataCalculator,
    selection_metric: str = "ic",
) -> Tuple[List[Expression], List[float], Dict[str, Any]]:
    if selection_metric not in _VALID_POOL_SELECTION_METRICS:
        raise ValueError(
            f"Invalid pool_selection_metric: {selection_metric}. "
            f"Expected one of {sorted(_VALID_POOL_SELECTION_METRICS)}."
        )

    best_exprs: Optional[List[Expression]] = None
    best_weights: Optional[List[float]] = None
    best_metadata: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, int]] = None

    candidates = list_low_level_pool_paths(low_level_result_dir)
    for step, pool_path in candidates:
        exprs, weights = load_pool_state(pool_path)
        resolved_weights = default_pool_weights(exprs, weights)
        metrics = score_pool_state(exprs, resolved_weights, calculator)
        metric_value = metrics[selection_metric]
        candidate_key = (metric_value, step)
        if best_key is None or candidate_key > best_key:
            best_key = candidate_key
            best_exprs = exprs
            best_weights = resolved_weights
            best_metadata = {
                "source": "low_level_result_dir",
                "low_level_result_dir": low_level_result_dir,
                "best_pool_path": pool_path,
                "best_step": int(step),
                "selection_metric": selection_metric,
                "selection_score": float(metric_value),
                "candidate_count": len(candidates),
                "selected_metrics": metrics,
            }

    assert best_exprs is not None and best_weights is not None and best_metadata is not None
    return best_exprs, best_weights, best_metadata


def load_best_pool_from_low_level_dir(
    low_level_result_dir: str,
    calculator: QLibStockDataCalculator,
) -> Tuple[List[Expression], List[float], Dict[str, Any]]:
    result_dir = Path(low_level_result_dir)
    if not result_dir.is_dir():
        raise ValueError(f"low_level_result_dir does not exist or is not a directory: {low_level_result_dir}")

    best_pool_path = result_dir / "best_pool.json"
    if not best_pool_path.is_file():
        raise ValueError(f"best_pool.json not found under {low_level_result_dir}")

    exprs, weights = load_pool_state(str(best_pool_path))
    resolved_weights = default_pool_weights(exprs, weights)
    metrics = score_pool_state(exprs, resolved_weights, calculator)
    metadata: Dict[str, Any] = {
        "source": "low_level_best_pool",
        "low_level_result_dir": low_level_result_dir,
        "best_pool_path": str(best_pool_path),
        "pool_size": len(exprs),
        "selected_metrics": metrics,
    }

    with open(best_pool_path, encoding="utf-8") as f:
        raw_pool = json.load(f)
    if not isinstance(raw_pool, dict):
        raise ValueError(f"Pool file {best_pool_path} does not contain a JSON object")
    for key in ("best_ic_ret", "best_obj", "ic_mean", "rank_ic_mean", "rank_ic", "rank_icir", "icir"):
        value = raw_pool.get(key)
        if isinstance(value, (int, float)):
            metadata[key] = float(value)
    for key in ("iteration", "snapshot_version", "num_timesteps", "pool_size"):
        value = raw_pool.get(key)
        if isinstance(value, int):
            metadata[key] = int(value)

    return exprs, resolved_weights, metadata


def resolve_factor_pool(
    calculator: QLibStockDataCalculator,
    pool_path: Optional[str],
    low_level_result_dir: Optional[str],
    pool_selection_metric: str,
) -> Tuple[List[Expression], Optional[List[float]], Dict[str, Any]]:
    if pool_path is not None:
        exprs, weights = load_pool_state(pool_path)
        resolved_weights = default_pool_weights(exprs, weights)
        return exprs, resolved_weights, {
            "source": "pool_json",
            "pool_path": pool_path,
            "pool_size": len(exprs),
        }
    if low_level_result_dir is not None:
        best_pool_path = Path(low_level_result_dir) / "best_pool.json"
        if best_pool_path.is_file():
            return load_best_pool_from_low_level_dir(
                low_level_result_dir=low_level_result_dir,
                calculator=calculator,
            )
        return select_best_pool_from_low_level_dir(
            low_level_result_dir=low_level_result_dir,
            calculator=calculator,
            selection_metric=pool_selection_metric,
        )
    raise ValueError("Either pool_path or low_level_result_dir must be provided.")


def _compute_mean_and_ir(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    tensor = torch.as_tensor(values, dtype=torch.float32)
    mean = float(tensor.mean().item())
    if tensor.numel() <= 1:
        return mean, 0.0
    std = float(tensor.std(unbiased=True).item())
    if not math.isfinite(std) or std <= 1e-12:
        if mean > 0.0:
            return mean, float("inf")
        if mean < 0.0:
            return mean, float("-inf")
        return mean, 0.0
    return mean, mean / std


def _extract_eval_date(env: HighLevelEnvWrapper, day_index: int) -> Optional[str]:
    core = env.unwrapped
    calculator = getattr(core, "_calculator", None)
    data = getattr(calculator, "data", None)
    dates = getattr(data, "_dates", None)
    max_backtrack_days = getattr(data, "max_backtrack_days", None)
    if dates is None or max_backtrack_days is None:
        return None
    raw_index = day_index + int(max_backtrack_days)
    if raw_index < 0 or raw_index >= len(dates):
        return None
    date_value = dates[raw_index]
    if hasattr(date_value, "strftime"):
        return date_value.strftime("%Y-%m-%d")
    return str(date_value)


def summarize_test_metrics(
    metrics_by_env: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    test_metrics = [metrics for name, metrics in metrics_by_env.items() if name.startswith("test_")]
    if len(test_metrics) == 0:
        return {}
    keys = [
        "mean_ic",
        "mean_rank_ic",
        "icir",
        "rank_icir",
    ]
    return {
        key: float(sum(float(metrics[key]) for metrics in test_metrics) / len(test_metrics))
        for key in keys
        if all(key in metrics for metrics in test_metrics)
    }


def select_best_model_score(metrics_by_env: Dict[str, Dict[str, float]]) -> Optional[float]:
    train_eval_metrics = metrics_by_env.get("train_eval")
    if train_eval_metrics is None:
        return None
    value = train_eval_metrics.get("mean_ic")
    if value is None:
        return None
    return float(value)


def evaluate_model(model: PPO, env: HighLevelEnvWrapper) -> Dict[str, Any]:
    obs, _ = env.reset(options={"random_start": False, "episode_days": None})
    done = False
    terminal_info: Dict[str, Any] = {}
    day_indices: List[int] = []
    dates: List[str] = []
    weights_by_day: List[List[float]] = []
    rewards: List[float] = []
    primary_rewards: List[float] = []
    weight_l2s: List[float] = []
    base_deviation_l2s: List[float] = []
    smoothness_l2s: List[float] = []
    action_applied_flags: List[bool] = []
    ics: List[float] = []
    rank_ics: List[float] = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, step_info = env.step(action)
        step_day_indices = step_info.get("day_indices")
        if isinstance(step_day_indices, list) and len(step_day_indices) > 0:
            expanded_day_indices = [int(day) for day in step_day_indices]
            expanded_weights = [
                [float(weight) for weight in weights]
                for weights in step_info.get("daily_weights", [])
            ]
            expanded_rewards = [float(value) for value in step_info.get("daily_reward", [])]
            expanded_primary_rewards = [
                float(value) for value in step_info.get("daily_primary_reward", [])
            ]
            expanded_weight_l2s = [float(value) for value in step_info.get("daily_weight_l2", [])]
            expanded_base_deviation_l2s = [
                float(value) for value in step_info.get("daily_base_deviation_l2", [])
            ]
            expanded_smoothness_l2s = [
                float(value) for value in step_info.get("daily_smoothness_l2", [])
            ]
            expanded_action_applied = [
                bool(value) for value in step_info.get("daily_action_applied", [])
            ]
            expanded_ics = [float(value) for value in step_info.get("daily_ic", [])]
            expanded_rank_ics = [float(value) for value in step_info.get("daily_rank_ic", [])]

            expected_len = len(expanded_day_indices)
            if not all(
                len(seq) == expected_len
                for seq in (
                    expanded_weights,
                    expanded_rewards,
                    expanded_primary_rewards,
                    expanded_weight_l2s,
                    expanded_base_deviation_l2s,
                    expanded_smoothness_l2s,
                    expanded_action_applied,
                    expanded_ics,
                    expanded_rank_ics,
                )
            ):
                raise ValueError("Inconsistent expanded daily trace lengths returned by env.step()")

            day_indices.extend(expanded_day_indices)
            for day_index in expanded_day_indices:
                date = _extract_eval_date(env, day_index)
                if date is not None:
                    dates.append(date)
            weights_by_day.extend(expanded_weights)
            rewards.extend(expanded_rewards)
            primary_rewards.extend(expanded_primary_rewards)
            weight_l2s.extend(expanded_weight_l2s)
            base_deviation_l2s.extend(expanded_base_deviation_l2s)
            smoothness_l2s.extend(expanded_smoothness_l2s)
            action_applied_flags.extend(expanded_action_applied)
            ics.extend(expanded_ics)
            rank_ics.extend(expanded_rank_ics)
        else:
            day_index = int(step_info["day_index"])
            day_indices.append(day_index)
            date = _extract_eval_date(env, day_index)
            if date is not None:
                dates.append(date)
            weights_by_day.append([float(weight) for weight in step_info["weights"].tolist()])
            rewards.append(float(step_info["reward"]))
            primary_rewards.append(float(step_info["primary_reward"]))
            weight_l2s.append(float(step_info["weight_l2"]))
            base_deviation_l2s.append(float(step_info["base_deviation_l2"]))
            smoothness_l2s.append(float(step_info["smoothness_l2"]))
            action_applied_flags.append(bool(step_info["action_applied"]))
            ics.append(float(step_info["ic"]))
            rank_ics.append(float(step_info["rank_ic"]))
        if done:
            terminal_info = step_info

    ic_mean, icir = _compute_mean_and_ir(ics)
    rank_ic_mean, rank_icir = _compute_mean_and_ir(rank_ics)
    summary: Dict[str, float] = {
        "mean_ic": ic_mean,
        "mean_rank_ic": rank_ic_mean,
        "icir": icir,
        "rank_icir": rank_icir,
        "length": float(len(day_indices)),
        "mean_base_deviation_l2": float(sum(base_deviation_l2s) / len(base_deviation_l2s)),
        "mean_smoothness_l2": float(sum(smoothness_l2s) / len(smoothness_l2s)),
    }
    for key in ("start_day", "end_day"):
        value = terminal_info.get(key)
        if isinstance(value, (int, float)):
            summary[key] = float(value)

    daily_trace: Dict[str, Any] = {
        "day_index": day_indices,
        "weights": weights_by_day,
        "reward": rewards,
        "primary_reward": primary_rewards,
        "weight_l2": weight_l2s,
        "base_deviation_l2": base_deviation_l2s,
        "smoothness_l2": smoothness_l2s,
        "action_applied": action_applied_flags,
        "ic": ics,
        "rank_ic": rank_ics,
    }
    if len(dates) == len(day_indices):
        daily_trace["date"] = dates

    return {
        "summary": summary,
        "daily_trace": daily_trace,
    }


def summarize_factor_pool(
    exprs: Sequence[Expression],
    base_weights: Sequence[float],
    limit: int = 5,
) -> str:
    ranked = sorted(
        enumerate(zip(exprs, base_weights)),
        key=lambda item: abs(float(item[1][1])),
        reverse=True,
    )
    lines = []
    for idx, (expr, weight) in ranked[:limit]:
        lines.append(
            f"#{idx} w={format_scalar(weight)} {shorten_middle(str(expr), max_length=84)}"
        )
    return "; ".join(lines) if lines else "<empty>"


def format_baseline_summary(
    metrics_by_env: Sequence[Tuple[str, Dict[str, float]]],
) -> str:
    lines = []
    for env_name, metrics in metrics_by_env:
        lines.append(
            f"{env_name:<8}: " + format_kv_pairs([
                ("mean_ic", metrics["ic"]),
                ("mean_rank_ic", metrics["rank_ic"]),
                ("icir", metrics["icir"]),
                ("rank_icir", metrics["rank_icir"]),
            ])
        )
    test_metrics = [metrics for env_name, metrics in metrics_by_env if env_name.startswith("test_")]
    if len(test_metrics) > 0:
        lines.append(
            f"{'test_mean':<8}: " + format_kv_pairs([
                ("mean_ic", sum(float(metrics["ic"]) for metrics in test_metrics) / len(test_metrics)),
                ("mean_rank_ic", sum(float(metrics["rank_ic"]) for metrics in test_metrics) / len(test_metrics)),
                ("icir", sum(float(metrics["icir"]) for metrics in test_metrics) / len(test_metrics)),
                ("rank_icir", sum(float(metrics["rank_icir"]) for metrics in test_metrics) / len(test_metrics)),
            ])
        )
    return format_block("[High-Level Baseline]", lines, width=96, border_char="-")


class HighLevelEvalCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        eval_envs: Dict[str, HighLevelEnvWrapper],
        reward_mode: str,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.eval_envs = eval_envs
        self.reward_mode = reward_mode
        os.makedirs(self.save_path, exist_ok=True)
        self._rollout_idx = 0
        self._best_model_score: Optional[float] = None

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        eval_results_by_env: Dict[str, Dict[str, Any]] = {}
        metrics_by_env: Dict[str, Dict[str, float]] = {}
        for name, env in self.eval_envs.items():
            eval_result = evaluate_model(self.model, env)  # type: ignore[arg-type]
            eval_results_by_env[name] = eval_result
            metrics = eval_result["summary"]
            metrics_by_env[name] = metrics
            for key in ("mean_ic", "mean_rank_ic", "icir", "rank_icir", "length"):
                if key in metrics:
                    self.logger.record(f"{name}/{key}", metrics[key])
        test_mean_metrics = summarize_test_metrics(metrics_by_env)
        for key in ("mean_ic", "mean_rank_ic", "icir", "rank_icir"):
            if key in test_mean_metrics:
                self.logger.record(f"test_mean/{key}", test_mean_metrics[key])
        self.logger.record("env/n_factors", self.env_core.n_factors)
        self.logger.record("env/n_action_factors", self.env_core.n_action_factors)
        self.logger.record("env/rebalance_interval_days", self.env_core.rebalance_interval_days)

        current_best_model_score = select_best_model_score(metrics_by_env)
        previous_best_model_score = self._best_model_score
        is_best_so_far = False
        if current_best_model_score is not None:
            if previous_best_model_score is None or current_best_model_score > previous_best_model_score:
                self._best_model_score = current_best_model_score
                is_best_so_far = True

        checkpoint_path, checkpoint_metadata = self.save_checkpoint(
            eval_results_by_env=eval_results_by_env,
            test_mean_metrics=test_mean_metrics,
            best_model_selection_score=current_best_model_score,
            best_model_score_so_far=self._best_model_score,
            is_best_so_far=is_best_so_far,
        )
        if is_best_so_far:
            self.save_best_model(checkpoint_path, checkpoint_metadata)
        self._rollout_idx += 1

        summary_rows = []
        for env_name in ("train_eval", "test_1", "test_2", "test_3"):
            metrics = metrics_by_env.get(env_name)
            if metrics is None:
                continue
            summary_rows.append((env_name, [
                ("mean_ic", metrics.get("mean_ic")),
                ("mean_rank_ic", metrics.get("mean_rank_ic")),
                ("icir", metrics.get("icir")),
                ("rank_icir", metrics.get("rank_icir")),
                ("length", int(metrics["length"]) if "length" in metrics else None),
            ]))
        if len(test_mean_metrics) > 0:
            summary_rows.append(("test_mean", [
                ("mean_ic", test_mean_metrics.get("mean_ic")),
                ("mean_rank_ic", test_mean_metrics.get("mean_rank_ic")),
                ("icir", test_mean_metrics.get("icir")),
                ("rank_icir", test_mean_metrics.get("rank_icir")),
            ]))

        print()
        print(format_high_level_iteration_summary(
            iteration=self._rollout_idx,
            total_timesteps=self.num_timesteps,
            n_factors=self.env_core.n_factors,
            n_action_factors=self.env_core.n_action_factors,
            rebalance_interval_days=self.env_core.rebalance_interval_days,
            action_mode=self.env_core.action_mode,
            reward_mode=self.reward_mode,
            metrics_by_env=summary_rows,
            checkpoint_path=checkpoint_path,
        ))

    def save_checkpoint(
        self,
        eval_results_by_env: Dict[str, Dict[str, Any]],
        test_mean_metrics: Dict[str, float],
        best_model_selection_score: Optional[float],
        best_model_score_so_far: Optional[float],
        is_best_so_far: bool,
    ) -> Tuple[str, Dict[str, Any]]:
        path = os.path.join(self.save_path, f"{self.num_timesteps}_steps")
        self.model.save(path)  # type: ignore[arg-type]
        metadata = {
            "iteration": int(self._rollout_idx + 1),
            "num_timesteps": int(self.num_timesteps),
            "factor_names": self.env_core.factor_names,
            "base_weights": self.env_core.base_weights.tolist(),
            "n_action_factors": self.env_core.n_action_factors,
            "action_factor_indices": self.env_core.action_factor_indices,
            "action_factor_names": self.env_core.action_factor_names,
            "rebalance_interval_days": self.env_core.rebalance_interval_days,
            "action_mode": self.env_core.action_mode,
            "best_model_selection_metric": _BEST_MODEL_SELECTION_METRIC,
            "best_model_selection_score": best_model_selection_score,
            "best_model_score_so_far": best_model_score_so_far,
            "is_best_so_far": is_best_so_far,
            "test_mean": test_mean_metrics,
            "eval_by_env": {
                env_name: {
                    "summary": eval_result["summary"],
                    "daily_trace": eval_result["daily_trace"],
                }
                for env_name, eval_result in eval_results_by_env.items()
            },
        }
        with open(f"{path}_meta.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return path, metadata

    def save_best_model(self, checkpoint_path: str, checkpoint_metadata: Dict[str, Any]) -> None:
        shutil.copyfile(f"{checkpoint_path}.zip", os.path.join(self.save_path, "best_model.zip"))
        best_metadata = dict(checkpoint_metadata)
        best_metadata["best_model_path"] = os.path.join(self.save_path, "best_model.zip")
        with open(os.path.join(self.save_path, "best_model_meta.json"), "w", encoding="utf-8") as f:
            json.dump(best_metadata, f, ensure_ascii=False, indent=2)

    @property
    def env_core(self) -> HighLevelEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore[return-value]


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_path: Optional[str] = None,
    low_level_result_dir: Optional[str] = None,
    pool_selection_metric: str = "ic",
    steps: int = 200_000,
    lookback_days: int = 20,
    episode_days: int = 252,
    target_horizon_days: int = 20,
    rebalance_interval_days: int = 1,
    max_leverage: float = 1.0,
    action_scale: float = 0.1,
    action_mode: str = "gate",
    action_top_k: Optional[int] = 8,
    reward_mode: str = "ic",
    weight_l2_penalty: float = 0.0,
    base_deviation_penalty: float = 1.0,
    smoothness_penalty: float = 0.5,
    learning_rate: float = 1e-4,
    n_steps: int = 512,
    batch_size: int = 128,
    ent_coef: float = 0.0,
    gamma: float = 0.99,
    policy_d_model: int = 128,
    policy_n_layers: int = 2,
    policy_dropout: float = 0.1,
    policy_log_std_init: float = -2.0,
    device_str: str = "cuda:4",
):
    patch_sb3_human_output_for_high_level()

    reseed_everything(seed)
    initialize_qlib("/home/liuyu/.qlib/AlphaGen_qlib_data/qlib_data/cn_data_rolling/")

    target_horizon_days = normalize_target_horizon_days(target_horizon_days)
    dataset_future_days = resolve_dataset_future_days(target_horizon_days)
    device = torch.device(device_str)
    close = Feature(FeatureType.CLOSE)
    target = build_forward_return_target(close, target_horizon_days)

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            max_future_days=dataset_future_days,
            device=device,
        )

    segments = [
        ("2012-01-01", "2021-12-31"),
        ("2022-01-01", "2022-06-30"),
        ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30"),
    ]
    datasets = [get_dataset(*segment) for segment in segments]
    calculators = [QLibStockDataCalculator(dataset, target) for dataset in datasets]

    exprs, base_weights, pool_metadata = resolve_factor_pool(
        calculator=calculators[0],
        pool_path=pool_path,
        low_level_result_dir=low_level_result_dir,
        pool_selection_metric=pool_selection_metric,
    )
    assert base_weights is not None

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    save_path = os.path.join(
        "./out/results/high_level",
        f"{instruments}_{len(exprs)}_{seed}_{timestamp}",
    )
    os.makedirs(save_path, exist_ok=True)
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    config_lines = [
        f"run     : {os.path.basename(save_path)}",
        f"start   : {start_time}",
        f"market  : {instruments}",
        f"seed    : {seed}",
        f"source  : {pool_metadata['source']}",
        f"pool    : path={pool_path or '<none>'} | low_level_dir={low_level_result_dir or '<none>'} | select={pool_selection_metric}",
        f"factors : count={len(exprs)} | top={summarize_factor_pool(exprs, base_weights, limit=5)}",
        f"env     : lookback={lookback_days} | episode_days={episode_days} | target_horizon={target_horizon_days} | rebalance={rebalance_interval_days} | future_days={dataset_future_days} | leverage={format_scalar(max_leverage)} | action_scale={format_scalar(action_scale)} | action_mode={action_mode} | action_top_k={action_top_k if action_top_k is not None else 'all'} | reward_mode={reward_mode}",
        f"train   : steps={steps} | lr={format_scalar(learning_rate)} | n_steps={n_steps} | batch={batch_size} | ent_coef={format_scalar(ent_coef)} | gamma={format_scalar(gamma)}",
        f"penalty : weight_l2={format_scalar(weight_l2_penalty)} | base_dev={format_scalar(base_deviation_penalty)} | smooth={format_scalar(smoothness_penalty)}",
        f"policy  : d_model={policy_d_model} | layers={policy_n_layers} | dropout={format_scalar(policy_dropout)} | log_std_init={format_scalar(policy_log_std_init)} | device={device}",
        f"output  : {save_path}",
    ]
    if pool_metadata["source"] == "low_level_best_pool":
        selected_metrics = pool_metadata["selected_metrics"]
        chosen_items = [("mode", "best_pool")]
        if "best_ic_ret" in pool_metadata:
            chosen_items.append(("best_ic_ret", pool_metadata["best_ic_ret"]))
        if "best_obj" in pool_metadata:
            chosen_items.append(("best_obj", pool_metadata["best_obj"]))
        if "iteration" in pool_metadata:
            chosen_items.append(("iteration", pool_metadata["iteration"]))
        if "snapshot_version" in pool_metadata:
            chosen_items.append(("snapshot", pool_metadata["snapshot_version"]))
        if "num_timesteps" in pool_metadata:
            chosen_items.append(("step", pool_metadata["num_timesteps"]))
        config_lines.append("chosen  : " + format_kv_pairs(chosen_items))
        config_lines.append(
            "metric  : "
            + format_kv_pairs([
                ("mean_ic", selected_metrics["ic"]),
                ("mean_rank_ic", selected_metrics["rank_ic"]),
                ("icir", selected_metrics["icir"]),
                ("rank_icir", selected_metrics["rank_icir"]),
            ])
        )
        config_lines.append(f"lowlvl  : {pool_metadata['best_pool_path']}")
    elif pool_metadata["source"] == "low_level_result_dir":
        selected_metrics = pool_metadata["selected_metrics"]
        config_lines.append(
            "chosen  : "
            + format_kv_pairs([
                ("mode", "scanned_checkpoints"),
                ("step", pool_metadata["best_step"]),
                ("by", pool_metadata["selection_metric"]),
                ("score", pool_metadata["selection_score"]),
                ("candidates", pool_metadata["candidate_count"]),
            ])
        )
        config_lines.append(
            "metric  : "
            + format_kv_pairs([
                ("mean_ic", selected_metrics["ic"]),
                ("mean_rank_ic", selected_metrics["rank_ic"]),
                ("icir", selected_metrics["icir"]),
                ("rank_icir", selected_metrics["rank_icir"]),
            ])
        )
        config_lines.append(f"lowlvl  : {pool_metadata['best_pool_path']}")

    print(format_block("[High-Level Training]", config_lines, width=96, border_char="="))

    env_config = HighLevelEnvConfig(
        lookback_days=lookback_days,
        episode_days=episode_days,
        target_horizon_days=target_horizon_days,
        rebalance_interval_days=rebalance_interval_days,
        max_leverage=max_leverage,
        action_scale=action_scale,
        action_mode=action_mode,
        action_top_k=action_top_k,
        reward_mode=reward_mode,
        weight_l2_penalty=weight_l2_penalty,
        base_deviation_penalty=base_deviation_penalty,
        smoothness_penalty=smoothness_penalty,
        random_start=True,
    )
    train_env = HighLevelEnv(
        exprs=exprs,
        calculator=calculators[0],
        base_weights=base_weights,
        config=env_config,
    )
    eval_config = replace(env_config, random_start=False, episode_days=None)
    eval_envs = {
        "train_eval": HighLevelEnv(exprs, calculators[0], base_weights, eval_config),
        "test_1": HighLevelEnv(exprs, calculators[1], base_weights, eval_config),
        "test_2": HighLevelEnv(exprs, calculators[2], base_weights, eval_config),
        "test_3": HighLevelEnv(exprs, calculators[3], base_weights, eval_config),
    }

    resolved_pool_payload: Dict[str, Any] = {
        "exprs": [str(expr) for expr in exprs],
        "weights": [float(weight) for weight in base_weights],
        "projected_base_weights": train_env.base_weights.tolist(),
        "action_factor_indices": train_env.action_factor_indices,
        "action_factor_names": train_env.action_factor_names,
        "rebalance_interval_days": train_env.rebalance_interval_days,
    }
    resolved_pool_payload.update(pool_metadata)
    with open(os.path.join(save_path, "resolved_pool.json"), "w", encoding="utf-8") as f:
        json.dump(resolved_pool_payload, f, ensure_ascii=False, indent=2)

    baseline_metrics = []
    for name, env in eval_envs.items():
        calc_idx = 0 if name == "train_eval" else int(name[-1])
        metrics = score_pool_state(
            exprs,
            env.base_weights.tolist(),
            calculators[calc_idx],
        )
        baseline_metrics.append((name, {
            "ic": metrics["ic"],
            "icir": metrics["icir"],
            "rank_ic": metrics["rank_ic"],
            "rank_icir": metrics["rank_icir"],
        }))

    print()
    print(format_baseline_summary(baseline_metrics))

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        ent_coef=ent_coef,
        gamma=gamma,
        policy_kwargs=dict(
            features_extractor_class=HighLevelSequenceNet,
            features_extractor_kwargs=dict(
                d_model=policy_d_model,
                n_layers=policy_n_layers,
                dropout=policy_dropout,
            ),
            log_std_init=policy_log_std_init,
        ),
        tensorboard_log="./out/tensorboard/high_level",
        device=device,
        verbose=1,
    )
    callback = HighLevelEvalCallback(
        save_path=save_path,
        eval_envs=eval_envs,
        reward_mode=reward_mode,
        verbose=1,
    )
    model.learn(total_timesteps=int(steps), callback=callback, tb_log_name=os.path.basename(save_path))


def main(
    random_seeds: Union[int, Tuple[int, ...]] = (0,),
    instruments: str = "csi300",
    pool_path: Optional[str] = None,
    low_level_result_dir: Optional[str] = "/home/liuyu/AlphaHRL/out/results/low_level/csi300_20_0_20260331182039_rl_pxd_tnh",
    pool_selection_metric: str = "ic",
    steps: int = 220_000,
    lookback_days: int = 20,
    episode_days: int = 256,
    target_horizon_days: int = 20,
    rebalance_interval_days: int = 10,
    max_leverage: float = 1.0,
    action_scale: float = 0.1,
    action_mode: str = "gate",
    action_top_k: Optional[int] = 8,
    reward_mode: str = "ic",
    weight_l2_penalty: float = 0.0,
    base_deviation_penalty: float = 1.0,
    smoothness_penalty: float = 0.5,
    learning_rate: float = 1e-4,
    n_steps: int = 512,
    batch_size: int = 128,
    ent_coef: float = 0.0,
    gamma: float = 0.99,
    policy_d_model: int = 128,
    policy_n_layers: int = 2,
    policy_dropout: float = 0.1,
    policy_log_std_init: float = -2.0,
    device_str: str = "cuda:4",
):
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds,)

    for seed in random_seeds:
        run_single_experiment(
            seed=int(seed),
            instruments=instruments,
            pool_path=pool_path,
            low_level_result_dir=low_level_result_dir,
            pool_selection_metric=pool_selection_metric,
            steps=int(steps),
            lookback_days=int(lookback_days),
            episode_days=int(episode_days),
            target_horizon_days=int(target_horizon_days),
            rebalance_interval_days=int(rebalance_interval_days),
            max_leverage=float(max_leverage),
            action_scale=float(action_scale),
            action_mode=action_mode,
            action_top_k=None if action_top_k is None else int(action_top_k),
            reward_mode=reward_mode,
            weight_l2_penalty=float(weight_l2_penalty),
            base_deviation_penalty=float(base_deviation_penalty),
            smoothness_penalty=float(smoothness_penalty),
            learning_rate=float(learning_rate),
            n_steps=int(n_steps),
            batch_size=int(batch_size),
            ent_coef=float(ent_coef),
            gamma=float(gamma),
            policy_d_model=int(policy_d_model),
            policy_n_layers=int(policy_n_layers),
            policy_dropout=float(policy_dropout),
            policy_log_std_init=float(policy_log_std_init),
            device_str=device_str,
        )


if __name__ == "__main__":
    fire.Fire(main)
