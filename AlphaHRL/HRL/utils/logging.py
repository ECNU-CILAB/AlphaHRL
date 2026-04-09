import logging
import math
import os
from numbers import Integral, Real
from typing import Any, Optional, Sequence, Tuple


def get_logger(name: str, file_path: Optional[str] = None) -> logging.Logger:
    if file_path is not None:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    logger = logging.getLogger(name)
    while logger.hasHandlers():
        handler = logger.handlers[0]
        handler.close()
        logger.removeHandler(handler)
    
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")

    if file_path is not None:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_null_logger() -> logging.Logger:
    logger = logging.getLogger("null_logger")
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def format_scalar(value: Any, digits: int = 4, none_text: str = "<none>") -> str:
    if value is None:
        return none_text
    if isinstance(value, str):
        return value if value != "" else none_text
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, Integral):
        return str(int(value))
    if isinstance(value, Real):
        value = float(value)
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def shorten_middle(text: str, max_length: int = 120) -> str:
    if max_length < 5 or len(text) <= max_length:
        return text
    remaining = max_length - 3
    head = remaining // 2
    tail = remaining - head
    return f"{text[:head]}...{text[-tail:]}"


def format_kv_pairs(
    items: Sequence[Tuple[str, Any]],
    digits: int = 4,
    none_text: str = "<none>",
    separator: str = " | ",
) -> str:
    return separator.join(
        f"{key}={format_scalar(value, digits=digits, none_text=none_text)}"
        for key, value in items
    )


def format_block(
    title: str,
    lines: Sequence[str],
    width: int = 96,
    border_char: str = "=",
) -> str:
    width = max(width, len(title))
    border = border_char * width
    return "\n".join([border, title, *lines, border])


def format_iteration_summary(
    iteration: int,
    total_timesteps: int,
    pool_size: int,
    pool_capacity: int,
    significant_count: int,
    eval_count: int,
    best_ic_ret: float,
    ic_test_mean: float,
    rank_ic_test_mean: float,
    checkpoint_path: str,
    test_details: Optional[Sequence[Tuple[str, float, float]]] = None,
    leader_entries: Optional[Sequence[Tuple[int, float, str]]] = None,
    best_ic_delta: Optional[float] = None,
    llm_use_count: Optional[int] = None,
    next_llm_step: Optional[int] = None,
) -> str:
    title = f"[Iteration {iteration}] timesteps={format_scalar(total_timesteps, digits=0)}"
    pool_items = [
        ("size", f"{pool_size}/{pool_capacity}"),
        ("significant", significant_count),
        ("eval", eval_count),
        ("best_ic", best_ic_ret),
    ]
    if best_ic_delta is not None:
        pool_items.append(("delta_ic", best_ic_delta))

    lines = [
        f"pool    : {format_kv_pairs(pool_items)}",
        f"test    : {format_kv_pairs([('ic_mean', ic_test_mean), ('rank_ic_mean', rank_ic_test_mean)])}",
    ]

    if test_details:
        split_parts = [
            f"{name} {format_kv_pairs([('ic', ic), ('rank_ic', rank_ic)])}"
            for name, ic, rank_ic in test_details
        ]
        lines.append(f"splits  : {'; '.join(split_parts)}")

    if leader_entries:
        leader_parts = [
            f"#{idx} w={format_scalar(weight)} {shorten_middle(expr, max_length=72)}"
            for idx, weight, expr in leader_entries
        ]
        lines.append(f"leaders : {'; '.join(leader_parts)}")

    if llm_use_count is not None or next_llm_step is not None:
        llm_items = []
        if llm_use_count is not None:
            llm_items.append(("uses", llm_use_count))
        if next_llm_step is not None:
            llm_items.append(("next_step", next_llm_step))
        lines.append(f"llm     : {format_kv_pairs(llm_items, digits=0)}")

    lines.append(f"ckpt    : {checkpoint_path}")
    return format_block(title, lines, width=96, border_char="-")


def format_high_level_iteration_summary(
    iteration: int,
    total_timesteps: int,
    n_factors: int,
    n_action_factors: int,
    rebalance_interval_days: int,
    action_mode: str,
    reward_mode: str,
    metrics_by_env: Sequence[Tuple[str, Sequence[Tuple[str, Any]]]],
    checkpoint_path: str,
) -> str:
    title = (
        f"[High-Level Iteration {iteration}] "
        f"timesteps={format_scalar(total_timesteps, digits=0)}"
    )
    lines = [
        "env     : " + format_kv_pairs([
            ("factors", n_factors),
            ("action_factors", n_action_factors),
            ("rebalance", rebalance_interval_days),
            ("action_mode", action_mode),
            ("reward_mode", reward_mode),
        ], digits=0),
    ]

    for env_name, items in metrics_by_env:
        lines.append(f"{env_name:<8}: {format_kv_pairs(items)}")

    lines.append(f"ckpt    : {checkpoint_path}")
    return format_block(title, lines, width=96, border_char="-")
