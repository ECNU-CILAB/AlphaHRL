import json
import math
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from typing import Optional, Tuple, List, Union
from datetime import datetime
from pathlib import Path
from openai import OpenAI
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback

from AlphaHRL.HRL.data.expression import *
from AlphaHRL.HRL.data.parser import ExpressionParser
from AlphaHRL.HRL.models.linear_alpha_pool import LinearAlphaPool, MseAlphaPool
from AlphaHRL.HRL.rl.env.wrapper import AlphaEnv
from AlphaHRL.HRL.rl.env.novelty_reward import (
    TerminalNoveltyRewardConfig,
    TerminalRedundancyPenaltyConfig,
)
from AlphaHRL.HRL.rl.env.proxy_reward import SubtreeProxyRewardConfig
from AlphaHRL.HRL.rl.low_level_policy import LSTMSharedNet
from AlphaHRL.HRL.utils import reseed_everything, get_logger
from AlphaHRL.HRL.utils.logging import format_block, format_iteration_summary
from AlphaHRL.HRL.rl.env.core import AlphaEnvCore
from AlphaHRL.alphahrl_qlib.calculator import QLibStockDataCalculator
from AlphaHRL.alphahrl_qlib.stock_data import initialize_qlib
from AlphaHRL.alphahrl_llm.client import ChatClient, OpenAIClient, ChatConfig
from AlphaHRL.alphahrl_llm.prompts.system_prompt import EXPLAIN_WITH_TEXT_DESC
from AlphaHRL.alphahrl_llm.prompts.interaction import InterativeSession, DefaultInteraction


def read_alphagpt_init_pool(seed: int) -> List[Expression]:
    DIR = "./out/llm-tests/interaction"
    parser = build_parser()
    for path in Path(DIR).glob(f"v0_{seed}*"):
        with open(path / "report.json") as f:
            data = json.load(f)
            pool_state = data[-1]["pool_state"]
            return [parser.parse(expr) for expr, _ in pool_state]
    return []


def build_parser() -> ExpressionParser:
    return ExpressionParser(
        Operators,
        ignore_case=True,
        non_positive_time_deltas_allowed=False,
        additional_operator_mapping={
            "Max": [Greater],
            "Min": [Less],
            "Delta": [Sub]
        }
    )


def build_chat_client(log_dir: str) -> ChatClient:
    logger = get_logger("llm", os.path.join(log_dir, "llm.log"))
    return OpenAIClient(
        client=OpenAI(base_url="https://api.ai.cs.ac.cn/v1"),
        config=ChatConfig(
            system_prompt=EXPLAIN_WITH_TEXT_DESC,
            logger=logger
        )
    )


def build_low_level_run_name(
    instruments: str,
    pool_capacity: int,
    seed: int,
    timestamp: str,
    tag: str,
    use_subtree_proxy_reward: bool,
    subtree_reward_mode: str,
    use_terminal_novelty_reward: bool,
    terminal_novelty_comparison_mode: str,
    use_terminal_redundancy_penalty: bool,
    terminal_redundancy_comparison_mode: str,
) -> str:
    proxy_tag = (
        "px0" if not use_subtree_proxy_reward else
        "pxs" if subtree_reward_mode == "shaping" else
        "pxd"
    )
    terminal_novelty_tag = (
        "tn0" if not use_terminal_novelty_reward else
        "tnp" if terminal_novelty_comparison_mode == "pool" else
        "tnh"
    )

    tags = [proxy_tag, terminal_novelty_tag]
    if use_terminal_redundancy_penalty:
        redundancy_tag = (
            "rdp" if terminal_redundancy_comparison_mode == "pool" else
            "rdh"
        )
        tags.append(redundancy_tag)

    return (
        f"{instruments}_{pool_capacity}_{seed}_{timestamp}_{tag}_"
        + "_".join(tags)
    )


def build_low_level_training_summary_lines(
    *,
    name_prefix: str,
    start_time: str,
    seed: int,
    instruments: str,
    steps: int,
    pool_capacity: int,
    use_llm: bool,
    alphagpt_init: bool,
    llm_every_n_steps: int,
    llm_replace_n: int,
    drop_rl_n: int,
    use_subtree_proxy_reward: bool,
    subtree_reward_mode: str,
    subtree_proxy_days: int,
    subtree_validity_reward: float,
    subtree_stability_reward: float,
    subtree_stable_ratio_threshold: float,
    use_terminal_novelty_reward: bool,
    terminal_novelty_reward_scale: float,
    terminal_novelty_depth_decay: float,
    terminal_novelty_comparison_mode: str,
    use_terminal_redundancy_penalty: bool,
    terminal_redundancy_penalty_scale: float,
    terminal_redundancy_depth_decay: float,
    terminal_redundancy_comparison_mode: str,
    save_path: str,
) -> List[str]:
    lines = [
        f"run     : {name_prefix}",
        f"start   : {start_time}",
        f"seed    : {seed}",
        f"market  : {instruments}",
        f"steps   : {steps}",
        f"pool    : {pool_capacity}",
        f"llm     : enabled={use_llm} | init_only={alphagpt_init} | every={llm_every_n_steps} | replace={llm_replace_n} | drop={drop_rl_n}",
        f"proxy   : enabled={use_subtree_proxy_reward} | mode={subtree_reward_mode} | days={subtree_proxy_days} | validity={subtree_validity_reward:.4f} | stability={subtree_stability_reward:.4f} | threshold={subtree_stable_ratio_threshold:.4f}",
        f"novelty : enabled={use_terminal_novelty_reward} | scale={terminal_novelty_reward_scale:.4f} | decay={terminal_novelty_depth_decay:.4f} | mode={terminal_novelty_comparison_mode}",
    ]
    if use_terminal_redundancy_penalty:
        lines.append(
            f"redund. : enabled={use_terminal_redundancy_penalty} | scale={terminal_redundancy_penalty_scale:.4f} | metric=mutual_ic | mode={terminal_redundancy_comparison_mode} | legacy_decay={terminal_redundancy_depth_decay:.4f}"
        )
    lines.append(f"output  : {save_path}")
    return lines


class CustomCallback(BaseCallback):
    def __init__(
        self,
        save_path: str,
        test_calculators: List[QLibStockDataCalculator],
        verbose: int = 0,
        chat_session: Optional[InterativeSession] = None,
        llm_every_n_steps: int = 25_000,
        drop_rl_n: int = 5
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.test_calculators = test_calculators
        os.makedirs(self.save_path, exist_ok=True)

        self.llm_use_count = 0
        self.last_llm_use = 0
        self.obj_history: List[Tuple[int, float]] = []
        self.llm_every_n_steps = llm_every_n_steps
        self.chat_session = chat_session
        self._drop_rl_n = drop_rl_n
        self._rollout_idx = 0
        self._last_best_ic_ret: Optional[float] = None
        self._best_pool_selection_key: Optional[Tuple[float, float, float, float]] = None
        self._best_pool_selection_key_loaded = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.chat_session is not None:
            self._try_use_llm()

        significant_count = int((np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', significant_count)
        self.logger.record('pool/best_ic_ret', self.pool.best_ic_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        ic_test_mean, rank_ic_test_mean = 0., 0.
        test_details = []
        if len(self.test_calculators) != 0:
            total_test_days = sum(self._calculator_n_days(calculator) for calculator in self.test_calculators)
            for i, test_calculator in enumerate(self.test_calculators, start=1):
                ic_test, rank_ic_test = self.pool.test_ensemble(test_calculator)
                days = self._calculator_n_days(test_calculator)
                ic_test_mean += ic_test * days / total_test_days
                rank_ic_test_mean += rank_ic_test * days / total_test_days
                self.logger.record(f'test/ic_{i}', ic_test)
                self.logger.record(f'test/rank_ic_{i}', rank_ic_test)
                test_details.append((f"test_{i}", ic_test, rank_ic_test))
        self.logger.record(f'test/ic_mean', ic_test_mean)
        self.logger.record(f'test/rank_ic_mean', rank_ic_test_mean)
        checkpoint_path = self.save_checkpoint()
        self._rollout_idx += 1
        current_ic_ret, current_obj = self.pool.calculate_ic_and_objective()
        _, rank_ic, rank_icir, icir = self._calculate_train_selection_metrics()
        self._save_best_pool_if_needed(
            best_ic_ret=current_ic_ret,
            best_obj=current_obj,
            rank_ic=rank_ic,
            rank_icir=rank_icir,
            icir=icir,
        )
        best_ic_delta = None if self._last_best_ic_ret is None else self.pool.best_ic_ret - self._last_best_ic_ret
        self._last_best_ic_ret = self.pool.best_ic_ret
        next_llm_step = None
        if self.chat_session is not None:
            next_llm_step = self.last_llm_use + self.llm_every_n_steps

        print()
        print(format_iteration_summary(
            iteration=self._rollout_idx,
            total_timesteps=self.num_timesteps,
            pool_size=self.pool.size,
            pool_capacity=self.pool.capacity,
            significant_count=significant_count,
            eval_count=self.pool.eval_cnt,
            best_ic_ret=self.pool.best_ic_ret,
            ic_test_mean=ic_test_mean,
            rank_ic_test_mean=rank_ic_test_mean,
            checkpoint_path=checkpoint_path,
            test_details=test_details,
            leader_entries=self._top_pool_entries(limit=3),
            best_ic_delta=best_ic_delta,
            llm_use_count=self.llm_use_count if self.chat_session is not None else None,
            next_llm_step=next_llm_step,
        ))

    def save_checkpoint(self) -> str:
        path = os.path.join(self.save_path, f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_json_dict(), f)
        return path

    def _save_best_pool_if_needed(
        self,
        best_ic_ret: float,
        best_obj: float,
        rank_ic: float,
        rank_icir: float,
        icir: float,
    ) -> None:
        self._ensure_best_pool_selection_key_loaded()
        selection_key = (
            float(best_ic_ret),
            float(rank_ic),
            float(rank_icir),
            float(icir),
        )
        if not self._is_better_best_pool_candidate(selection_key):
            return

        best_pool_path = os.path.join(self.save_path, "best_pool.json")
        best_payload = self.pool.to_json_dict()
        best_payload.update({
            "best_ic_ret": float(best_ic_ret),
            "rank_ic": float(rank_ic),
            "rank_icir": float(rank_icir),
            "icir": float(icir),
            "iteration": int(self._rollout_idx),
            "num_timesteps": int(self.num_timesteps),
            "pool_size": int(self.pool.size),
        })
        snapshot_version = getattr(self.pool, "best_snapshot_version", None)
        if isinstance(snapshot_version, int):
            best_payload["snapshot_version"] = int(snapshot_version)
        if not math.isclose(float(best_obj), float(best_ic_ret), rel_tol=1e-12, abs_tol=1e-12):
            best_payload["best_obj"] = float(best_obj)
        with open(best_pool_path, "w", encoding="utf-8") as f:
            json.dump(best_payload, f, ensure_ascii=False, indent=2)
        self._best_pool_selection_key = selection_key

    def _is_better_best_pool_candidate(self, selection_key: Tuple[float, float, float, float]) -> bool:
        if self._best_pool_selection_key is None:
            return True

        for candidate_value, saved_value in zip(selection_key, self._best_pool_selection_key):
            if math.isclose(candidate_value, saved_value, rel_tol=1e-12, abs_tol=1e-12):
                continue
            return candidate_value > saved_value
        return False

    def _ensure_best_pool_selection_key_loaded(self) -> None:
        if self._best_pool_selection_key_loaded:
            return
        self._best_pool_selection_key_loaded = True
        self._load_existing_best_pool_selection_key()

    def _load_existing_best_pool_selection_key(self) -> None:
        best_pool_path = os.path.join(self.save_path, "best_pool.json")
        if not os.path.isfile(best_pool_path):
            return

        try:
            with open(best_pool_path, encoding="utf-8") as f:
                raw_pool = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        if not isinstance(raw_pool, dict):
            return

        best_ic_ret = raw_pool.get("best_ic_ret")
        rank_ic = raw_pool.get("rank_ic")
        rank_icir = raw_pool.get("rank_icir")
        icir = raw_pool.get("icir")
        if all(isinstance(value, (int, float)) for value in (best_ic_ret, rank_ic, rank_icir, icir)):
            self._best_pool_selection_key = (
                float(best_ic_ret),
                float(rank_ic),
                float(rank_icir),
                float(icir),
            )
            return

        inferred_metrics = self._infer_train_selection_metrics_from_saved_pool(raw_pool)
        if inferred_metrics is not None:
            self._best_pool_selection_key = inferred_metrics
            return

        if isinstance(best_ic_ret, (int, float)):
            self._best_pool_selection_key = (
                float(best_ic_ret),
                float("-inf"),
                float("-inf"),
                float("-inf"),
            )

    def _infer_train_selection_metrics_from_saved_pool(
        self,
        raw_pool: dict,
    ) -> Optional[Tuple[float, float, float, float]]:

        expr_strings = raw_pool.get("exprs")
        weights = raw_pool.get("weights")
        if not isinstance(expr_strings, list) or not isinstance(weights, list):
            return None
        if len(expr_strings) != len(weights):
            return None
        if any(not isinstance(expr, str) for expr in expr_strings):
            return None
        if any(not isinstance(weight, (int, float)) for weight in weights):
            return None

        parser = build_parser()
        try:
            exprs = [parser.parse(expr) for expr in expr_strings]
        except Exception:
            return None

        return self._calculate_train_selection_metrics(
            exprs=exprs,
            weights=[float(weight) for weight in weights],
        )

    def _calculate_train_selection_metrics(
        self,
        exprs: Optional[List[Expression]] = None,
        weights: Optional[List[float]] = None,
    ) -> Tuple[float, float, float, float]:
        if exprs is None:
            exprs = self.pool.exprs[:self.pool.size]  # type: ignore
        if weights is None:
            weights = [float(weight) for weight in self.pool.weights]
        if len(exprs) == 0:
            return 0.0, 0.0, 0.0, 0.0

        calculator = self.pool.calculator
        if hasattr(calculator, "calc_pool_all_ret_with_ir"):
            try:
                ic, icir, rank_ic, rank_icir = calculator.calc_pool_all_ret_with_ir(exprs, weights)
                return float(ic), float(rank_ic), float(rank_icir), float(icir)
            except ZeroDivisionError:
                pass

        ic, rank_ic = calculator.calc_pool_all_ret(exprs, weights)
        return (
            float(ic),
            float(rank_ic),
            self._fallback_ir(rank_ic),
            self._fallback_ir(ic),
        )

    def _fallback_ir(self, mean_value: float) -> float:
        if mean_value > 0.0:
            return float("inf")
        if mean_value < 0.0:
            return float("-inf")
        return 0.0

    def _calculator_n_days(self, calculator: QLibStockDataCalculator) -> int:
        data = getattr(calculator, "data", None)
        if data is not None and hasattr(data, "n_days"):
            return int(data.n_days)
        if hasattr(calculator, "n_days"):
            return int(calculator.n_days)
        raise AttributeError("Calculator must expose either data.n_days or n_days.")

    def show_pool_state(self):
        state = self.pool.state
        print('---------------------------------------------')
        for i in range(self.pool.size):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    def _try_use_llm(self) -> None:
        n_steps = self.num_timesteps
        if n_steps - self.last_llm_use < self.llm_every_n_steps:
            return
        self.last_llm_use = n_steps
        self.llm_use_count += 1
        
        assert self.chat_session is not None
        self.chat_session.client.reset()
        logger = self.chat_session.logger
        logger.debug(
            f"[Step: {n_steps}] Trying to invoke LLM (#{self.llm_use_count}): "
            f"IC={self.pool.best_ic_ret:.4f}, obj={self.pool.best_obj:.4f}")

        try:
            remain_n = max(0, self.pool.size - self._drop_rl_n)
            remain = self.pool.most_significant_indices(remain_n)
            self.pool.leave_only(remain)
            self.chat_session.update_pool(self.pool)
        except Exception as e:
            logger.warning(f"LLM invocation failed due to {type(e)}: {str(e)}")

    def _top_pool_entries(self, limit: int = 3) -> List[Tuple[int, float, str]]:
        if self.pool.size == 0 or limit <= 0:
            return []

        indices = self.pool.most_significant_indices(min(limit, self.pool.size))
        indices = sorted(indices, key=lambda idx: abs(float(self.pool.weights[idx])), reverse=True)
        state = self.pool.state
        return [
            (idx, float(self.pool.weights[idx]), str(state['exprs'][idx]))
            for idx in indices
        ]

    @property
    def pool(self) -> LinearAlphaPool:
        assert(isinstance(self.env_core.pool, LinearAlphaPool))
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def run_single_experiment(
    seed: int = 0,
    instruments: str = "csi300",
    pool_capacity: int = 10,
    steps: int = 200_000,
    alphagpt_init: bool = False,
    use_llm: bool = False,
    llm_every_n_steps: int = 25_000,
    drop_rl_n: int = 5,
    llm_replace_n: int = 3,
    use_subtree_proxy_reward: bool = True,
    subtree_reward_mode: str = "shaping",
    subtree_proxy_days: int = 64,
    subtree_validity_reward: float = 0.002,
    subtree_stability_reward: float = 0.001,
    subtree_stable_ratio_threshold: float = 0.95,
    use_terminal_novelty_reward: bool = False,
    terminal_novelty_reward_scale: float = 0.01,
    terminal_novelty_depth_decay: float = 0.8,
    terminal_novelty_comparison_mode: str = "pool",
    use_terminal_redundancy_penalty: bool = False,
    terminal_redundancy_penalty_scale: float = 0.01,
    terminal_redundancy_depth_decay: float = 0.8,
    terminal_redundancy_comparison_mode: str = "pool",
):
    if subtree_reward_mode not in ("shaping", "dense"):
        raise ValueError(
            f"Invalid subtree_reward_mode: {subtree_reward_mode}. "
            "Expected 'shaping' or 'dense'."
        )
    if terminal_novelty_comparison_mode not in ("pool", "history"):
        raise ValueError(
            f"Invalid terminal_novelty_comparison_mode: {terminal_novelty_comparison_mode}. "
            "Expected 'pool' or 'history'."
        )
    if terminal_redundancy_comparison_mode not in ("pool", "history"):
        raise ValueError(
            f"Invalid terminal_redundancy_comparison_mode: {terminal_redundancy_comparison_mode}. "
            "Expected 'pool' or 'history'."
        )

    reseed_everything(seed)
    initialize_qlib("./qlib_data/cn_data_rolling/")

    llm_replace_n = 0 if not use_llm else llm_replace_n
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # tag = "rlv2" if llm_add_subexpr == 0 else f"afs{llm_add_subexpr}aar1-5"
    tag = (
        "agpt" if alphagpt_init else
        "rl" if not use_llm else
        f"llm_d{drop_rl_n}")
    name_prefix = build_low_level_run_name(
        instruments=instruments,
        pool_capacity=pool_capacity,
        seed=seed,
        timestamp=timestamp,
        tag=tag,
        use_subtree_proxy_reward=use_subtree_proxy_reward,
        subtree_reward_mode=subtree_reward_mode,
        use_terminal_novelty_reward=use_terminal_novelty_reward,
        terminal_novelty_comparison_mode=terminal_novelty_comparison_mode,
        use_terminal_redundancy_penalty=use_terminal_redundancy_penalty,
        terminal_redundancy_comparison_mode=terminal_redundancy_comparison_mode,
    )
    save_path = os.path.join("./out/results/low_level", name_prefix)
    os.makedirs(save_path, exist_ok=True)

    print(format_block(
        "[Low-Level Training]",
        build_low_level_training_summary_lines(
            name_prefix=name_prefix,
            start_time=start_time,
            seed=seed,
            instruments=instruments,
            steps=steps,
            pool_capacity=pool_capacity,
            use_llm=use_llm,
            alphagpt_init=alphagpt_init,
            llm_every_n_steps=llm_every_n_steps,
            llm_replace_n=llm_replace_n,
            drop_rl_n=drop_rl_n,
            use_subtree_proxy_reward=use_subtree_proxy_reward,
            subtree_reward_mode=subtree_reward_mode,
            subtree_proxy_days=subtree_proxy_days,
            subtree_validity_reward=subtree_validity_reward,
            subtree_stability_reward=subtree_stability_reward,
            subtree_stable_ratio_threshold=subtree_stable_ratio_threshold,
            use_terminal_novelty_reward=use_terminal_novelty_reward,
            terminal_novelty_reward_scale=terminal_novelty_reward_scale,
            terminal_novelty_depth_decay=terminal_novelty_depth_decay,
            terminal_novelty_comparison_mode=terminal_novelty_comparison_mode,
            use_terminal_redundancy_penalty=use_terminal_redundancy_penalty,
            terminal_redundancy_penalty_scale=terminal_redundancy_penalty_scale,
            terminal_redundancy_depth_decay=terminal_redundancy_depth_decay,
            terminal_redundancy_comparison_mode=terminal_redundancy_comparison_mode,
            save_path=save_path,
        ),
        width=96,
        border_char="="
    ))

    device = torch.device("cuda:4")
    close = Feature(FeatureType.CLOSE)
    target = Ref(close, -20) / close - 1

    def get_dataset(start: str, end: str) -> StockData:
        return StockData(
            instrument=instruments,
            start_time=start,
            end_time=end,
            device=device
        )

    segments = [
        ("2012-01-01", "2021-12-31"),
        ("2022-01-01", "2022-06-30"),
        ("2022-07-01", "2022-12-31"),
        ("2023-01-01", "2023-06-30")
    ]
    datasets = [get_dataset(*s) for s in segments]
    calculators = [QLibStockDataCalculator(d, target) for d in datasets]

    def build_pool(exprs: List[Expression]) -> LinearAlphaPool:
        pool = MseAlphaPool(
            capacity=pool_capacity,
            calculator=calculators[0],
            ic_lower_bound=None,
            l1_alpha=5e-3,
            device=device
        )
        if len(exprs) != 0:
            pool.force_load_exprs(exprs)
        return pool

    chat, inter, pool = None, None, build_pool([])
    if alphagpt_init:
        pool = build_pool(read_alphagpt_init_pool(seed))
    elif use_llm:
        chat = build_chat_client(save_path)
        inter = DefaultInteraction(
            build_parser(), chat, build_pool,
            calculator_train=calculators[0], calculators_test=calculators[1:],
            replace_k=llm_replace_n, forgetful=True
        )
        pool = inter.run()

    env = AlphaEnv(
        pool=pool,
        device=device,
        print_expr=True,
        subtree_reward_config=SubtreeProxyRewardConfig(
            enabled=use_subtree_proxy_reward,
            proxy_days=subtree_proxy_days,
            validity_reward=subtree_validity_reward,
            stability_reward=subtree_stability_reward,
            stable_ratio_threshold=subtree_stable_ratio_threshold
        ),
        terminal_novelty_reward_config=TerminalNoveltyRewardConfig(
            enabled=use_terminal_novelty_reward,
            max_reward=terminal_novelty_reward_scale,
            depth_decay=terminal_novelty_depth_decay,
            comparison_mode=terminal_novelty_comparison_mode
        ),
        terminal_redundancy_penalty_config=TerminalRedundancyPenaltyConfig(
            enabled=use_terminal_redundancy_penalty,
            max_penalty=terminal_redundancy_penalty_scale,
            depth_decay=terminal_redundancy_depth_decay,
            comparison_mode=terminal_redundancy_comparison_mode,
        ),
        proxy_reward_mode=subtree_reward_mode
    )
    checkpoint_callback = CustomCallback(
        save_path=save_path,
        test_calculators=calculators[1:],
        verbose=1,
        chat_session=inter,
        llm_every_n_steps=llm_every_n_steps,
        drop_rl_n=drop_rl_n
    )
    model = MaskablePPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log="./out/tensorboard/low_level",
        device=device,
        verbose=1,
    )
    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=name_prefix,
    )


def main(
    random_seeds: Union[int, Tuple[int]] = 0,
    pool_capacity: int = 20,
    instruments: str = "csi300",
    alphagpt_init: bool = False,
    use_llm: bool = False,
    drop_rl_n: int = 10,
    steps: Optional[int] = 250000,
    llm_every_n_steps: int = 25000,
    use_subtree_proxy_reward: bool = True,
    subtree_reward_mode: str = "dense",
    subtree_proxy_days: int = 252,
    subtree_validity_reward: float = 0.001,
    subtree_stability_reward: float = 0.002,
    subtree_stable_ratio_threshold: float = 0.90,
    use_terminal_novelty_reward: bool = True,
    terminal_novelty_reward_scale: float = 0.01,
    terminal_novelty_depth_decay: float = 0.8,
    terminal_novelty_comparison_mode: str = "history",
):
    """
    :param random_seeds: Random seeds
    :param pool_capacity: Maximum size of the alpha pool
    :param instruments: Stock subset name
    :param alphagpt_init: Use an alpha set pre-generated by LLM as the initial pool
    :param use_llm: Enable LLM usage
    :param drop_rl_n: Drop n worst alphas before invoke the LLM
    :param steps: Total iteration steps
    :param llm_every_n_steps: Invoke LLM every n steps
    :param use_subtree_proxy_reward: Enable subtree-level proxy reward
    :param subtree_reward_mode: "shaping" for terminal settlement, "dense" for no terminal settlement
    :param subtree_proxy_days: Number of days used by the proxy evaluator
    :param subtree_validity_reward: Reward for each newly formed valid subtree
    :param subtree_stability_reward: Maximum absolute reward contributed by subtree stability
    :param subtree_stable_ratio_threshold: Stability threshold for deciding positive/negative reward
    :param use_terminal_novelty_reward: Enable novelty reward on completed expressions
    :param terminal_novelty_reward_scale: Maximum reward assigned by terminal novelty
    :param terminal_novelty_depth_decay: Depth decay used by the structural formula embedding
    :param terminal_novelty_comparison_mode: "pool" compares to retained pool formulas, "history" compares to all historical completed formulas
    """
    if isinstance(random_seeds, int):
        random_seeds = (random_seeds, )
    default_steps = {
        10: 200_000,
        20: 250_000,
        50: 300_000,
        100: 350_000
    }
    for s in random_seeds:
        run_single_experiment(
            seed=s,
            instruments=instruments,
            pool_capacity=pool_capacity,
            steps=default_steps[int(pool_capacity)] if steps is None else int(steps),
            alphagpt_init=alphagpt_init,
            drop_rl_n=drop_rl_n,
            use_llm=use_llm,
            llm_every_n_steps=llm_every_n_steps,
            use_subtree_proxy_reward=use_subtree_proxy_reward,
            subtree_reward_mode=subtree_reward_mode,
            subtree_proxy_days=subtree_proxy_days,
            subtree_validity_reward=subtree_validity_reward,
            subtree_stability_reward=subtree_stability_reward,
            subtree_stable_ratio_threshold=subtree_stable_ratio_threshold,
            use_terminal_novelty_reward=use_terminal_novelty_reward,
            terminal_novelty_reward_scale=terminal_novelty_reward_scale,
            terminal_novelty_depth_decay=terminal_novelty_depth_decay,
            terminal_novelty_comparison_mode=terminal_novelty_comparison_mode,
        )


if __name__ == '__main__':
    fire.Fire(main)
