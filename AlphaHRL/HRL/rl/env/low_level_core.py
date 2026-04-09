from typing import Any, Dict, List, Optional, Set, Tuple
import math

import gymnasium as gym
import torch

from AlphaHRL.HRL.config import MAX_EXPR_LENGTH
from AlphaHRL.HRL.data.expression import *
from AlphaHRL.HRL.data.tokens import *
from AlphaHRL.HRL.data.tree import ExpressionBuilder
from AlphaHRL.HRL.models.alpha_pool import AlphaPoolBase
from AlphaHRL.HRL.rl.env.novelty_reward import (
    TerminalRedundancyPenaltyConfig,
    TerminalRedundancyPenaltyResult,
    TerminalRedundancyPenaltyer,
    TerminalNoveltyRewardConfig,
    TerminalNoveltyRewardResult,
    TerminalNoveltyRewarder,
)
from AlphaHRL.HRL.rl.env.proxy_reward import SubtreeProxyRewardConfig, SubtreeProxyRewarder
from AlphaHRL.HRL.utils import reseed_everything
from AlphaHRL.HRL.utils.logging import format_block, format_kv_pairs, shorten_middle


class LowLevelEnvCore(gym.Env):
    _VALID_PROXY_REWARD_MODES = {"shaping", "dense"}
    pool: AlphaPoolBase
    _tokens: List[Token]
    _builder: ExpressionBuilder
    _print_expr: bool
    _log_rewards: bool
    _seen_featured_exprs: Set[str]
    _proxy_potential: float
    _proxy_validity_potential: float
    _proxy_stability_potential: float
    _proxy_stable_days_total: float
    _proxy_total_days_total: float
    _proxy_subtree_count_total: float
    _proxy_finite_ratio_total: float
    _proxy_stability_ratio_total: float
    _proxy_reward_mode: str

    def __init__(
        self,
        pool: AlphaPoolBase,
        device: torch.device = torch.device("cuda:0"),
        print_expr: bool = False,
        log_rewards: bool = True,
        subtree_reward_config: Optional[SubtreeProxyRewardConfig] = None,
        terminal_novelty_reward_config: Optional[TerminalNoveltyRewardConfig] = None,
        terminal_redundancy_penalty_config: Optional[TerminalRedundancyPenaltyConfig] = None,
        proxy_reward_mode: str = "shaping"
    ):
        super().__init__()

        if proxy_reward_mode not in self._VALID_PROXY_REWARD_MODES:
            raise ValueError(
                f"Invalid proxy_reward_mode: {proxy_reward_mode}. "
                f"Expected one of {sorted(self._VALID_PROXY_REWARD_MODES)}."
            )

        self.pool = pool
        self._print_expr = print_expr
        self._log_rewards = log_rewards
        self._device = device
        self._proxy_rewarder = SubtreeProxyRewarder(pool, subtree_reward_config)
        self._terminal_novelty_rewarder = TerminalNoveltyRewarder(pool, terminal_novelty_reward_config)
        self._terminal_redundancy_penaltyer = TerminalRedundancyPenaltyer(
            pool,
            terminal_redundancy_penalty_config,
        )
        self._proxy_reward_mode = proxy_reward_mode

        self.eval_cnt = 0

        self.render_mode = None
        self.reset()

    def reset(
        self, *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ) -> Tuple[List[Token], dict]:
        reseed_everything(seed)
        self._tokens = [BEG_TOKEN]
        self._builder = ExpressionBuilder()
        self._seen_featured_exprs = set()
        self._proxy_potential = 0.0
        self._proxy_validity_potential = 0.0
        self._proxy_stability_potential = 0.0
        self._proxy_stable_days_total = 0.0
        self._proxy_total_days_total = 0.0
        self._proxy_subtree_count_total = 0.0
        self._proxy_finite_ratio_total = 0.0
        self._proxy_stability_ratio_total = 0.0
        return self._tokens, self._valid_action_types()

    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        expr_idx = self.eval_cnt
        proxy_info = self._empty_proxy_info()
        novelty_info = self._empty_novelty_info()
        redundancy_info = self._empty_redundancy_info()
        task_reward = 0.0
        if (
            isinstance(action, SequenceIndicatorToken) and
            action.indicator == SequenceIndicatorType.SEP
        ):
            task_reward, novelty_info, redundancy_info = self._evaluate()
            proxy_info = self._terminal_proxy_info()
            reward = (
                task_reward
                + novelty_info["novelty_reward"]
                - redundancy_info["redundancy_penalty"]
                + proxy_info["proxy_reward"]
            )
            self._log_reward(
                event="final_sep",
                expr_idx=expr_idx,
                expr_repr=self._builder_repr(),
                reward=reward,
                task_reward=task_reward,
                novelty_info=novelty_info,
                redundancy_info=redundancy_info,
                proxy_info=proxy_info,
            )
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            proxy_info = self._evaluate_new_subtrees()
            reward = proxy_info["proxy_reward"]
            self._log_reward(
                event="step",
                expr_idx=expr_idx,
                expr_repr=self._builder_repr(),
                reward=reward,
                task_reward=task_reward,
                novelty_info=novelty_info,
                redundancy_info=redundancy_info,
                proxy_info=proxy_info,
            )
        else:
            done = True
            if self._builder.is_valid():
                task_reward, novelty_info, redundancy_info = self._evaluate()
            else:
                task_reward = -1.0
            proxy_info = self._terminal_proxy_info()
            reward = (
                task_reward
                + novelty_info["novelty_reward"]
                - redundancy_info["redundancy_penalty"]
                + proxy_info["proxy_reward"]
            )
            self._log_reward(
                event="final_max_length",
                expr_idx=expr_idx,
                expr_repr=self._builder_repr(),
                reward=reward,
                task_reward=task_reward,
                novelty_info=novelty_info,
                redundancy_info=redundancy_info,
                proxy_info=proxy_info,
            )

        if math.isnan(reward):
            reward = 0.0

        info = self._valid_action_types()
        info["task_reward"] = task_reward
        info.update(novelty_info)
        info.update(redundancy_info)
        info.update({k: v for k, v in proxy_info.items() if k != "subtree_logs"})
        return self._tokens, reward, done, False, info

    def _evaluate(self) -> Tuple[float, Dict[str, Any], Dict[str, Any]]:
        expr: Expression = self._builder.get_tree()
        if self._print_expr and not self._log_rewards:
            print(expr)
        novelty_result = self._terminal_novelty_rewarder.evaluate(expr)
        redundancy_result = self._terminal_redundancy_penaltyer.evaluate(expr)
        self._terminal_novelty_rewarder.register(expr)
        self._terminal_redundancy_penaltyer.register(expr)
        novelty_info = self._novelty_info(novelty_result)
        redundancy_info = self._redundancy_info(redundancy_result)
        try:
            ret = self.pool.try_new_expr(expr)
            self.eval_cnt += 1
            return ret, novelty_info, redundancy_info
        except OutOfDataRangeError:
            return 0.0, novelty_info, redundancy_info

    def _evaluate_new_subtrees(self) -> Dict[str, Any]:
        results = []
        subtree_logs = []
        for expr in self._builder.stack:
            if not expr.is_featured:
                continue
            expr_key = str(expr)
            if expr_key in self._seen_featured_exprs:
                continue
            self._seen_featured_exprs.add(expr_key)
            result = self._proxy_rewarder.evaluate(expr)
            if result.evaluated:
                results.append(result)
                subtree_logs.append({
                    "expr": expr_key,
                    "reward": result.total_reward,
                    "validity": result.validity_reward,
                    "stability": result.stability_reward,
                    "finite_ratio": result.finite_ratio,
                    "stability_ratio": result.stability_ratio,
                    "valid_days": result.valid_days,
                    "total_days": result.total_days,
                    "stable_days": result.stable_days,
                })

        if len(results) == 0:
            return self._empty_proxy_info()

        subtree_count = len(results)
        validity_reward = sum(r.validity_reward for r in results)
        stability_reward = sum(r.stability_reward for r in results)
        proxy_reward = validity_reward + stability_reward
        stable_days = float(sum(r.stable_days for r in results))
        total_days = float(sum(r.total_days for r in results))
        finite_ratio = sum(r.finite_ratio for r in results) / subtree_count
        stability_ratio = sum(r.stability_ratio for r in results) / subtree_count

        self._proxy_potential += proxy_reward
        self._proxy_validity_potential += validity_reward
        self._proxy_stability_potential += stability_reward
        self._proxy_stable_days_total += stable_days
        self._proxy_total_days_total += total_days
        self._proxy_subtree_count_total += float(subtree_count)
        self._proxy_finite_ratio_total += sum(r.finite_ratio for r in results)
        self._proxy_stability_ratio_total += sum(r.stability_ratio for r in results)
        return {
            "proxy_reward": proxy_reward,
            "proxy_validity_reward": validity_reward,
            "proxy_stability_reward": stability_reward,
            "proxy_finite_ratio": finite_ratio,
            "proxy_stability_ratio": stability_ratio,
            "proxy_stable_days": stable_days,
            "proxy_total_days": total_days,
            "proxy_subtree_count": float(subtree_count),
            "proxy_potential": self._proxy_potential,
            "subtree_logs": subtree_logs,
        }

    def _empty_proxy_info(self) -> Dict[str, Any]:
        return {
            "proxy_reward": 0.0,
            "proxy_validity_reward": 0.0,
            "proxy_stability_reward": 0.0,
            "proxy_finite_ratio": self._cumulative_proxy_finite_ratio(),
            "proxy_stability_ratio": self._cumulative_proxy_stability_ratio(),
            "proxy_stable_days": self._proxy_stable_days_total,
            "proxy_total_days": self._proxy_total_days_total,
            "proxy_subtree_count": self._proxy_subtree_count_total,
            "proxy_potential": self._proxy_potential,
            "subtree_logs": [],
        }

    def _empty_novelty_info(self) -> Dict[str, Any]:
        return self._novelty_info(TerminalNoveltyRewardResult())

    def _novelty_info(self, result: TerminalNoveltyRewardResult) -> Dict[str, Any]:
        return {
            "novelty_reward": result.novelty_reward,
            "novelty_score": result.novelty_score,
            "novelty_max_similarity": result.max_similarity,
            "novelty_mean_similarity": result.mean_similarity,
            "novelty_compared_expr_count": result.compared_expr_count,
            "novelty_nearest_expr": result.nearest_expr,
            "novelty_evaluated": result.evaluated,
        }

    def _empty_redundancy_info(self) -> Dict[str, Any]:
        return self._redundancy_info(TerminalRedundancyPenaltyResult())

    def _redundancy_info(self, result: TerminalRedundancyPenaltyResult) -> Dict[str, Any]:
        return {
            "redundancy_penalty": result.redundancy_penalty,
            "redundancy_score": result.redundancy_score,
            "redundancy_max_similarity": result.max_similarity,
            "redundancy_mean_similarity": result.mean_similarity,
            "redundancy_compared_expr_count": result.compared_expr_count,
            "redundancy_nearest_expr": result.nearest_expr,
            "redundancy_evaluated": result.evaluated,
        }

    def _cumulative_proxy_finite_ratio(self) -> float:
        subtree_count = self._proxy_subtree_count_total
        if subtree_count == 0.0:
            return 0.0
        return self._proxy_finite_ratio_total / subtree_count

    def _cumulative_proxy_stability_ratio(self) -> float:
        subtree_count = self._proxy_subtree_count_total
        if subtree_count == 0.0:
            return 0.0
        return self._proxy_stability_ratio_total / subtree_count

    def _terminal_proxy_info(self) -> Dict[str, Any]:
        if self._proxy_reward_mode == "dense":
            return self._empty_proxy_info()

        penalty = -self._proxy_potential
        return {
            "proxy_reward": penalty,
            "proxy_validity_reward": -self._proxy_validity_potential,
            "proxy_stability_reward": -self._proxy_stability_potential,
            "proxy_finite_ratio": self._cumulative_proxy_finite_ratio(),
            "proxy_stability_ratio": self._cumulative_proxy_stability_ratio(),
            "proxy_stable_days": self._proxy_stable_days_total,
            "proxy_total_days": self._proxy_total_days_total,
            "proxy_subtree_count": self._proxy_subtree_count_total,
            "proxy_potential": self._proxy_potential,
            "subtree_logs": [],
        }

    def _builder_repr(self) -> str:
        if self._builder.is_valid():
            return str(self._builder.get_tree())
        if len(self._builder.stack) == 0:
            return "[]"
        return "[" + ", ".join(str(expr) for expr in self._builder.stack) + "]"

    def _log_reward(
        self,
        event: str,
        expr_idx: int,
        expr_repr: str,
        reward: float,
        task_reward: float,
        novelty_info: Dict[str, Any],
        redundancy_info: Dict[str, Any],
        proxy_info: Dict[str, Any],
    ) -> None:
        if not self._log_rewards:
            return
        if event == "step":
            print(self._format_step_log(expr_idx, expr_repr, reward, proxy_info))
            return

        print()
        print(
            self._format_final_log(
                event,
                expr_idx,
                expr_repr,
                reward,
                task_reward,
                novelty_info,
                redundancy_info,
                proxy_info,
            )
        )

    def _format_step_log(
        self,
        expr_idx: int,
        expr_repr: str,
        reward: float,
        proxy_info: Dict[str, Any],
    ) -> str:
        summary = format_kv_pairs([
            ("reward", reward),
            ("proxy", proxy_info["proxy_reward"]),
            ("validity", proxy_info["proxy_validity_reward"]),
            ("stability", proxy_info["proxy_stability_reward"]),
            ("subtrees", int(proxy_info["proxy_subtree_count"])),
            ("potential", proxy_info["proxy_potential"]),
        ])
        lines = [f"[Expr #{expr_idx} | STEP] {shorten_middle(expr_repr, max_length=120)} | {summary}"]
        lines.extend(self._format_subtree_logs(proxy_info.get("subtree_logs", [])))
        return "\n".join(lines)

    def _format_final_log(
        self,
        event: str,
        expr_idx: int,
        expr_repr: str,
        reward: float,
        task_reward: float,
        novelty_info: Dict[str, Any],
        redundancy_info: Dict[str, Any],
        proxy_info: Dict[str, Any],
    ) -> str:
        reason = "SEP" if event == "final_sep" else "MAX_LEN"
        title = f"[Expr #{expr_idx} | FINAL | {reason}]"
        if event == "final_max_length" and not self._builder.is_valid():
            title = f"{title} INVALID"

        lines = [
            f"expr    : {expr_repr}",
            "reward  : " + format_kv_pairs([
                ("total", reward),
                ("task", task_reward),
                ("novelty", novelty_info["novelty_reward"]),
                ("redundancy", -redundancy_info["redundancy_penalty"]),
                ("proxy", proxy_info["proxy_reward"]),
            ]),
        ]

        if bool(novelty_info.get("novelty_evaluated", False)):
            lines.append(
                "novelty : " + format_kv_pairs([
                    ("reward", novelty_info["novelty_reward"]),
                    ("score", novelty_info["novelty_score"]),
                    ("compared", novelty_info["novelty_compared_expr_count"]),
                    ("max_sim", novelty_info["novelty_max_similarity"]),
                    ("mean_sim", novelty_info["novelty_mean_similarity"]),
                    (
                        "nearest",
                        shorten_middle(novelty_info["novelty_nearest_expr"], max_length=60)
                        if novelty_info["novelty_nearest_expr"] else "<none>",
                    ),
                ])
            )

        if bool(redundancy_info.get("redundancy_evaluated", False)):
            lines.append(
                "redund. : " + format_kv_pairs([
                    ("penalty", redundancy_info["redundancy_penalty"]),
                    ("score", redundancy_info["redundancy_score"]),
                    ("compared", redundancy_info["redundancy_compared_expr_count"]),
                    ("max_sim", redundancy_info["redundancy_max_similarity"]),
                    ("mean_sim", redundancy_info["redundancy_mean_similarity"]),
                    (
                        "nearest",
                        shorten_middle(redundancy_info["redundancy_nearest_expr"], max_length=60)
                        if redundancy_info["redundancy_nearest_expr"] else "<none>",
                    ),
                ])
            )

        lines.append(
            "proxy   : " + format_kv_pairs([
                ("validity", proxy_info["proxy_validity_reward"]),
                ("stability", proxy_info["proxy_stability_reward"]),
                ("finite", proxy_info["proxy_finite_ratio"]),
                ("stable", proxy_info["proxy_stability_ratio"]),
                ("subtrees", int(proxy_info["proxy_subtree_count"])),
                ("potential", proxy_info["proxy_potential"]),
            ])
        )

        return format_block(title, lines, width=96, border_char="=") + "\n"

    def _format_subtree_logs(self, subtree_logs: List[Dict[str, Any]]) -> List[str]:
        formatted = []
        for subtree in subtree_logs:
            metrics = format_kv_pairs([
                ("reward", subtree["reward"]),
                ("validity", subtree["validity"]),
                ("stability", subtree["stability"]),
                ("finite", subtree["finite_ratio"]),
                ("stable", subtree["stability_ratio"]),
                ("days", f"{subtree['valid_days']}/{subtree['total_days']}"),
                ("stable_days", f"{subtree['stable_days']}/{subtree['valid_days']}"),
            ])
            formatted.append(
                f"  + subtree {shorten_middle(subtree['expr'], max_length=96)} | {metrics}"
            )
        return formatted

    def _valid_action_types(self) -> dict:
        valid_op_unary = self._builder.validate_op(UnaryOperator)
        valid_op_binary = self._builder.validate_op(BinaryOperator)
        valid_op_rolling = self._builder.validate_op(RollingOperator)
        valid_op_pair_rolling = self._builder.validate_op(PairRollingOperator)

        valid_op = valid_op_unary or valid_op_binary or valid_op_rolling or valid_op_pair_rolling
        valid_dt = self._builder.validate_dt()
        valid_const = self._builder.validate_const()
        valid_feature = self._builder.validate_featured_expr()
        valid_stop = self._builder.is_valid()

        return {
            "select": [valid_op, valid_feature, valid_const, valid_dt, valid_stop],
            "op": {
                UnaryOperator: valid_op_unary,
                BinaryOperator: valid_op_binary,
                RollingOperator: valid_op_rolling,
                PairRollingOperator: valid_op_pair_rolling,
            }
        }

    def valid_action_types(self) -> dict:
        return self._valid_action_types()

    def render(self, mode="human"):
        pass


AlphaEnvCore = LowLevelEnvCore
