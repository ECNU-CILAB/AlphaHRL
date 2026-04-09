from collections import defaultdict
from dataclasses import dataclass
import math
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from AlphaHRL.HRL.data.expression import (
    Constant,
    DeltaTime,
    Expression,
    Feature,
    Operator,
    OutOfDataRangeError,
)
from AlphaHRL.HRL.models.alpha_pool import AlphaPoolBase


_VALID_COMPARISON_MODES = {"pool", "history"}


@dataclass(frozen=True)
class TerminalNoveltyRewardConfig:
    enabled: bool = False
    max_reward: float = 0.01
    depth_decay: float = 0.8
    comparison_mode: str = "pool"

    def __post_init__(self) -> None:
        if self.comparison_mode not in _VALID_COMPARISON_MODES:
            raise ValueError(
                f"Invalid terminal novelty comparison_mode: {self.comparison_mode}. "
                f"Expected one of {sorted(_VALID_COMPARISON_MODES)}."
            )


@dataclass(frozen=True)
class TerminalRedundancyPenaltyConfig:
    enabled: bool = False
    max_penalty: float = 0.01
    depth_decay: float = 0.8
    comparison_mode: str = "pool"

    def __post_init__(self) -> None:
        if self.comparison_mode not in _VALID_COMPARISON_MODES:
            raise ValueError(
                f"Invalid terminal redundancy comparison_mode: {self.comparison_mode}. "
                f"Expected one of {sorted(_VALID_COMPARISON_MODES)}."
            )


@dataclass(frozen=True)
class TerminalNoveltyRewardResult:
    novelty_reward: float = 0.0
    novelty_score: float = 0.0
    max_similarity: float = 0.0
    mean_similarity: float = 0.0
    compared_expr_count: int = 0
    nearest_expr: str = ""
    evaluated: bool = False


@dataclass(frozen=True)
class TerminalRedundancyPenaltyResult:
    redundancy_penalty: float = 0.0
    redundancy_score: float = 0.0
    max_similarity: float = 0.0
    mean_similarity: float = 0.0
    compared_expr_count: int = 0
    nearest_expr: str = ""
    evaluated: bool = False


class _TerminalSemanticSimilarityHelper:
    def __init__(self, pool: AlphaPoolBase) -> None:
        self._pool = pool
        self._known_expr_keys: Set[str] = set()
        self._history_expr_keys: List[str] = []
        self._expr_by_key: Dict[str, Expression] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}
        self._sync_exprs_with_pool()

    def _pool_exprs(self) -> List[Expression]:
        exprs = getattr(self._pool, "exprs", None)
        size = int(getattr(self._pool, "size", 0))
        if exprs is not None and size > 0:
            return [expr for expr in exprs[:size] if isinstance(expr, Expression)]

        state = self._pool.state
        state_exprs = state.get("exprs", []) if isinstance(state, dict) else []
        return [expr for expr in state_exprs if isinstance(expr, Expression)]

    def _sync_exprs_with_pool(self) -> None:
        for expr in self._pool_exprs():
            self._remember_expr(expr, append_history=False)

    def _remember_expr(self, expr: Expression, append_history: bool) -> None:
        expr_key = str(expr)
        self._expr_by_key.setdefault(expr_key, expr)
        if append_history or expr_key not in self._known_expr_keys:
            self._history_expr_keys.append(expr_key)
        self._known_expr_keys.add(expr_key)

    def pool_expr_keys(self) -> List[str]:
        exprs = self._pool_exprs()
        for expr in exprs:
            self._expr_by_key.setdefault(str(expr), expr)
        return [str(expr) for expr in exprs]

    def history_expr_keys(self) -> List[str]:
        return list(self._history_expr_keys)

    def expr_by_key(self, expr_key: str) -> Optional[Expression]:
        return self._expr_by_key.get(expr_key)

    def semantic_similarity(
        self,
        lhs_expr: Expression,
        rhs_expr_key: str,
        rhs_expr: Optional[Expression] = None,
    ) -> float:
        lhs_expr_key = str(lhs_expr)
        cache_key = (
            (lhs_expr_key, rhs_expr_key)
            if lhs_expr_key <= rhs_expr_key
            else (rhs_expr_key, lhs_expr_key)
        )
        cached = self._similarity_cache.get(cache_key)
        if cached is not None:
            return cached

        if rhs_expr is None:
            rhs_expr = self.expr_by_key(rhs_expr_key)
        if rhs_expr is None:
            return 0.0

        calculator = getattr(self._pool, "calculator", None)
        if calculator is None or not hasattr(calculator, "calc_mutual_IC"):
            similarity = 0.0
        else:
            try:
                similarity = float(calculator.calc_mutual_IC(lhs_expr, rhs_expr))
            except (
                OutOfDataRangeError,
                RuntimeError,
                TypeError,
                ValueError,
                ZeroDivisionError,
                OverflowError,
            ):
                similarity = 0.0
            if not math.isfinite(similarity):
                similarity = 0.0
            similarity = min(1.0, max(0.0, abs(similarity)))

        self._similarity_cache[cache_key] = similarity
        return similarity


class TerminalNoveltyRewarder:
    def __init__(
        self,
        pool: AlphaPoolBase,
        config: Optional[TerminalNoveltyRewardConfig] = None
    ) -> None:
        self._pool = pool
        self._config = config or TerminalNoveltyRewardConfig()
        self._embedding_cache: Dict[str, Dict[str, float]] = {}
        self._known_expr_keys: Set[str] = set()
        self._history_expr_keys: List[str] = []
        self._sync_exprs_with_pool()

    def evaluate(self, expr: Expression) -> TerminalNoveltyRewardResult:
        if not self._config.enabled or not expr.is_featured:
            return TerminalNoveltyRewardResult()

        self._sync_exprs_with_pool()
        expr_key = str(expr)
        reference_expr_keys = self._reference_expr_keys(expr_key)
        if len(reference_expr_keys) == 0:
            return TerminalNoveltyRewardResult(
                novelty_reward=self._config.max_reward,
                novelty_score=1.0,
                max_similarity=0.0,
                mean_similarity=0.0,
                compared_expr_count=0,
                nearest_expr="",
                evaluated=True
            )

        expr_embedding = self._embedding(expr)
        max_similarity = 0.0
        similarity_sum = 0.0
        nearest_expr = ""
        compared_expr_count = len(reference_expr_keys)
        for reference_expr_key in reference_expr_keys:
            reference_embedding = self._embedding_cache.get(reference_expr_key, {})
            similarity = self._cosine_similarity(expr_embedding, reference_embedding)
            similarity_sum += similarity
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_expr = reference_expr_key

        mean_similarity = similarity_sum / compared_expr_count if compared_expr_count > 0 else 0.0
        novelty_score = max(0.0, 1.0 - max_similarity)
        return TerminalNoveltyRewardResult(
            novelty_reward=self._config.max_reward * novelty_score,
            novelty_score=novelty_score,
            max_similarity=max_similarity,
            mean_similarity=mean_similarity,
            compared_expr_count=compared_expr_count,
            nearest_expr=nearest_expr,
            evaluated=True
        )

    def register(self, expr: Expression) -> None:
        if expr.is_featured:
            self._remember_expr(expr, append_history=True)

    def _reference_expr_keys(self, expr_key: str) -> List[str]:
        if self._config.comparison_mode == "history":
            return list(self._history_expr_keys)

        reference_expr_keys = [str(expr) for expr in self._pool_exprs()]
        # Preserve the anti-farming behavior in pool mode: if a complete formula
        # has already appeared before but is not currently retained in the pool,
        # add one exact-match reference so its novelty becomes zero.
        if expr_key in self._known_expr_keys and expr_key not in reference_expr_keys:
            reference_expr_keys.append(expr_key)
        return reference_expr_keys

    def _pool_exprs(self) -> List[Expression]:
        exprs = getattr(self._pool, "exprs", None)
        size = int(getattr(self._pool, "size", 0))
        if exprs is not None and size > 0:
            return [expr for expr in exprs[:size] if isinstance(expr, Expression)]

        state = self._pool.state
        state_exprs = state.get("exprs", []) if isinstance(state, dict) else []
        return [expr for expr in state_exprs if isinstance(expr, Expression)]

    def _sync_exprs_with_pool(self) -> None:
        for expr in self._pool_exprs():
            self._remember_expr(expr, append_history=False)

    def _remember_expr(self, expr: Expression, append_history: bool) -> None:
        expr_key = str(expr)
        self._embedding(expr)
        if append_history or expr_key not in self._known_expr_keys:
            self._history_expr_keys.append(expr_key)
        self._known_expr_keys.add(expr_key)

    def _embedding(self, expr: Expression) -> Dict[str, float]:
        expr_key = str(expr)
        cached = self._embedding_cache.get(expr_key)
        if cached is not None:
            return cached

        weights: DefaultDict[str, float] = defaultdict(float)
        self._encode_expr(expr, depth=0, weights=weights, parent_label=None, child_index=None)
        embedding = self._normalize(weights)
        self._embedding_cache[expr_key] = embedding
        return embedding

    def _encode_expr(
        self,
        expr: Expression,
        depth: int,
        weights: DefaultDict[str, float],
        parent_label: Optional[str],
        child_index: Optional[int]
    ) -> None:
        label = self._node_label(expr)
        node_weight = self._config.depth_decay ** depth
        weights[f"node:{label}"] += node_weight
        if parent_label is not None and child_index is not None:
            weights[f"edge:{parent_label}->{label}"] += node_weight
            weights[f"slot:{parent_label}:{child_index}->{label}"] += 0.5 * node_weight

        if isinstance(expr, Operator):
            weights[f"arity:{type(expr).__name__}:{len(expr.operands)}"] += 0.5 * node_weight
            for idx, operand in enumerate(expr.operands):
                self._encode_expr(
                    operand,
                    depth=depth + 1,
                    weights=weights,
                    parent_label=label,
                    child_index=idx
                )

    def _node_label(self, expr: Expression) -> str:
        if isinstance(expr, Feature):
            return f"feature:{str(expr)}"
        if isinstance(expr, Constant):
            return (
                f"constant:{self._constant_sign(expr.value)}:"
                f"{self._constant_bucket(expr.value)}:{expr.value:g}"
            )
        if isinstance(expr, DeltaTime):
            return f"dt:{self._dt_bucket(expr._delta_time)}:{expr._delta_time}"
        if isinstance(expr, Operator):
            return f"op:{type(expr).__name__}"
        return type(expr).__name__

    def _constant_sign(self, value: float) -> str:
        if math.isclose(value, 0.0):
            return "zero"
        return "pos" if value > 0 else "neg"

    def _constant_bucket(self, value: float) -> str:
        abs_value = abs(value)
        if abs_value < 1e-6:
            return "0"
        if abs_value < 0.1:
            return "lt0.1"
        if abs_value < 1.0:
            return "lt1"
        if abs_value < 10.0:
            return "lt10"
        return "ge10"

    def _dt_bucket(self, delta_time: int) -> str:
        if delta_time <= 5:
            return "short"
        if delta_time <= 20:
            return "mid"
        if delta_time <= 60:
            return "long"
        return "xlong"

    def _normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        norm = math.sqrt(sum(value * value for value in weights.values()))
        if norm <= 1e-12:
            return {}
        return {key: value / norm for key, value in weights.items()}

    def _cosine_similarity(self, lhs: Dict[str, float], rhs: Dict[str, float]) -> float:
        if len(lhs) == 0 or len(rhs) == 0:
            return 0.0

        smaller, larger = (lhs, rhs) if len(lhs) <= len(rhs) else (rhs, lhs)
        similarity = sum(value * larger.get(key, 0.0) for key, value in smaller.items())
        return min(1.0, max(0.0, similarity))


class TerminalRedundancyPenaltyer:
    def __init__(
        self,
        pool: AlphaPoolBase,
        config: Optional[TerminalRedundancyPenaltyConfig] = None
    ) -> None:
        self._config = config or TerminalRedundancyPenaltyConfig()
        # `depth_decay` is kept in the config for CLI/test compatibility, but the
        # redundancy signal is now semantic and based on mutual IC.
        self._helper = _TerminalSemanticSimilarityHelper(pool)

    def evaluate(self, expr: Expression) -> TerminalRedundancyPenaltyResult:
        if not self._config.enabled or not expr.is_featured:
            return TerminalRedundancyPenaltyResult()

        self._helper._sync_exprs_with_pool()
        reference_expr_keys = self._reference_expr_keys()
        if len(reference_expr_keys) == 0:
            return TerminalRedundancyPenaltyResult(
                redundancy_penalty=0.0,
                redundancy_score=0.0,
                max_similarity=0.0,
                mean_similarity=0.0,
                compared_expr_count=0,
                nearest_expr="",
                evaluated=True,
            )

        max_similarity = 0.0
        similarity_sum = 0.0
        nearest_expr = ""
        compared_expr_count = len(reference_expr_keys)
        for reference_expr_key in reference_expr_keys:
            similarity = self._helper.semantic_similarity(expr, reference_expr_key)
            similarity_sum += similarity
            if similarity > max_similarity:
                max_similarity = similarity
                nearest_expr = reference_expr_key

        mean_similarity = similarity_sum / compared_expr_count if compared_expr_count > 0 else 0.0
        redundancy_score = max_similarity
        return TerminalRedundancyPenaltyResult(
            redundancy_penalty=self._config.max_penalty * redundancy_score,
            redundancy_score=redundancy_score,
            max_similarity=max_similarity,
            mean_similarity=mean_similarity,
            compared_expr_count=compared_expr_count,
            nearest_expr=nearest_expr,
            evaluated=True,
        )

    def register(self, expr: Expression) -> None:
        if expr.is_featured:
            self._helper._remember_expr(expr, append_history=True)

    def _reference_expr_keys(self) -> List[str]:
        if self._config.comparison_mode == "history":
            return self._helper.history_expr_keys()
        return self._helper.pool_expr_keys()
