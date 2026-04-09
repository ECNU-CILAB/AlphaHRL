from typing import List, Optional, Tuple

import gymnasium as gym
import numpy as np

from AlphaHRL.HRL.config import *
from AlphaHRL.HRL.data.expression import Expression
from AlphaHRL.HRL.data.tokens import *
from AlphaHRL.HRL.models.alpha_pool import AlphaPoolBase
from AlphaHRL.HRL.rl.env.low_level_core import AlphaEnvCore, LowLevelEnvCore

SIZE_NULL = 1
SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP


class LowLevelEnvWrapper(gym.Wrapper):
    state: np.ndarray
    env: LowLevelEnvCore
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Box
    counter: int

    def __init__(
        self,
        env: LowLevelEnvCore,
        subexprs: Optional[List[Expression]] = None
    ):
        super().__init__(env)
        self.subexprs = subexprs or []
        self.size_action = SIZE_ACTION + len(self.subexprs)
        self.action_space = gym.spaces.Discrete(self.size_action)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=self.size_action + SIZE_NULL - 1,
            shape=(MAX_EXPR_LENGTH,),
            dtype=np.uint8,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        self.counter = 0
        self.state = np.zeros(MAX_EXPR_LENGTH, dtype=np.uint8)
        self.env.reset(**kwargs)
        return self.state, {}

    def step(self, action: int):
        _, reward, done, truncated, info = self.env.step(self.action(action))
        if not done:
            self.state[self.counter] = action
            self.counter += 1
        return self.state, self.reward(reward), done, truncated, info

    def action(self, action: int) -> Token:
        return self.action_to_token(action)

    def reward(self, reward: float) -> float:
        return reward + REWARD_PER_STEP

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0
        for i in range(offset, offset + SIZE_OP):
            if valid["op"][OPERATORS[i - offset].category_type()]:
                res[i] = True
        offset += SIZE_OP
        if valid["select"][1]:
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid["select"][2]:
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid["select"][3]:
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid["select"][1]:
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid["select"][4]:
            res[offset] = True
        return res

    def action_to_token(self, action: int) -> Token:
        if action < 0:
            raise ValueError("Action index must be non-negative")
        if action < SIZE_OP:
            return OperatorToken(OPERATORS[action])
        action -= SIZE_OP
        if action < SIZE_FEATURE:
            return FeatureToken(FeatureType(action))
        action -= SIZE_FEATURE
        if action < SIZE_CONSTANT:
            return ConstantToken(CONSTANTS[action])
        action -= SIZE_CONSTANT
        if action < SIZE_DELTA_TIME:
            return DeltaTimeToken(DELTA_TIMES[action])
        action -= SIZE_DELTA_TIME
        if action < len(self.subexprs):
            return ExpressionToken(self.subexprs[action])
        action -= len(self.subexprs)
        if action == 0:
            return SequenceIndicatorToken(SequenceIndicatorType.SEP)
        raise ValueError(f"Invalid discrete action index: {action}")

    def token_to_action(self, token: Token) -> int:
        if isinstance(token, OperatorToken):
            return OPERATORS.index(token.operator)
        offset = SIZE_OP
        if isinstance(token, FeatureToken):
            return offset + int(token.feature)
        offset += SIZE_FEATURE
        if isinstance(token, ConstantToken):
            return offset + CONSTANTS.index(token.constant)
        offset += SIZE_CONSTANT
        if isinstance(token, DeltaTimeToken):
            return offset + DELTA_TIMES.index(token.delta_time)
        offset += SIZE_DELTA_TIME
        if isinstance(token, ExpressionToken):
            expr_key = str(token.expression)
            for idx, expr in enumerate(self.subexprs):
                if str(expr) == expr_key:
                    return offset + idx
            raise ValueError(f"Expression token {expr_key} is not registered in subexprs")
        offset += len(self.subexprs)
        if (
            isinstance(token, SequenceIndicatorToken) and
            token.indicator == SequenceIndicatorType.SEP
        ):
            return offset
        raise ValueError(f"Unsupported token type for token_to_action: {type(token)}")


def LowLevelEnv(
    pool: AlphaPoolBase,
    subexprs: Optional[List[Expression]] = None,
    **kwargs
):
    return LowLevelEnvWrapper(LowLevelEnvCore(pool=pool, **kwargs), subexprs=subexprs)


AlphaEnvWrapper = LowLevelEnvWrapper
AlphaEnv = LowLevelEnv
