from typing import Optional
from torch import Tensor
from AlphaHRL.HRL.data.calculator import TensorAlphaCalculator
from AlphaHRL.HRL.data.expression import Expression
from AlphaHRL.HRL.utils.pytorch_utils import normalize_by_day
from AlphaHRL.alphahrl_qlib.stock_data import StockData


class QLibStockDataCalculator(TensorAlphaCalculator):
    def __init__(self, data: StockData, target: Optional[Expression] = None):
        super().__init__(normalize_by_day(target.evaluate(data)) if target is not None else None)
        self.data = data

    def evaluate_alpha(self, expr: Expression) -> Tensor:
        return normalize_by_day(expr.evaluate(self.data))
    
    @property
    def n_days(self) -> int:
        return self.data.n_days
