from alphagen.data.expression import Feature, Ref, Div, Sub, Constant
from alphagen_qlib.stock_data import FeatureType

# 创建 close
close = Feature(FeatureType.CLOSE)
print(f"close 类型：{type(close).__name__}")
# 输出：close 类型：Feature

# 创建 Ref
ref_obj = Ref(close, -20)
print(f"\nref_obj 类型：{type(ref_obj).__name__}")
# 输出：ref_obj 类型：Ref

print(f"ref_obj._operand 类型：{type(ref_obj._operand).__name__}")
# 输出：ref_obj._operand 类型：Feature

print(f"ref_obj._delta_time: {ref_obj._delta_time}")
# 输出：ref_obj._delta_time: -20

# 创建除法
div_obj = ref_obj / close
print(f"\ndiv_obj 类型：{type(div_obj).__name__}")
# 输出：div_obj 类型：Div

print(f"div_obj._lhs 类型：{type(div_obj._lhs).__name__}")
# 输出：div_obj._lhs 类型：Ref

print(f"div_obj._rhs 类型：{type(div_obj._rhs).__name__}")
# 输出：div_obj._rhs 类型：Feature

# 创建最终表达式
target = div_obj - 1
print(f"\ntarget 类型：{type(target).__name__}")
# 输出：target 类型：Sub

print(f"target._lhs 类型：{type(target._lhs).__name__}")
# 输出：target._lhs 类型：Div

print(f"target._rhs 类型：{type(target._rhs).__name__}")
# 输出：target._rhs 类型：Constant

# 打印表达式字符串
print(f"\ntarget 的字符串表示：{target}")
# 输出：target 的字符串表示：Sub(Div(Ref($close,20d),$close),1.0)


print("==========================")
# 创建表达式
close = Feature(FeatureType.CLOSE)
target = Ref(close, -20) / close - 1


print(f"target 类型：{type(target).__name__}")
# 输出：Sub

print(f"\ntarget._lhs 类型：{type(target._lhs).__name__}")
# 输出：Div

print(f"target._rhs 类型：{type(target._rhs).__name__}")
# 输出：Constant

print(f"\ntarget._lhs._lhs 类型：{type(target._lhs._lhs).__name__}")
# 输出：Ref

print(f"target._lhs._rhs 类型：{type(target._lhs._rhs).__name__}")
# 输出：Feature

print(f"\ntarget._lhs._lhs._operand 类型：{type(target._lhs._lhs._operand).__name__}")
# 输出：Feature

print(f"target._rhs.value: {target._rhs.value}")
# 输出：1.0

print(f"\n===== 在调用 evaluate 之前，结构已经确定 =====")
print(f"target 字符串：{target}")
# 输出：Sub(Div(Ref($close,20d),$close),1.0)

