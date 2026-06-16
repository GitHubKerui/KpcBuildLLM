"""
log_p 对数概率的log_softmax的底层处理办法。对于在e的超大指数的inf处理。
"""
import torch
import torch.nn.functional as F

# 这是一个极端的 Logits 输入
logits = torch.tensor([[1000.0, 2000.0, 3000.0]])
print(f"原始输入 Logits: {logits.tolist()}")

print("\n--- 朴素计算 (Naive Approach) 必定炸裂 ---")
# 计算指数：这里会溢出产生 inf
exp_naive = torch.exp(logits)
print(f"e^logits 结果: {exp_naive.tolist()}  <- 出现了 inf")

sum_exp_naive = torch.sum(exp_naive, dim=-1, keepdim=True)
print(f"Sum(e^logits): {sum_exp_naive.tolist()}  <- 依然是 inf")

# 算概率：inf / inf 会得到 nan
softmax_naive = exp_naive / sum_exp_naive
print(f"Softmax 概率: {softmax_naive.tolist()}  <- 炸裂成了 nan")

log_softmax_naive = torch.log(softmax_naive)
print(f"朴素 Log-Softmax 结果: {log_softmax_naive.tolist()} <- 彻底无法计算")


print("\n--- 稳定计算 (Log-Sum-Exp Approach) 安全着陆 ---")
M = torch.max(logits, dim=-1, keepdim=True).values
print(f"找到最大值 M: {M.tolist()}")

logits_shifted = logits - M
print(f"平移后的 Logits: {logits_shifted.tolist()} <- 全局转为非正数")

# 计算指数：负数次幂会安全下溢出为 0.0
exp_shifted = torch.exp(logits_shifted)
print(f"e^(z-M) 结果: {exp_shifted.tolist()} <- 极小值变成了 0.0，最大值是 1.0")

sum_exp_shifted = torch.sum(exp_shifted, dim=-1, keepdim=True)
print(f"Sum(e^(z-M)): {sum_exp_shifted.tolist()}")

log_sum_exp = torch.log(sum_exp_shifted)
log_softmax_stable = logits_shifted - log_sum_exp
print(f"稳定 Log-Softmax 结果: {log_softmax_stable.tolist()} <- 完美算出结果")

print("\n--- 对比 PyTorch 官方 API ---")
official_result = F.log_softmax(logits, dim=-1)
print(f"官方 F.log_softmax 结果: {official_result.tolist()}")