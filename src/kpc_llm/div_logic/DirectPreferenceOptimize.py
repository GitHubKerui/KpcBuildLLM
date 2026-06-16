"""
DPO process used in RLHB
"""
import torch
import torch.nn.functional as F

# 为了保证结果和刚才的手算推演完全一致，我们设置打印精度
torch.set_printoptions(precision=3, sci_mode=False)

print("=== 1. 宇宙设定与张量初始化 ===")
# 特征张量 h (形状: [1, 2])
h = torch.tensor([[1.0, 0.5]])

# 目标 Token 的索引
y_w_idx = 1  # Chosen 优质回答 ("好")
y_l_idx = 2  # Rejected 劣质回答 ("坏")

# 基座/参考模型权重 W_ref (冻结, requires_grad=False)
W_ref = torch.tensor([
    [1.0, 0.0, -1.0],
    [0.0, 1.0,  1.0]
], requires_grad=False)

# 策略/优化模型权重 W_theta (需训练, 设置 requires_grad=True 追踪梯度)
W_theta = torch.tensor([
    [1.0, 0.5, -1.5],
    [0.0, 1.0,  0.5]
], requires_grad=True)

beta = 0.1  # KL 惩罚系数

print("\n=== 2. 参考模型 (W_ref) 的前向计算 ===")
with torch.no_grad(): # 参考模型不需要计算梯度
    logits_ref = torch.matmul(h, W_ref)
    log_probs_ref = F.log_softmax(logits_ref, dim=-1)
    
    # 使用索引提取对应 token 的 log-prob (Gather 过程)
    log_prob_ref_w = log_probs_ref[0, y_w_idx]
    log_prob_ref_l = log_probs_ref[0, y_l_idx]
    
    print(f"基座 Logits: {logits_ref.tolist()[0]}")
    print(f"基座 log_probs_ref: {log_probs_ref.tolist()}")
    print(f"基座 Chosen (Index 1) Log-Prob: {log_prob_ref_w.item():.3f}")
    print(f"基座 Rejected (Index 2) Log-Prob: {log_prob_ref_l.item():.3f}")

print("\n=== 3. 策略模型 (W_theta) 的前向计算 ===")
logits_theta = torch.matmul(h, W_theta)
log_probs_theta = F.log_softmax(logits_theta, dim=-1)

log_prob_theta_w = log_probs_theta[0, y_w_idx]
log_prob_theta_l = log_probs_theta[0, y_l_idx]

print(f"策略 Logits: {logits_theta.tolist()[0]}")
print(f"策略 log_probs_theta: {log_probs_theta.tolist()[0]}")
print(f"策略 Chosen (Index 1) Log-Prob: {log_prob_theta_w.item():.3f}")
print(f"策略 Rejected (Index 2) Log-Prob: {log_prob_theta_l.item():.3f}")

print("\n=== 4. 计算隐式奖励 (Implicit Reward) 与优势差 ===")
# DPO 核心：新旧 log-prob 差值乘以 beta 等价于 Reward
reward_w = beta * (log_prob_theta_w - log_prob_ref_w)
reward_l = beta * (log_prob_theta_l - log_prob_ref_l)
margin = reward_w - reward_l

print(f"Chosen 奖励: {reward_w.item():.3f}")
print(f"Rejected 奖励: {reward_l.item():.3f}")
print(f"Margin 优势差: {margin.item():.3f}")

print("\n=== 5. 计算 DPO Loss 与反向传播 ===")
# 使用 logsigmoid 提升数值稳定性： -log(sigmoid(margin))
loss = -F.logsigmoid(margin)
print(f"DPO Loss: {loss.item():.3f}")

# 触发反向传播计算梯度
loss.backward()

print("\n=== 6. 查看梯度更新方向 ===")
print("W_theta 的梯度 (Gradient):\n", W_theta.grad)