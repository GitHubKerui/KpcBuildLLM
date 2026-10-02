"""ax.errorbar - 区域填充与误差: 三组实验的验证 loss 随 epoch 变化(带 95% 置信区间误差棒).

适用场景: 展示随自变量变化的趋势, 并在每个数据点上标出误差范围(标准差/置信区间), 说明差异是否显著.
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
rng = np.random.default_rng(11)
epochs = np.arange(1, 13)

# (初始 loss, 衰减速率): 正则化越强, 收敛后的 loss 越低
methods = {
    "Baseline": (2.60, 0.16),
    "Dropout 0.2": (2.50, 0.19),
    "Dropout 0.2 + WD": (2.35, 0.23),
}
colors = ["#4C72B0", "#55A868", "#C44E52"]

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(7.5, 5))

for (name, (start, decay)), color in zip(methods.items(), colors, strict=False):
    mean = start * np.exp(-decay * epochs) + 0.55 + rng.normal(0, 0.02, epochs.size)
    # 5 次不同随机种子跑出来的标准差, 后期训练越稳定标准差越小
    std = 0.05 + 0.09 * np.exp(-0.12 * epochs) + rng.uniform(0.01, 0.03)
    """ ax_errorbar.py → ax.errorbar() (Error Bar)
    • 命名逻辑：error（误差）+ bar（条/线），即误差棒图。
    • 核心作用：在折线图或数据点上叠加误差范围，展示数据的测量不确定度。
    • 核心参数：
	• x, y：数据点的位置。
	• xerr, yerr：水平或垂直方向的误差大小（可以是固定值或数组）。
	• fmt：数据点和连接线的样式。"""
    ax.errorbar(
        epochs, mean,
        yerr=1.96 * std,          # 95% 置信区间
        fmt="-o", color=color,
        capsize=4, capthick=1.2,  # 误差棒两端的横向短帽
        elinewidth=1.3, markersize=5,
        alpha=0.9, label=name,
    )

# 3. 图表美化与细节调整
ax.set_title("Validation Loss with 95% CI (5 random seeds)", fontsize=13, pad=12)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation loss")

# epoch 是离散整数, 强制只在这些位置打刻度
ax.set_xticks(epochs)

ax.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 图例放到右上, 并留白, 避免压住曲线末端
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
plt.show()
