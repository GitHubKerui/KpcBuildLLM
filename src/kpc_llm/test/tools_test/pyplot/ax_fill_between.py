"""ax.fill_between - 区域填充与误差: 训练/验证 loss 的均值曲线与 ±1 标准差置信带.

适用场景: 在折线周围填充区间, 表达波动范围/置信带, 或高亮两条曲线之间的差异区域(如泛化间隙).
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
rng = np.random.default_rng(2024)
steps = np.linspace(0, 1000, 200)

train_mean = 2.5 * np.exp(-steps / 220) + 0.55 + rng.normal(0, 0.015, steps.size)
val_mean = train_mean + 0.06 + 0.12 * (1 - np.exp(-steps / 500)) + rng.normal(0, 0.02, steps.size)
train_std = 0.18 * np.exp(-steps / 400) + 0.05
val_std = 0.25 * np.exp(-steps / 500) + 0.08

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(steps, train_mean, color="#4C72B0", linewidth=2, label="Train")
ax.fill_between(
    steps, train_mean - train_std, train_mean + train_std,
    color="#4C72B0", alpha=0.25, linewidth=0, label="Train ±1σ",
)

ax.plot(steps, val_mean, color="#C44E52", linewidth=2, label="Validation")
ax.fill_between(
    steps, val_mean - val_std, val_mean + val_std,
    color="#C44E52", alpha=0.25, linewidth=0, label="Validation ±1σ",
)


""" ax_fill_between.py → ax.fill_between() (Fill Between)
• 命名逻辑：fill（填充）+ between（在……之间）。
• 核心作用：在两条曲线之间填充颜色，常用于表示置信区间或范围。
• 核心参数：
	• x：横坐标的定义域。
	• y1：第一条曲线的纵坐标。
	• y2：第二条曲线的纵坐标（默认是 0，即填充到 X 轴）。
	• where：一个布尔条件，可以指定只在特定区域内填充。
 """
# where= 只在满足条件的区间填充, 用来高亮泛化间隙
ax.fill_between(
    steps, train_mean, val_mean,
    where=val_mean > train_mean,
    color="gray", alpha=0.22, linewidth=0, label="Generalization gap",
)

# 3. 图表美化与细节调整
ax.set_title("Learning Curve with ±1σ Band", fontsize=13, pad=12)
ax.set_xlabel("Training steps")
ax.set_ylabel("Loss")

ax.set_xlim(steps.min(), steps.max())
ax.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 4 条图例项分两列显示, 避免纵向过长压住曲线
ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)

plt.tight_layout()
plt.show()
