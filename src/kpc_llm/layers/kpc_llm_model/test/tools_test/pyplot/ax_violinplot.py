"""ax.violinplot - 统计分布: 不同学习率下验证 loss 的概率密度形态.

适用场景: 比 boxplot 更进一步, 展示数据在整个取值区间的密度形状(多峰, 拖尾都能看出来).
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
rng = np.random.default_rng(123)
learning_rates = ["1e-4", "3e-4", "1e-3", "3e-3"]

# 学习率过小: loss 高但集中; 适中: 最低且集中; 过大: loss 高且方差大, 甚至双峰
losses = [
    rng.normal(loc=1.85, scale=0.09, size=400),
    rng.normal(loc=1.42, scale=0.07, size=400),
    rng.normal(loc=1.55, scale=0.13, size=400),
    np.concatenate([
        rng.normal(loc=1.95, scale=0.15, size=260),
        rng.normal(loc=2.65, scale=0.20, size=140),   # 训练不稳定的一簇, 形成双峰
    ]),
]

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(7.5, 4.8))

""" ax_violinplot.py → ax.violinplot() (Violin Plot)
    • 命名逻辑：因图表形状酷似小提琴而得名。
    • 核心作用：结合了箱线图和密度图，展示数据分布的概率密度。
    • 核心参数：
	• dataset：输入的数值数据。
	• positions：小提琴在轴上的分布位置。
	• showmeans / showmedians：是否显示均值或中位数。  """
parts = ax.violinplot(
    losses,
    showmeans=False,
    showmedians=True,    # 显示中位数
    showextrema=True,    # 显示最小/最大值的横线
    widths=0.85,
)

colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]
for body, color in zip(parts["bodies"], colors, strict=False):
    body.set_facecolor(color)
    body.set_alpha(0.75)
    body.set_edgecolor("black")
    body.set_linewidth(1)

for key in ("cmedians", "cmins", "cmaxes", "cbars"):
    parts[key].set_color("black")
    parts[key].set_linewidth(1.3)

# 3. 图表美化与细节调整
ax.set_title("Validation Loss Density by Learning Rate", fontsize=13, pad=12)
ax.set_xlabel("Learning rate")
ax.set_ylabel("Validation loss")

# 防标签重叠: 旋转 xticklabels
ax.set_xticks(range(1, len(learning_rates) + 1))
ax.set_xticklabels(learning_rates, rotation=15, ha="right")

ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    plt.Line2D([], [], color="black", linewidth=1.3, label="Median"),
    plt.Line2D([], [], color="black", linewidth=1.3, linestyle="", marker="|",
               markersize=12, label="Min / Max"),
]
ax.legend(handles=handles, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()
