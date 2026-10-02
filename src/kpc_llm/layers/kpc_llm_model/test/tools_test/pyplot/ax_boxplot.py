"""ax.boxplot - 统计分布: 四个模型推理延迟的分布对比(中位数/四分位/离群点).

适用场景: 比较多组数据的分布特征(中位数, 离散程度, 异常值), 同时保留离群点信息.
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
rng = np.random.default_rng(7)
models = ["ResNet-18", "ResNet-50", "ViT-B/16", "ViT-L/16"]
# 对数正态分布模拟右偏的延迟数据(长尾是延迟类指标的典型特征)
latency = [
    rng.lognormal(mean=2.0, sigma=0.25, size=220),
    rng.lognormal(mean=2.8, sigma=0.28, size=220),
    rng.lognormal(mean=3.4, sigma=0.30, size=220),
    rng.lognormal(mean=4.2, sigma=0.35, size=220),
]

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(7.5, 4.8))


""" ax_boxplot.py → ax.boxplot() (Box Plot)
• 命名逻辑：box（箱子）+ plot（图），直译为箱线图。
• 核心作用：展示一组数据的分布（中位数、四分位数、异常值等）。
• 核心参数：
	• x：输入的数据（数组或向量序列）。
	• vert：是否垂直放置。默认 True（垂直），若设为 False 则横向。
	• notch：是否在中位数处切出凹口（展示置信区间）。
"""
box = ax.boxplot(
    latency,
    patch_artist=True,   # 允许填充箱体颜色
    showmeans=True,      # 额外显示均值(菱形)
    widths=0.6,
    medianprops=dict(color="black", linewidth=2),
    meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black", markersize=6),
    flierprops=dict(marker="o", markersize=4, alpha=0.35, markeredgecolor="none"),
)

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
for patch, color in zip(box["boxes"], colors, strict=False):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)

# 3. 图表美化与细节调整
ax.set_title("Inference Latency Distribution by Model", fontsize=13, pad=12)
ax.set_xlabel("Model")
ax.set_ylabel("Latency (ms)")

# 防标签重叠: 模型名较长, 旋转 15 度并右对齐
ax.set_xticks(range(1, len(models) + 1))
ax.set_xticklabels(models, rotation=15, ha="right")

ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

handles = [
    plt.Line2D([], [], color="black", linewidth=2, label="Median"),
    plt.Line2D([], [], marker="D", linestyle="", markerfacecolor="white",
               markeredgecolor="black", markersize=7, label="Mean"),
]
ax.legend(handles=handles, loc="upper left", frameon=False)

plt.tight_layout()
plt.show()
