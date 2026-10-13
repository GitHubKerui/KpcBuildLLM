"""ax.barh - 关系与对比: 训练流水线各阶段耗时排行.

适用场景: 比较多个类别的数值大小, 尤其适合类别名称较长的场景(横向排列文字不会被挤成一团).
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
stages = [
    "Data loading & tokenization",
    "Forward pass",
    "Backward pass",
    "Optimizer step",
    "Validation",
    "Checkpoint saving",
    "Logging & metrics",
]
hours = np.array([12.4, 48.2, 61.7, 9.8, 22.5, 15.3, 3.1])

# 按耗时升序排列, 最长的排在最上面, 排行图才易读
order = np.argsort(hours)
stages_sorted = [stages[i] for i in order]
hours_sorted = hours[order]

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(8.5, 5))

cmap = mpl.colormaps["Blues"]
colors = cmap(np.linspace(0.42, 0.88, len(hours_sorted)))

""" ax_barh.py → ax.barh() (Horizontal Bar Chart)
• 命名逻辑：bar（条形图）+ h（Horizontal，水平）。
• 核心作用：绘制水平条形图。
• 核心参数：
	• y：条形在纵坐标上的中心位置。
	• width：条形的长度（即数据大小）。
	• height：条形本身的粗细（默认 0.8）。 """
bars = ax.barh(
    stages_sorted, hours_sorted,
    color=colors, edgecolor="white", linewidth=0.8, height=0.7,
)

# 在柱尾标注数值, 省去读者比对坐标轴
ax.bar_label(bars, fmt="%.1f h", padding=4, fontsize=9)

# 3. 图表美化与细节调整
ax.set_title("Training Pipeline Time Breakdown (100 epochs)", fontsize=13, pad=12)
ax.set_xlabel("Time (hours)")
ax.set_ylabel("Pipeline stage")

# 给 bar_label 留出空间, 否则最右侧的数值标签会被裁掉
ax.set_xlim(0, hours_sorted.max() * 1.18)

# 防标签重叠: y 轴类别名较长, 缩小字号并让画布更宽
ax.tick_params(axis="y", labelsize=9)
ax.margins(y=0.02)

ax.grid(True, linestyle="--", alpha=0.4, axis="x")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
