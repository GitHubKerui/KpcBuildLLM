"""ax.hist - 统计分布: 两个班级的考试成绩分布对比.

适用场景: 查看单个数值变量的取值分布(是否正态/偏态/双峰), 或比较多组数据的分布形态.

说明: 脚本统一使用英文标签, 避免不同系统缺少中文字体导致的方块乱码.
如需中文标签, 在 import matplotlib 后加两行:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# 1. 模拟数据生成
rng = np.random.default_rng(42)
class_a = np.clip(rng.normal(loc=72, scale=10, size=320), 0, 100)  # A 班: 均值低, 更分散
class_b = np.clip(rng.normal(loc=80, scale=7, size=280), 0, 100)   # B 班: 均值高, 更集中

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(7, 4.5))

""" ax_hist.py → ax.hist() (Histogram)
• 命名逻辑：Histogram 的缩写，中文叫直方图（或频数分布图）。
• 核心作用：把数据分成若干连续的区间，统计落入每个区间的数据量。
• 核心参数：
	• x：输入的一维数值数据。
	• bins：分箱（柱子）的数量或区间边界（例如 bins=10 表示分成 10 组）。
	• density：若为 True，纵轴将显示概率密度而不是绝对频数。 """
# 两组共用同一套分箱边界, 否则柱子的宽度/位置不一致, 视觉上无法比较
bins = np.linspace(40, 100, 21)
ax.hist(
    class_a, bins=bins, alpha=0.65, color="#4C72B0",
    edgecolor="white", linewidth=0.8, label=f"Class A (n={class_a.size})",
)
ax.hist(
    class_b, bins=bins, alpha=0.65, color="#DD8452",
    edgecolor="white", linewidth=0.8, label=f"Class B (n={class_b.size})",
)

# 3. 图表美化与细节调整
ax.set_title("Exam Score Distribution by Class", fontsize=13, pad=12)
ax.set_xlabel("Score")
ax.set_ylabel("Number of Students")

# 人数强制整数刻度, 避免出现 12.5 人这种无意义的刻度值
ax.yaxis.set_major_locator(MaxNLocator(integer=True))
ax.set_xlim(40, 100)

ax.grid(True, linestyle="--", alpha=0.5, axis="y")
ax.set_axisbelow(True)  # 网格线画在数据下面, 避免遮挡柱体
ax.legend(loc="upper left", frameon=False)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
