"""ax.scatter - 关系与对比: 参数量 vs 吞吐量的关系(颜色=精度, 点大小=显存占用).

适用场景: 观察两个数值变量之间的相关性/聚类/离群点, 并可额外用颜色和大小编码第三, 第四个维度.
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
rng = np.random.default_rng(2024)
n = 90
params = rng.uniform(0.1, 7.0, n)                                # 参数量 (B)
throughput = 1500 / (params ** 0.85) * rng.normal(1.0, 0.09, n)  # tokens/s, 大体与参数量成反比
accuracy = np.clip(0.55 + 0.095 * np.log1p(params) + rng.normal(0, 0.022, n), 0, 1)
gpu_memory = 2 + 1.7 * params + rng.normal(0, 0.6, n)            # 显存 (GB)

SIZE_SCALE = 18.0   # 显存 GB -> 散点面积 的换算系数(后面图例要用)

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(8, 5.2))

""" ax_scatter.py → ax.scatter() (Scatter Plot)
• 命名逻辑：scatter 意为散开、分散，即散点图。
• 核心作用：展示两个变量之间的相关性或分布规律。
• 核心参数：
	• x, y：点的横纵坐标数据。
	• s：点的大小（Size）。
	• c：点的颜色（Color）。
	• marker：点的形状（如圆形 'o'、正方形 's'）。 """
sc = ax.scatter(
    params, throughput,
    c=accuracy, s=gpu_memory * SIZE_SCALE,
    cmap="viridis", alpha=0.85,
    edgecolors="white", linewidths=0.6,
)

cbar = fig.colorbar(sc, ax=ax, pad=0.02)
cbar.set_label("Accuracy")

# 3. 图表美化与细节调整
ax.set_title("Params vs Throughput (size = GPU memory, color = accuracy)", fontsize=12.5, pad=12)
ax.set_xlabel("Parameters (B)")
ax.set_ylabel("Throughput (tokens/s)")

ax.grid(True, linestyle="--", alpha=0.4)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# 气泡大小图例: 手动构造, 比 legend_elements 更可控(它的默认标签是 mathtext 形式)
mem_levels = [4, 8, 12, 16]
size_handles = [
    ax.scatter([], [], s=lv * SIZE_SCALE, c="gray", alpha=0.65,
               edgecolors="white", linewidths=0.6)
    for lv in mem_levels
]
ax.legend(size_handles, [f"{lv} GB" for lv in mem_levels],
          title="GPU memory", loc="upper right",
          frameon=False, fontsize=8, title_fontsize=8)

plt.tight_layout()
plt.show()
