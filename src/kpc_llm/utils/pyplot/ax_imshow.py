"""ax.imshow - 空间与图像: 10 类分类任务的混淆矩阵热力图.

适用场景: 展示二维矩阵数据(混淆矩阵, 注意力权重, 相关性矩阵, 灰度图), 用颜色编码数值大小.
"""
import matplotlib.pyplot as plt
import numpy as np

# 1. 模拟数据生成
rng = np.random.default_rng(5)
classes = ["cat", "dog", "bird", "fish", "horse",
           "sheep", "cow", "elephant", "zebra", "giraffe"]
n = len(classes)

cm = rng.uniform(0, 0.05, (n, n))                    # 非对角线: 少量误判
np.fill_diagonal(cm, rng.uniform(0.72, 0.96, n))     # 对角线: 正确分类占多数
cm = cm / cm.sum(axis=1, keepdims=True)              # 按真实类别行归一化

# 2. 创建画布与面向对象绘图
fig, ax = plt.subplots(figsize=(8, 6.2))

""" ax_imshow.py → ax.imshow() (Image Show)
• 命名逻辑：im（Image，图像）+ show（展示）。
• 核心作用：将二维矩阵/数组渲染为热力图或直接显示图片。
• 核心参数：
	• X：二维数组（如高低起伏的数据）或三维 RGB 图像数据。
	• cmap：颜色映射表（Colormap），决定数值大小对应的颜色（如 'viridis', 'gray'）。
	• interpolation：插值方式（如 'nearest', 'bilinear'），决定图像放大时是像素风还是平滑风。 """
im = ax.imshow(cm, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label("Proportion (row-normalized)")

# 3. 图表美化与细节调整
ax.set_title("Confusion Matrix - 10-class Classifier", fontsize=13, pad=12)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")

# 防标签重叠: 类别名旋转 45 度并右对齐
ax.set_xticks(np.arange(n))
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
ax.set_yticks(np.arange(n))
ax.set_yticklabels(classes, fontsize=9)

# 只给数值够大的格子标注文字, 避免小格子里数字互相挤压
threshold = 0.35
for i in range(n):
    for j in range(n):
        value = cm[i, j]
        if value >= 0.04:
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if value > threshold else "black",
            )

# 用 minor ticks 画白色分隔线, 让格子边界更清晰
ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax.grid(which="minor", color="white", linewidth=0.8)
ax.tick_params(which="minor", length=0)

plt.tight_layout()
plt.show()
