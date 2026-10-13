"""ax.bar - 关系与对比: 三个模型在五个数据集上的准确率分组柱状图.

适用场景: 比较离散类别之间的数值大小. 类别名短、需要按时间/顺序阅读时用竖向柱状图;
类别名长或类别很多时改用 ax.barh(见 ax_barh.py).
"""
from collections.abc import Sequence

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb

_COLOR_SEED = 42   # 配色随机种子, 写死保证每次运行颜色一致


def _random_colors(count: int, seed: int = _COLOR_SEED) -> list[tuple[float, float, float]]:
    """按系列数量随机生成可区分的配色.

    在 HSV 空间均匀取色相后打乱顺序, 固定饱和度与亮度,
    避免直接随机 RGB 时出现灰暗或刺眼的颜色; 种子写死, 保证结果可复现.
    """
    rng = np.random.default_rng(seed)
    hues = rng.permutation(np.linspace(0.0, 1.0, count, endpoint=False))
    colors: list[tuple[float, float, float]] = []
    for hue in hues:
        r, g, b = hsv_to_rgb((hue, 0.55, 0.88))
        colors.append((float(r), float(g), float(b)))
    return colors


def plot_grouped_bar(
    dataGrp_names: Sequence[str],
    dataVal_names: Sequence[str],
    dataset: np.ndarray,
    *,
    figsize=(8.5, 5),
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    group_width: float = 0.8,
) -> Axes:
    """在传入的 ax 上绘制分组柱状图(Grouped Bar Chart)并完成美化.

    ax.bar() 核心参数:
    - x: 条形中心的横坐标(数值, 或 ['A', 'B', 'C'] 这类类别标签)
    - height: 条形高度, 即要展示的核心数据(对应 ax.barh 中的 width)
    - width: 条形本身的宽度(默认 0.8), 用来调整柱子的粗细
    - bottom: 条形的基线起始位置(默认 0), 设置它可以把柱子叠在另一个上方, 即堆叠柱状图
    - color / edgecolor: 条形的填充颜色 / 边框颜色

    Args:
        ax: 目标坐标轴, 由调用方创建(fig, ax = plt.subplots())
        categories: x 轴上的分组名称
        series_names: 每个分组内的系列名称, 即图例项
        values: 形状为 (len(series_names), len(categories)) 的二维数组
        title: 图表标题
        xlabel: x 轴标签
        ylabel: y 轴标签
        group_width: 单个分组占据的总宽度
        颜色: 由内部按系列数量随机生成, 随机种子固定(_COLOR_SEED), 无需外部传入

    Returns:
        传入的 ax, 方便调用方继续加工(如叠加参考线)
    """
    values = np.asarray(dataset)
    if values.shape != (len(dataGrp_names), len(dataVal_names)):
        raise ValueError(
            f"values 形状应为 (系列数, 分组数) = ({len(dataGrp_names)}, {len(dataVal_names)}), "
            f"实际为 {values.shape}; 检查 dataVal_names / dataGrp_names 是否传反了"
        )

    _fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(dataVal_names))
    width = group_width / len(dataGrp_names)   # 每组总宽按系列数平分
    colors = _random_colors(len(dataGrp_names))   # 按系列数量生成配色, 种子固定可复现
    

    for i, (name, series_values) in enumerate(zip(dataGrp_names, values, strict=False)):
        # 把每组柱子整体居中: 第 i 根柱子相对组中心的偏移
        offset = (i - len(dataGrp_names) / 2 + 0.5) * width

        bars = ax.bar(
            x + offset, series_values, width,
            label=name, color=colors[i],
            edgecolor="white", linewidth=0.8,
        )
        # 柱顶标注百分比数值(labels 传字符串列表, 避免 fmt 与 PercentFormatter 冲突)
        ax.bar_label(bars, labels=[f"{v * 100:.0f}" for v in series_values], padding=2, fontsize=8)

    # 3. 图表美化与细节调整
    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # 柱顶数值标签需要额外空间, 否则会被画布裁掉
    ax.set_ylim(0, float(values.max()) * 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    # 刻度回到每组的中心位置, 并旋转防重叠
    ax.set_xticks(x)
    ax.set_xticklabels(list(dataVal_names), rotation=15, ha="right")

    ax.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # 图例横排放在上方, 避免遮挡最高的柱子
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.98),
        ncol=len(dataGrp_names), frameon=False, fontsize=9,
    )
    plt.tight_layout()
    plt.show()

    return ax


def test_plot_grouped_bar() -> None:
   
    # 1. 模拟数据生成
    grp_names = ["ResNet-50", "ViT-B/16", "ConvNeXt-T"]
    val_names = ["ImageNet-1k", "COCO", "ADE20K", "CIFAR-100", "Food-101"]

    # 行=模型, 列=数据集, 取值 0~1(准确率)
    # datase shape = (grp_names,val_names)
    dataset = np.array([
    [0.761, 0.423, 0.412, 0.812, 0.703],
    [0.842, 0.481, 0.469, 0.901, 0.788],
    [0.822, 0.465, 0.452, 0.884, 0.771],
])

    plot_grouped_bar(
        grp_names,    # dataVal_names: 数据每组的名字dim=-2的名字
        val_names,    # dataGrp_names: 每组数据里每个数据的名字 dim =-1的name
        dataset,    # 形状必须是 (组的len, val的len) = (dim-2, dim-1)
        title="Top-1 Accuracy by Model and Dataset",
        xlabel="Dataset",
        ylabel="Top-1 accuracy",
    )


if __name__ == "__main__":
    test_plot_grouped_bar()
