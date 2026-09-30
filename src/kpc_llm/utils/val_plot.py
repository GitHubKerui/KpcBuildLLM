import matplotlib.pyplot as plt
# 用于自动控制坐标轴刻度的显示方式（后面用来强制 X 轴只显示整数）。
from matplotlib.ticker import MaxNLocator


# epocks,tokens做两个x轴
# train_loss,val_loss做两个纵轴
def plot_loss(epochs,tokens,train_losses,val_losses):
    # 1. 创建图形画布figure，坐标轴axis1，宽 5 英寸，高3英尺
    figure,axis1 = plt.subplots(figsize=(5, 3))

    # 2. 先设置 ax1 的坐标轴标签
    axis1.set_xlabel("Epochs")
    axis1.set_ylabel("Loss")
    # 强制刻度为整数
    axis1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 2. 再在 ax1 上绘制线条 参数1： x=epochs,参数2： y=train_losses/val_losses,参数3： 曲线的名称
    axis1.plot(epochs, train_losses, label="Training loss")
    axis1.plot(epochs, val_losses, linestyle="-.", label="Validation loss")
    # 曲线的图例legend（图例必须在 plot后面）因为图例在plot画出来后才可以绘画
    # 定位数据线的图例在右上角
    axis1.legend(loc="upper right")  

    # 3. 创建并设置第二条 X 轴
    # twiny克隆了 ax1 的 Y 轴（共享相同的损失刻度），twinx()是复制X轴
    axis2 = axis1.twiny()
    axis2.set_xlabel("Tokens")
    # alpha=0透明，alpha=0.5：半透明，alpha=1：完全不透明
    axis2.plot(tokens, train_losses, alpha=0)  # 用于对齐刻度
    # 自动调整图表的边距和布局，因为我们加了双 X 轴，上方会有文字标签，如果不加这行，文字可能会被画布边缘裁剪掉
    figure.tight_layout()
    # 保存为高清晰度的 PDF 
    plt.savefig("loss-plot.pdf")
    # 屏幕上显示
    plt.show()


if "__main__" ==  __name__:
    epochs = [i for i in range(1,21)]
    tokens = [i * 5000 for i in range(1, 21)]
    train_losses = [  2.5 - (i * 0.1) for i in range(1, 21)]
    val_losses = [2.6 - (i * 0.08) for i in range(1, 21)]
    plot_loss(epochs,tokens,train_losses,val_losses)