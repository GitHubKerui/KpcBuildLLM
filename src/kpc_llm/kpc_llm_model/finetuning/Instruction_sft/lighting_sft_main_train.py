import argparse

from rich.traceback import install

from kpc_llm.kpc_llm_model.finetuning.Instruction_sft.prepare_dataloader import prepare_dataloader
install(show_locals=True)
import time
from lightning.pytorch.callbacks import ModelCheckpoint
from sympy import fraction
import torch
import lightning as L
from torch.utils.data import dataloader
from kpc_llm.kpc_llm_model.finetuning.Instruction_sft.lighting_InstructionSFT import LightingInstructionSFT
from kpc_llm.kpc_llm_model.finetuning.Instruction_sft.load_pre_gpt2_small import load_gpt2_small_model
from kpc_llm.utils.pyplot.plot_values import plot_values

# ===================== 训练前自检（非 Lightning） =====================


def check_loader(**loaders):
    """数据自检 1：形状、屏蔽比例、标签范围。
    有效标签占比必须明显小于 100%，否则说明 prompt 没被屏蔽、会泄漏标签。
    """
    for name,loader in loaders.items():
        inputs,targets = next(iter(loader))
        valid = int((targets != -100).sum())
        total = targets.numel()
        # 有效字符比例必须小于1 否则灭有屏蔽
        fraction = valid / total
        print(
            f"{name:5s} inputs {inputs.shape} targets {targets.shape}"
            f"有效标签比例: {fraction:.2%}"
            f"target标签 max_id = {targets.max()}  (应 <= 50257) "
        )

def check_model(lit_model:LightingInstructionSFT,loader,max_batches: int = 10):
    """模型权重自检 2：未训练时验证集 loss。
    预训练权重正常时应落在 2~5(-ln(1/e^2 ~ 1/e^5)；若 ≈ -ln(1/50257)≈10.8，说明权重没加载成功。
    这里的loss 也是困惑度的e的指数。正常是约等于 8~250个左右的字符选择范围
    """
    # lighting_model.parameters()第一个元素是生成器，第二个才是第一个参数（张量）
    device = next(lit_model.parameters()).device
    loss_sum, tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = (batch[0].to(device), batch[1].to(device))
        loss, _acc, n_tokens = lit_model._shared_train_step(batch)
        loss_sum += loss.item() * n_tokens
        tokens += n_tokens
    lit_model.train()
    loss = loss_sum / max(tokens, 1)
    return loss

def plot_history(history: dict[str, list[float]], num_epochs: int) -> None:
    """把历史画成 loss / token 准确率两条曲线（plot_values 内部会 savefig + show）。"""
    n = len(history["val_loss"])
    if n == 0:
        print("没有可绘制的验证记录（smoke 模式下可能未触发验证）")
        return
    examples_seen = history["examples"][-1]
    # .cpu()：linspace 落在「默认设备」上，一旦默认设备是 cuda，
    # matplotlib 转 numpy 会报 "can't convert cuda:0 device type tensor to numpy"
    epochs = torch.linspace(0, num_epochs, n).cpu()
    examples = torch.linspace(0, examples_seen, n).cpu()
    plot_values(epochs, examples, history["train_loss"], history["val_loss"], label="loss")
    plot_values(
        epochs, examples, history["train_token_acc"], history["val_token_acc"], label="token-acc"
    )

# ===================== 主流程 =====================

def main(smoke:bool = False,epochs_n:int=1):

    L.seed_everything(123)
    # 本机显卡带 Tensor Core，允许 TF32 换性能,这里存储是FP32，lighting后面混合精度会在前向时候自动替换
    torch.set_float32_matmul_precision("high")

    # 1) 加载模型 准备数据 初始化训练类 数据和权重基本检查
    model = load_gpt2_small_model()
    train_loader, val_loader, test_loader = prepare_dataloader()
    # 用Lightning 初始化model的全部训练类，fit的逻辑都在这个类里。
    lit_model = LightingInstructionSFT(model)
    # fit前检查数据和model
    check_loader(train=train_loader, val=val_loader, test=test_loader)
    base_loss = check_model(lit_model, val_loader)
    print(f"未训练 baseline: val loss {base_loss:.3f} (预训练权重正常应在 2~5)")


    # 2) checkpoint：只监控 val_loss，存最优一份
    # min 是loss 最小保存，max 是最大保存， verbose true是控制控制台打印。
    # 这个太方便了，原生的就这一坨费劲巴拉，频率太高还怕把硬盘搞坏了
    checkpoint = ModelCheckpoint(
        dirpath ="./checkpoints/lightning/instruction",
        filename="gpt2-small-sft-{epoch:02d}-{val_loss:.3f}",
        monitor = "val_loss",
        mode="min",
        save_top_k = 1,
        save_weights_only = True,
        verbose= True
    )

    # 3) Trainer：替代手写循环里的 autocast / optimizer.step / scheduler.step / 断点评估
    n_epochs = 1 if smoke else epochs_n

    trainer = L.Trainer(
        max_epochs = n_epochs,
        accelerator = "auto",                           #我去我的get_adaptive_device也废了，哈哈自动识别 GPU / CPU，替代手写 
        devices = 1,
        precision = "bf16-mixed",                        # 替代手写 torch.autocast(bf16)
                                                        # 也就是要把所有梯度总模长缩放到1.0 ，其实跟去掉bias的norm类似。但是保持结构比例不变 
        gradient_clip_val = 1.0,                        # 指令 SFT 首步 loss 波动大，裁剪可防梯度爆炸, 所有梯度向量和的总模长 *(1/所有梯度向量和的总模长)
                                                        # 下面这个决定是按上面的默认norm范数裁剪，还是 value 真裁剪超过value的值，所以注销了
                                                        # gradient_clip_algorithm = "norm",
        val_check_interval = 0.1,                       # 一个epoch的多少比例的batch评估一次，等于一个epoch评估10次。原生的方法又没用了
        limit_train_batches = 0.1 if smoke else 1.0,    #前面n_epochs在冒烟测试的时候只有10%的1个epoch数据量
        limit_val_batches = 1.0,                        # 验证集不大，全量评估
        callbacks =[checkpoint],                        #训练后的保存回调list
        enable_progress_bar = True,                     #打开进度条显示
        log_every_n_steps =1,                           #日志打印频率

    )

    start_time = time.time()
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    end_time = time.time()
    print(f"训练耗时 {(end_time - start_time) /60:.2f} 分钟" )
    print(f"最优模型 checkpoint ： {checkpoint.best_model_path}")

    # 4）用最优模型权重做test测试,真方便啊，不用自己原生写了
    trainer.test(model=lit_model, dataloaders=test_loader, ckpt_path=checkpoint.best_model_path)

    # 5） 画曲线图
    print(f"lit_model.history : {lit_model.history}")
    plot_history(lit_model.history, n_epochs)

if __name__ == "__main__":
    # 1. 创建参数解析器
    parser = argparse.ArgumentParser(description="运行我的脚本")
    # 2. 添加 smoke 参数 (使用 action='store_true'，只要命令行出现 --smoke 就视为 True)
    parser.add_argument(
        "--smoke", action="store_true", help="是否开启烟雾测试"
    )
    # 3. 添加 epochs_n 参数 (指定类型为 int，并设置默认值)
    parser.add_argument(
        "--epochs_n", type=int, default=1, help="训练的轮数 (Epochs)"
    )
    # 4. 解析控制台输入的参数
    args = parser.parse_args()
    # 5. 将解析后的参数传给 main 函数
    # main(smoke=args.smoke, epochs_n=args.epochs_n)
    main(False,5)
