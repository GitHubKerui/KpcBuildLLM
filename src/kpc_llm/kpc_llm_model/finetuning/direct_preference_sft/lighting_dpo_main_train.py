import argparse

from rich.traceback import install

install(show_locals=True)
import time
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
import lightning as L
from kpc_llm.kpc_llm_model.finetuning.Instruction_sft.load_pre_gpt2_small import load_sft_model
from kpc_llm.utils.pyplot.plot_values import plot_values
from kpc_llm.kpc_llm_model.finetuning.direct_preference_sft.lighting_dpo_train_module import LightingInstructionSFT
from kpc_llm.kpc_llm_model.finetuning.direct_preference_sft.dpo_data_loader import yeild_div_dataloader

# ===================== 训练前自检（非 Lightning） =====================


def check_loader(**loaders):
    """数据自检 1：形状、mask 比例、token id 范围。
    response 占比必须明显小于 100%，否则说明 prompt 没被 mask。
    """
    for name, loader in loaders.items():
        batch = next(iter(loader))
        chosen = batch["chosen"]
        mask = batch["chosen_mask"]
        print(
            f"{name:5s} chosen {tuple(chosen.shape)} rejected {tuple(batch['rejected'].shape)}"
            f"  response占比 {int(mask.sum()) / mask.numel():.2%}"
            f"  max_id = {int(chosen.max())} (应 <= 50256)"
        )

def check_model(lit_model:LightingInstructionSFT,loader,max_batches: int = 10):
    """模型权重自检 2：未训练时的 DPO loss。

    policy 与 reference 权重相同时两者 logprob 完全相等，logit=0，loss 必然 = ln2 ≈ 0.693。
    偏离这个数就说明 logprob 口径（shift / mask / 符号）有问题。
    """
    device = next(lit_model.parameters()).device
    loss_sum, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        # collate 把数据放在 GPU，但 fit 之前 LightningModule 可能还在 CPU，这里统一到模型所在设备。
        # prompt 在 collate 里没有被 stack，仍是 list（DPO 也用不到），所以只搬 tensor。
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, _pref_hit, _acc_hit, _n_tokens = lit_model._shared_train_step_dpo(batch)
        loss_sum += loss.item()
        n += 1
    lit_model.train()
    return loss_sum / max(n, 1)

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
    # DPO 看偏好准确率；token-acc 是 MLE 指标，DPO 下通常会下降（likelihood displacement）
    plot_values(
        epochs, examples, history["train_pref_acc"], history["val_pref_acc"], label="pref-acc"
    )

# ===================== 主流程 =====================
GPT2_SMALL = "gpt2-small-124M"
GPT2_MEDIUM = "gpt2-medium-355M"
GPT2_LARGE = "gpt2-large-774M"

# DPO 的起点是 SFT 之后的模型，不是原始预训练权重
SFT_CKPT = "checkpoints/lightning/instruction/gpt2-medium-355M-sft-epoch=02-val_loss=0.736.ckpt"

def main(smoke:bool = False,epochs_n:int=1):
    model_name = GPT2_MEDIUM         # 与 SFT checkpoint 的规格保持一致
    L.seed_everything(123)
    # 本机显卡带 Tensor Core，允许 TF32 换性能,这里存储是FP32，lighting后面混合精度会在前向时候自动替换
    torch.set_float32_matmul_precision("high")

    # 1) 加载 SFT 模型（policy 与 reference 各一份，权重相同） 准备数据 初始化训练类
    #    两份必须是独立实例：policy 会被训练，reference 必须保持 SFT 那一刻的权重
    model_policy = load_sft_model(SFT_CKPT, GPT2_MEDIUM + ".pth")
    model_ref = load_sft_model(SFT_CKPT, GPT2_MEDIUM + ".pth")
    train_loader,test_loader, val_loader = yeild_div_dataloader()
    # 用Lightning 初始化model的全部训练类，fit的逻辑都在这个类里。
    lit_model = LightingInstructionSFT(model_policy,model_ref)
    # fit前检查数据和model
    check_loader(train=train_loader, val=val_loader, test=test_loader)
    base_loss = check_model(lit_model, val_loader)
    print(f"未训练 baseline: DPO loss {base_loss:.3f} (policy 与 ref 相同，应 ≈ 0.693)")


    # 2) checkpoint：只监控 val_loss，存最优一份
    # min 是loss 最小保存，max 是最大保存， verbose true是控制控制台打印。
    # 这个太方便了，原生的就这一坨费劲巴拉，频率太高还怕把硬盘搞坏了
    checkpoint = ModelCheckpoint(
        dirpath ="./checkpoints/lightning/dpo",
        filename= model_name + "-dpo-{epoch:02d}-{val_loss:.3f}",
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
    plot_history(lit_model.history, n_epochs)

if __name__ == "__main__":
    main(False,5)
