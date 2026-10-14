"""GPT-2 large 垃圾短信二分类（SFT）的 Lightning 实现。

与手写循环 train_main_loop.py 的对应关系：
    手写 for batch 里 optimizer.step() / scheduler.step()   → Trainer + configure_optimizers
    with torch.autocast(bf16)                              → Trainer(precision="bf16-mixed")
    每 10% 步调 eval_by_step 评估                            → Trainer(val_check_interval=0.1)
    eval_freq / eval_btc_nun 两个抽样参数                    → val_check_interval / limit_val_batches
    每 epoch 调 accuracy_by_step                             → 模块内累加，验证时一并上报
    train_losses / val_losses 等历史列表                      → 模块自带的 history
    手写保存模型                                             → ModelCheckpoint
"""

import time
from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from transformers import get_cosine_schedule_with_warmup

from kpc_llm.kpc_llm_model.finetuning.classify_sft.load_pre_gpt2 import load_GPT2_model
from kpc_llm.kpc_llm_model.finetuning.classify_sft.prepare_cls_fintuning_model import (
    prepare_finetuning_model,
)
from kpc_llm.kpc_llm_model.finetuning.classify_sft.prepare_dataloader import prepare_dataloader
from kpc_llm.utils.pyplot.plot_values import plot_values


class GPT2SpamClassifier(L.LightningModule):
    """GPT-2 large + 二分类头。

    只训练 out_head、最后一个 transformer block 和 final_norm（约 19.7M / 774M 参数），
    其余冻结，由 prepare_finetuning_model 完成。
    """

    hparams: Any  # save_hyperparameters 动态注入，声明一下供类型检查使用

    def __init__(
        self,
        model,
        lr: float = 5e-5,
        weight_decay: float = 0.1,
        warmup_frac: float = 0.15,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])  # 只存超参，不存 774M 权重
        self.model = model

        # 单批 loss 抖动大，用 EMA 平滑；比滑动窗口简单，曲线也更稳
        self.train_loss_ema: float | None = None

        # 画图用的历史。刻意由模块自己维护：Callback 与模块钩子的调用先后顺序在 Lightning 里
        # 不直观，指标只在一个地方记录最不容易出错。
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "examples": [],
        }
        self.examples_seen = 0
        self._reset_train_stats()
        self._reset_val_stats()

    # ================= 前向与损失 =================

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        # 因果注意力下最后一个 token 能看到全部上下文，特征最全，用它做分类
        return self.model(input_batch)[:, -1, :]

    def _shared_step(self, batch) -> tuple[torch.Tensor, int, int]:
        """返回 (loss, 本批预测正确的样本数, 本批样本数)。"""
        input_batch, target_batch = batch
        logits = self(input_batch)
        loss = F.cross_entropy(logits, target_batch)
        correct = int((logits.argmax(dim=-1) == target_batch).sum())
        return loss, correct, target_batch.numel()

    # ================= 训练 =================

    def _reset_train_stats(self) -> None:
        self._trn_correct = 0
        self._trn_total = 0

    def on_train_epoch_start(self) -> None:
        self._reset_train_stats()

    def training_step(self, batch, batch_idx):
        loss, correct, total = self._shared_step(batch)
        self._trn_correct += correct
        self._trn_total += total
        self.examples_seen += total
        self.train_loss_ema = (
            loss.item()
            if self.train_loss_ema is None
            else 0.93 * self.train_loss_ema + 0.07 * loss.item()
        )
        # 进度条看逐批的原始 loss，画曲线用平滑后的 ema
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss_ema", self.train_loss_ema, on_step=True, on_epoch=False)
        return loss

    # ================= 验证 =================

    def _reset_val_stats(self) -> None:
        self._val_loss_sum = 0.0
        self._val_correct = 0
        self._val_total = 0

    def on_validation_epoch_start(self) -> None:
        # val_check_interval<1 时一个训练 epoch 内会验证多次，每次都要清零，
        # 否则「本次验证的均值」会变成历次验证的累计平均，曲线严重滞后。
        self._reset_val_stats()

    def validation_step(self, batch, batch_idx):
        loss, correct, total = self._shared_step(batch)
        self._val_loss_sum += loss.item() * total
        self._val_correct += correct
        self._val_total += total
        return loss

    def on_validation_epoch_end(self) -> None:
        if self.trainer.sanity_checking:  # 训练前的 sanity check 不进历史
            return

        val_loss = self._val_loss_sum / self._val_total
        val_acc = self._val_correct / self._val_total
        # 这个钩子只允许 on_epoch=True（fx_validator 的硬性要求）。
        # 好在 Lightning 每次验证结束都会 reset_results()，所以聚合窗口就是「本次验证」，
        # val_check_interval=0.1 时每个评估点拿到的都是当次均值，不会累计滞后。
        self.log_dict(
            {"val_loss": val_loss, "val_acc": val_acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        # 历史由模块自己维护：模块钩子执行时 callback_metrics 还没刷新，
        # 放到 Callback 里读又要依赖回调与模块钩子的先后顺序，就地记录最省心。
        assert self.train_loss_ema is not None  # 能走到这里说明已经跑过训练 batch
        self.history["train_loss"].append(self.train_loss_ema)
        self.history["val_loss"].append(val_loss)
        self.history["train_acc"].append(self._trn_correct / self._trn_total)
        self.history["val_acc"].append(val_acc)
        self.history["examples"].append(self.examples_seen)

    def on_train_epoch_end(self) -> None:
        # 到 epoch 末尾才把训练准确率上报，对应旧 accuracy_by_step 的「每 epoch 算一次」
        self.log("train_acc", self._trn_correct / self._trn_total,
                 on_step=False, on_epoch=True, prog_bar=True)

    # ================= 测试 =================

    def test_step(self, batch, batch_idx):
        loss, correct, total = self._shared_step(batch)
        # 测试只跑一轮，on_epoch=True 就是整轮均值；test_loader 没有 drop_last，
        # 末批更小，所以要按样本数加权
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=total)
        self.log("test_acc", correct / total, on_step=False, on_epoch=True, batch_size=total)
        return loss

    # ================= 优化器与学习率 =================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # 必须是含所有 epoch 的总步数：只写单个 epoch 的批次数，
        # 会让学习率在第 2 个 epoch 起恒为 0（旧 warmup_cosin_annealing 的坑）
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.hparams.warmup_frac),
            num_training_steps=total_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

    # ================= checkpoint =================

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """只保存可训练参数。

        SFT 只训练了 19.7M / 774M 参数，冻结的 754M 每次原样写一遍要 3.1GB；
        而 Lightning 的 _atomic_save 会先把整个 ckpt torch.save 到 BytesIO、
        再 getvalue() 复制一份，保存瞬间产生约 10GB 内存尖峰，会把系统提交量打满，
        在 Windows 上被误报成 OSError: [Errno 28] No space left on device。
        过滤后 ckpt 从 3.2GB 降到约 80MB。
        """
        full_state = checkpoint["state_dict"]
        trainable = {n for n, p in self.model.named_parameters() if p.requires_grad}
        # LightningModule 的 state_dict 键带 "model." 前缀（self.model = model），比对前剥掉
        checkpoint["state_dict"] = {
            k: v for k, v in full_state.items() if k.removeprefix("model.") in trainable
        }

    def load_trainable_weights(self, state_dict: dict) -> None:
        """把 ckpt 里的可训练参数灌回模型。

        冻结的 754M 不在 ckpt 里，保持 load_GPT2_model() 加载的预训练值，语义上正确，
        所以故意不用 strict=True —— missing 里全是冻结层，属预期。
        """
        weights = {k.removeprefix("model."): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(weights, strict=False)
        if unexpected:
            raise ValueError(f"ckpt 里存在模型没有的参数：{unexpected}")
        print(f"载入可训练参数 {len(weights)} 个；ckpt 中不存在的 {len(missing)} 个（均为冻结层）")


# =====================================================================
# 数据与非 Lightning 的自检
# =====================================================================

def check_dataset(**loaders) -> None:
    """自检：任一 split 变成单一类别，说明切分有偏，指标不可信。"""
    for name, loader in loaders.items():
        labels = [int(y) for _, y in loader.dataset]
        spam = sum(labels)
        print(f"{name:5s} spam: {spam:4d}  ham: {len(labels) - spam:4d}  总数: {len(labels)}")


@torch.no_grad()
def check_baseline(model, loader, device) -> tuple[float, float]:
    """自检：未训练时分类头是随机初始化，二分类 loss 应接近 ln2≈0.693、准确率接近 50%。"""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for input_batch, target_batch in loader:
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        logits = model(input_batch)[:, -1, :]
        loss_sum += F.cross_entropy(logits, target_batch).item() * target_batch.numel()
        correct += int((logits.argmax(dim=-1) == target_batch).sum())
        total += target_batch.numel()
    model.train()
    return loss_sum / total, correct / total


def plot_history(history: dict[str, list[float]], num_epochs: int) -> None:
    """把历史画成 loss / acc 两条曲线，plot_values 内部会 savefig + show。"""
    examples_seen = history["examples"][-1]
    # .cpu()：linspace 落在「默认设备」上，一旦默认设备是 cuda，
    # matplotlib 转 numpy 时会报 "can't convert cuda:0 device type tensor to numpy"
    for label, train_key, val_key in (
        ("loss", "train_loss", "val_loss"),
        ("accuracy", "train_acc", "val_acc"),
    ):
        epochs = torch.linspace(0, num_epochs, len(history[train_key])).cpu()
        examples = torch.linspace(0, examples_seen, len(history[train_key])).cpu()
        plot_values(epochs, examples, history[train_key], history[val_key], label=label)


# =====================================================================
# 主流程
# =====================================================================

def main(smoke: bool = False) -> None:
    # 一次性固定 python / numpy / torch / cuda 的随机性。
    # 分类头初始化是否可复现很关键：不固定种子时，某些随机头会让训练前 100 步不降反升。
    L.seed_everything(123)
    # 本机显卡带 Tensor Core，允许 TF32 换性能（Lightning 启动时也会提示）
    torch.set_float32_matmul_precision("high")

    model = prepare_finetuning_model(load_GPT2_model())
    train_loader, val_loader, test_loader = prepare_dataloader()
    device = next(model.parameters()).device

    check_dataset(train=train_loader, val=val_loader, test=test_loader)
    base_loss, base_acc = check_baseline(model, val_loader, device)
    print(f"未训练 baseline: val loss {base_loss:.3f}（二分类随机应接近 0.693）, "
          f"val acc {base_acc:.2%}（应接近 50%）")

    lit_model = GPT2SpamClassifier(model)
    checkpoint = ModelCheckpoint(
        dirpath="./checkpoints/lightning",
        filename="gpt2-spam-best-{epoch:02d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True,  # 不存 optimizer/lr 状态，只有断点续训才需要
        verbose=True,
    )

    num_epochs = 1 if smoke else 5
    trainer = L.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",        # 自动识别 GPU / CPU，替代手写的 get_adaptive_device
        devices=1,
        precision="bf16-mixed",    # 替代手写的 torch.autocast(bf16)
        val_check_interval=0.1,    # 每训练 10% 的 batch 验证一次
        limit_train_batches=0.1 if smoke else 1.0,
        # 验证集只有 14 个 batch，全量评估比抽 50% 更稳，且耗时可以忽略
        limit_val_batches=1.0,
        callbacks=[checkpoint],
        enable_progress_bar=True,
        log_every_n_steps=1,
    )

    start_time = time.time()
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"训练耗时 {(time.time() - start_time) / 60:.2f} 分钟")
    print(f"最优 checkpoint: {checkpoint.best_model_path}")

    # 用最优 checkpoint 的可训练参数做测试集闭卷评估。
    # 不用 trainer.test(ckpt_path="best")：它走 LightningModule.load_from_checkpoint，
    # 默认 strict=True，而 on_save_checkpoint 只存了 19.7M 可训练参数，会报大量 missing keys。
    best = torch.load(checkpoint.best_model_path, map_location="cpu", weights_only=True)
    lit_model.load_trainable_weights(best["state_dict"])
    trainer.test(lit_model, dataloaders=test_loader)

    plot_history(lit_model.history, num_epochs)


if __name__ == "__main__":
    import sys
    main(smoke="--smoke" in sys.argv)
