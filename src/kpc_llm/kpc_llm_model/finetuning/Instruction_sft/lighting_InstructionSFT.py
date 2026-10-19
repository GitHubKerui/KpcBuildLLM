"""GPT-2 small 指令 SFT 的 Lightning 实现。

数据契约（来自 prepare_dataloader.custom_collate_fn）：
    inputs  [B, T] int64   已左移一位的输入
    targets [B, T] int64   prompt 段 / padding 段为 -100，仅 response 段为有效 token

与参考实现 classify_sft/GPT2SFT.py 的差异（只这四处）：
    forward  : 返回整段 logits [B,T,V]      （参考实现取 [:, -1, :]）
    loss     : 展平 + ignore_index=-100     （参考实现是类别交叉熵）
    指标     : token 准确率 + perplexity    （参考实现是类别准确率）
    模型     : 保持 out_head(50257)、全参数可训练（参考实现换成 Linear(1280,2) 并冻结主干）
"""

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from kpc_llm.utils import getlogger

log = getlogger()
class LightingInstructionSFT(L.LightningModule):
    """ GPT-2 small 124M 指令 SFT：只对 response 段算 loss。其他地方-100 padding使梯度为0，不算 loss。"""

    hparams: Any # save_hyperparameters 动态注入，声明供类型检查使用

    def __init__(self,model,lr:float=5e-5,weight_decay:float=0.1,warmup_frac:float=0.15,ignore_index=-100):
        super().__init__()
        self.save_hyperparameters(ignore=["model"]) # self.hparams只存超参，不存 124M 权重，就是存在类头生命的那个hparams
        self.model = model
        self.ignore_index = ignore_index

        # 单批 loss 抖动大，用 EMA 平滑，比滑动窗口简单且曲线更稳
        self.train_loss_ema: float | None = None

        # 画图历史刻意由模块自身维护：Lightning 里 callback 与模块钩子的先后顺序不直观，
        # 指标只在一个地方记录最不容易出错。
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_token_acc": [],
            "val_loss": [],
            "val_ppl": [],# 参考实现是 perplexity 困惑度可有可无啊
            "val_token_acc": [],
            "examples": [],
        }
        # 样本训练数等于batch_size的和
        self.example_seen = 0
        self._reset_train_stats()
        self._reset_val_stats()

    # ================= 前向与损失 =================

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        # 指令 SFT 要每个位置的 logits，才能逐 token 算 next-token 预测误差。
        return self.model(input_batch)  #输出 [B, T, vocab]

    def _shared_train_step(self, batch) -> tuple[torch.Tensor, int, int]:
        """返回 (loss, 预测逻辑值, 最后一个token的logits)。"""
        inputs, targets = batch
        """ ***** 这里就是model(inputs) *****  """
        logits = self(inputs)                 # [B, T, V]
        vocab = logits.size(-1)

        # 必须把 [B, T, V] 展平成 [B*T, V]、target 展平成 [B*T]：
        # cross_entropy 只接受 (N, C) 与 (N,)，维度不对会报 "Expected target size"。
        logits_flat = logits.reshape(-1, vocab)
        targets_flat = targets.reshape(-1) 
        
        # ignore_index=-100：prompt 段与 padding 段不产生梯度，这是指令 SFT 的核心。
        loss = F.cross_entropy(logits_flat,targets_flat,ignore_index=self.ignore_index)

        # 准确率只在有效位置上算，pad位置不算，最后的结束endoftoken要保留，学习文章结束
        mask = targets_flat != self.ignore_index
        preds = logits_flat.argmax(-1) # 长度 = B*T = 512（所有 token，含 padding/prompt）
        # 把真的true ==1 然后相加来算正确的次数（preds 也要按 mask 筛，否则长度对不上）
        acc = int((preds[mask] == targets_flat[mask]).sum()) # targets_flat 长度 = 有效 token 数 = 134（只保留非 -100）
        # 真正的训练返回元祖
        return loss ,acc,int(mask.sum())

    # ================= 训练 =================
    
    def _reset_train_stats(self):
        # 重置训练统计数据
        self._trn_loss_sum = 0.0
        self._trn_acc = 0
        self._trn_tokens  = 0

    def on_train_epoch_start(self) -> None:
        self._reset_train_stats()

    def training_step(self, batch, batch_idx) :
        # 框架自动调用 model.eval()
        loss,acc,n_tokens = self._shared_train_step(batch)
        # token 数加权：不同批的有效 token 数不同，直接平均会偏袒短批
        self._trn_loss_sum += loss.item() * n_tokens
        self._trn_acc += acc
        self._trn_tokens += n_tokens
        self.example_seen += int(batch[0].shape[0])
        # 第一次调用 train_loss_ema is None 会获得loss，后面获得的都是加权平均。
        # 0.93 换算下来 约等于 15次数据的加权平均。只是约等于
        # 注意 (1-0.93)=0.07 必须写：漏了的话稳态 ema = loss/0.07，数值被放大 14 倍
        self.train_loss_ema = ( 
            loss.item()
            if self.train_loss_ema is None 
            else 0.93 * self.train_loss_ema + 0.07 * loss.item())
        # Lightning 框架进度条 用 self.log
        self.log("train_loss",loss,on_step=True,on_epoch=False,prog_bar=True)
        self.log("train_loss_ema",self.train_loss_ema,on_step=True,on_epoch=False)
        # 这里不需要写 loss.backward,也不用optimizer.step框架会处理
        return loss
    
    def on_train_epoch_end(self) -> None:
        if self._trn_tokens == 0:
            return
        self.log(
            "train_token_loss_epoch",
            self._trn_loss_sum / self._trn_tokens,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # ================= 验证 =================

    def _reset_val_stats(self):
        self._val_loss_sum = 0.0
        self._val_acc = 0
        self._val_tokens = 0

    def on_validation_epoch_start(self) -> None:
        # val_check_interval<1,一个epoch 验证次数== 1/val_check_interval,每次都要清零，
        # 否则「本次验证均值」会变成历次累计平均，曲线严重滞后。
        self._reset_val_stats()

    def validation_step(self, batch, batch_idx):
        # 获取验证，loss,acc,n_tokens,lightning 框架自动调用 model.eval()
        loss,acc,n_tokens = self._shared_train_step(batch)
        self._val_loss_sum += loss.item() * n_tokens
        self._val_acc += acc
        self._val_tokens += n_tokens
        return loss

    def on_validation_epoch_end(self) -> None:
        # 训练前的sanity check 的bool标志 ，如果是这个阶段则不进指标历史统计数据， (训练前会试跑默认2个批次来验证整体正常度)
        if self.trainer.sanity_checking:     
            return
        # 如果验证集为空，则不记录
        if self._val_tokens == 0 or self.train_loss_ema is None: 
            return
        # 先算均值再算困惑度：val_ppl 依赖 val_loss
        val_loss = self._val_loss_sum / self._val_tokens
        val_acc = self._val_acc / self._val_tokens
        val_ppl = float(torch.exp(torch.tensor(val_loss)))
        # 记录指标历史
        self.history["train_loss"].append(self.train_loss_ema)
        # 训练集的token准确率,max函数防止分母为0
        self.history["train_token_acc"].append(
            self._trn_acc / max(self._trn_tokens, 1)
        )
        self.history["val_loss"].append(val_loss)
        self.history["val_ppl"].append(val_ppl)
        self.history["val_token_acc"].append(val_acc)
        self.history["examples"].append(self.example_seen)
        # 记录并显示指标
        """ 
        self.log_dict(
            {"val_loss": val_loss, "val_ppl": val_ppl, "val_token_acc": val_acc},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        """
        self.log("val_loss",val_loss,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_ppl",val_ppl,on_step=False,on_epoch=True,prog_bar=True)
        self.log("val_token_acc",val_acc,on_step=False,on_epoch=True,prog_bar=True)

    # ================= 测试 =================

    def test_step(self, batch, batch_idx):
        loss, acc, n_tokens = self._shared_train_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=n_tokens)
        self.log(
            "test_token_acc",
            acc / max(n_tokens, 1),
            on_step=False,
            on_epoch=True,
            batch_size=n_tokens,
        )
        return loss

    # ================= 优化器与学习率 =================
    def configure_optimizers(self) :
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        # 总步数 = 训练批数 * 训练轮数 ，在Lightning里 就是 self.trainer.estimated_stepping_batches这一个参数
        total_steps = int(self.trainer.estimated_stepping_batches)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=int(total_steps * self.hparams.warmup_frac),
            num_training_steps=total_steps,
        )
        # "interval": "step" 表示 scheduler 在 optimizer.step() 时(每批次一个step update)调用，而不是 epoch 时调用
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }