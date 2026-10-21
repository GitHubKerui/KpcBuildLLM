"""DPO（Direct Preference Optimization）的 Lightning 实现。

数据契约（来自 dpo_data_loader.collate_fn）：
    batch 是 dict：
        chosen / rejected        [B, T] int64  完整 token 序列（prompt + response），已 padding
        chosen_mask / rejected_mask [B, T] bool  True 表示 response 段（prompt 与 padding 为 False）

DPO 与 SFT 的三个关键差异：
    1. 两个模型：self.model 是 policy（训练），self.ref_model 是 reference（冻结、永远 eval）
    2. 两路前向：chosen 和 rejected 各算一次，拿的是「序列 logprob」而不是逐 token 均值
    3. next-token 对齐：位置 t 的 logits 预测 t+1 的 token，logits 丢最后一列、targets 和 mask 丢第一列
"""

from typing import Any

import lightning as L
import torch
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

from kpc_llm.kpc_llm_model.finetuning.direct_preference_sft.compute_dpo_loss import compute_dpo_loss
from kpc_llm.utils import getlogger

log = getlogger()


class LightingInstructionSFT(L.LightningModule):
    """DPO：冻结 reference，只训练 policy，拉大 chosen 与 rejected 的 logprob 差。"""

    hparams: Any  # save_hyperparameters 动态注入，声明供类型检查使用

    def __init__(self, model, ref_model, lr: float = 5e-5, weight_decay: float = 0.1,
                 warmup_frac: float = 0.15, beta: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "ref_model"])
        self.model = model          # policy model，要训练
        self.ref_model = ref_model  # reference model，只读
        # 注意必须用 requires_grad_()：给 nn.Module 赋 requires_grad=False 只是加了个普通属性，
        # 内部参数仍然是 True，ref 会被一起训练。
        self.ref_model.requires_grad_(False)
        self.ref_model.eval()
        self.ignore_index = ignore_index
        self.beta = beta

        # 单批 loss 抖动大，用 EMA 平滑，比滑动窗口简单且曲线更稳
        self.train_loss_ema: float | None = None

        # 画图历史刻意由模块自身维护：Lightning 里 callback 与模块钩子的先后顺序不直观，
        # 指标只在一个地方记录最不容易出错。
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_pref_acc": [],    # DPO 主指标：policy 认为 chosen 优于 rejected 的比例
            "train_token_acc": [],   # 仅参考：next-token 命中率，DPO 并不优化它，通常会下降
            "val_loss": [],
            "val_pref_acc": [],
            "val_token_acc": [],
            "examples": [],
        }
        # 样本训练数等于batch_size的和
        self.example_seen = 0
        self._reset_train_stats()
        self._reset_val_stats()

    # ================= 前向与损失 =================

    def forward(self, input_batch: torch.Tensor) -> torch.Tensor:
        # 这里是self(input_batch)的入口，跟原生的不一样
        return self.model(input_batch)  # 输出 [B, T, vocab]

    def train(self, mode: bool = True):
        """ 为了保证 refer model 被定死处于eval() """
        super().train(mode)
        self.ref_model.eval()  # 永远钉死
        return self

    def _seq_logps(self, logits, targets, mask):
        """[B,T,V] + [B,T] + [B,T] → [B]：response 段的序列 logprob（DPO 要的就是这个）。

        next-token：位置 t 的 logits 预测 t+1 的 token，所以 logits 丢最后一列，
        targets 和 mask 丢第一列，三者必须同步移动，否则整体错位一个 token。
        """
        logp = F.log_softmax(logits[:, :-1].float(), dim=-1)   # [B,T-1,V]
        tgt = targets[:, 1:].clamp(min=0).unsqueeze(-1)        # [B,T-1,1]
        m = mask[:, 1:]                                        # [B,T-1]
        token_logp = logp.gather(-1, tgt).squeeze(-1)          # [B,T-1]
        return (token_logp * m).sum(-1)                        # [B]

    def _dpo_two_shared_step(self, inputs, mask):
        """policy / reference 各算一次序列 logprob，外加 token 准确率（仅作参考指标）"""
        logits_policy = self(inputs)             # [B,T,V]
        logits_ref = self.ref_model(inputs)      # [B,T,V]

        logp_policy = self._seq_logps(logits_policy, inputs, mask)  # [B]
        logp_ref = self._seq_logps(logits_ref, inputs, mask)        # [B]

        with torch.no_grad():
            m = mask[:, 1:]
            tgt = inputs[:, 1:]
            pred_p = logits_policy[:, :-1].argmax(-1)[m]
            pred_r = logits_ref[:, :-1].argmax(-1)[m]
            acc_policy = int((pred_p == tgt[m]).sum())
            acc_ref = int((pred_r == tgt[m]).sum())

        return logp_policy, logp_ref, acc_policy, acc_ref, int(m.sum())

    def _shared_train_step_dpo(self, batch):
        """chosen / rejected 各算一次，再拼成 DPO loss。

        返回 (loss, 偏好命中数, token 命中数, 有效 token 数)。
        偏好命中 = policy 给 chosen 的 logprob 高于 rejected 的样本数，这才是 DPO 在优化的东西。
        """
        c_logp_p, c_logp_r, acc_p_ch, _acc_r_ch, n_c = self._dpo_two_shared_step(
            batch["chosen"], batch["chosen_mask"]
        )
        r_logp_p, r_logp_r, acc_p_rej, _acc_r_rej, n_r = self._dpo_two_shared_step(
            batch["rejected"], batch["rejected_mask"]
        )

        # compute_dpo_loss 吃的是 NLL=-logp，所以这里取负号
        loss = compute_dpo_loss(-c_logp_p, -r_logp_p, -c_logp_r, -r_logp_r, self.beta)
        pref_hit = int((c_logp_p > r_logp_p).sum())
        return loss, pref_hit, acc_p_ch + acc_p_rej, n_c + n_r

    # ================= 训练 =================

    def _reset_train_stats(self):
        # 重置训练统计数据
        self._trn_loss_sum = 0.0
        self._trn_samples = 0       # 样本数：DPO loss 是 per-sample 均值，加权用它
        self._trn_pref = 0          # 偏好命中数（样本级），DPO 的主指标
        self._trn_acc = 0           # token 命中数（仅参考）
        self._trn_tokens = 0        # 有效 token 数，给 token acc 做分母

    def on_train_epoch_start(self) -> None:
        self._reset_train_stats()

    def training_step(self, batch, batch_idx):
        loss, pref_hit, acc_hit, n_tokens = self._shared_train_step_dpo(batch)
        batch_size = int(batch["chosen"].shape[0])
        # DPO loss 是 per-sample 的均值，按样本数加权；乘 token 数会把长 response 的批次过度放大
        self._trn_loss_sum += loss.item() * batch_size
        self._trn_samples += batch_size
        self._trn_pref += pref_hit
        self._trn_acc += acc_hit
        self._trn_tokens += n_tokens
        self.example_seen += batch_size
        # 第一次调用 train_loss_ema is None 会获得loss，后面获得的都是加权平均。
        # 0.93 换算下来 约等于 15次数据的加权平均。只是约等于
        # 注意 (1-0.93)=0.07 必须写：漏了的话稳态 ema = loss/0.07，数值被放大 14 倍
        self.train_loss_ema = (
            loss.item()
            if self.train_loss_ema is None
            else 0.93 * self.train_loss_ema + 0.07 * loss.item())
        # Lightning 框架进度条 用 self.log
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        self.log("train_loss_ema", self.train_loss_ema, on_step=True, on_epoch=False)
        # 这里不需要写 loss.backward,也不用optimizer.step框架会处理
        return loss

    def on_train_epoch_end(self) -> None:
        if self._trn_samples == 0:
            return
        self.log(
            "train_loss_epoch",
            self._trn_loss_sum / self._trn_samples,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    # ================= 验证 =================

    def _reset_val_stats(self):
        self._val_loss_sum = 0.0
        self._val_samples = 0
        self._val_pref = 0
        self._val_acc = 0
        self._val_tokens = 0

    def on_validation_epoch_start(self) -> None:
        # val_check_interval<1,一个epoch 验证次数== 1/val_check_interval,每次都要清零，
        # 否则「本次验证均值」会变成历次累计平均，曲线严重滞后。
        self._reset_val_stats()

    def validation_step(self, batch, batch_idx):
        # 获取验证，loss,acc,n_tokens,lightning 框架自动调用 model.eval()
        loss, pref_hit, acc_hit, n_tokens = self._shared_train_step_dpo(batch)
        batch_size = int(batch["chosen"].shape[0])
        self._val_loss_sum += loss.item() * batch_size
        self._val_samples += batch_size
        self._val_pref += pref_hit
        self._val_acc += acc_hit
        self._val_tokens += n_tokens
        return loss

    def on_validation_epoch_end(self) -> None:
        # 训练前的sanity check 的bool标志 ，如果是这个阶段则不进指标历史统计数据， (训练前会试跑默认2个批次来验证整体正常度)
        if self.trainer.sanity_checking:
            return
        # 如果验证集为空，则不记录
        if self._val_samples == 0 or self.train_loss_ema is None:
            return
        val_loss = self._val_loss_sum / self._val_samples
        val_pref_acc = self._val_pref / max(self._val_samples, 1)    # 主指标：chosen 压过 rejected 的比例
        val_token_acc = self._val_acc / max(self._val_tokens, 1)     # 参考指标：next-token 命中率
        # 记录指标历史
        self.history["train_loss"].append(self.train_loss_ema)
        self.history["train_pref_acc"].append(self._trn_pref / max(self._trn_samples, 1))
        self.history["train_token_acc"].append(self._trn_acc / max(self._trn_tokens, 1))
        self.history["val_loss"].append(val_loss)
        self.history["val_pref_acc"].append(val_pref_acc)
        self.history["val_token_acc"].append(val_token_acc)
        self.history["examples"].append(self.example_seen)
        # 记录并显示指标
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_pref_acc", val_pref_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_token_acc", val_token_acc, on_step=False, on_epoch=True, prog_bar=True)

    # ================= 测试 =================

    def test_step(self, batch, batch_idx):
        loss, pref_hit, acc_hit, n_tokens = self._shared_train_step_dpo(batch)
        batch_size = int(batch["chosen"].shape[0])
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(
            "test_pref_acc",
            pref_hit / max(batch_size, 1),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test_token_acc",
            acc_hit / max(n_tokens, 1),
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss

    # ================= 优化器与学习率 =================

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),   # 只传 policy，ref 绝不能进优化器
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
