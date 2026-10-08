import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import tiktoken


# ----------------------------------------------------
# 1. 配置定义
# ----------------------------------------------------
@dataclass
class TrainConfig:
    cnf_name: str
    vcab_sz: int = 100277       # cl100k_base 的词表大小
    cntext_lnth: int = 256
    max_cntxt_pstion_lnth: int = 256
    emb_dim: int = 512
    heads_num: int = 8
    trnsf_blocks_num: int = 8
    batch_size: int = 16
    lr: float = 3e-4
    warmup_steps: int = 50      # 线性预热步数
    max_steps: int = 1000       # 训练总步数
    dropout: float = 0.1


GPT2_cl100k_base_CNF = TrainConfig(
    cnf_name="GPT2_cl100k_base_CNF",
    vcab_sz=100277,
    cntext_lnth=256,
    max_cntxt_pstion_lnth=256,
    emb_dim=512,
    heads_num=8,
    trnsf_blocks_num=8,
    batch_size=16,
    warmup_steps=50,
    max_steps=500
)


# ----------------------------------------------------
# 2. 真实文本字符串 Dataset (基于 tiktoken cl100k_base)
# ----------------------------------------------------
class RawTextDataset(Dataset):
    """
    接收原始长字符串文本，通过 tiktoken 编码，
    并切分为长度为 (cntext_lnth) 的 (input, target) 元组。
    """
    def __init__(self, raw_text: str, context_length: int, enc_name: str = "cl100k_base"):
        self.context_length = context_length
        # 1. 加载 tiktoken 分词器并编码原始长文本
        tokenizer = tiktoken.get_encoding(enc_name)
        token_ids = tokenizer.encode(raw_text, allowed_special={"<|endoftext|>"})
        
        # 转为 1 维 LongTensor 存入内存
        self.token_tensor = torch.tensor(token_ids, dtype=torch.long)
        
        # 每个训练块需要 (context_length + 1) 个 token 来形成 shifted 对
        self.chunk_size = context_length + 1
        self.num_samples = len(self.token_tensor) // self.chunk_size
        
        if self.num_samples == 0:
            raise ValueError(f"输入文本长度过短！总 Token 数为 {len(self.token_tensor)}，至少需要 {self.chunk_size} 个 Token。")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        start_idx = idx * self.chunk_size
        chunk = self.token_tensor[start_idx : start_idx + self.chunk_size]
        
        # 切分成 inputs 和 targets，返回二元组
        inputs = chunk[:-1]
        targets = chunk[1:]
        return inputs, targets


# ----------------------------------------------------
# 3. 纯 PyTorch 模型结构
# ----------------------------------------------------
class GPT2LMModel(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vcab_sz, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.max_cntxt_pstion_lnth, cfg.emb_dim)
        self.drop = nn.Dropout(cfg.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.emb_dim,
            nhead=cfg.heads_num,
            dim_feedforward=cfg.emb_dim * 4,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=cfg.trnsf_blocks_num)
        self.ln_f = nn.LayerNorm(cfg.emb_dim)
        self.lm_head = nn.Linear(cfg.emb_dim, cfg.vcab_sz, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.size()
        positions = torch.arange(0, t, dtype=torch.long, device=idx.device)
        
        x = self.tok_emb(idx) + self.pos_emb(positions)
        x = self.drop(x)

        causal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        x = self.blocks(x, mask=causal_mask, is_causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)


# ----------------------------------------------------
# 4. PyTorch Lightning 包装器（带完整 Warmup 调度）
# ----------------------------------------------------
class LitGPT2Trainer(pl.LightningModule):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.save_hyperparameters(vars(cfg))
        self.cfg = cfg
        self.model = GPT2LMModel(cfg)

    def forward(self, x):
        return self.model(x)

    def _compute_loss(self, batch):
        inputs, targets = batch
        logits = self(inputs)
        
        # 展平计算全局平均交叉熵损失
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten()
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch)
        val_ppl = torch.exp(loss.clamp(max=20.0))
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_ppl", val_ppl, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        # 1. 过滤不进行权重衰减的参数（LayerNorm 与 Bias）
        decay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() >= 2]
        nodecay_params = [p for n, p in self.named_parameters() if p.requires_grad and p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": 0.01},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.cfg.lr, betas=(0.9, 0.95))

        # 2. 真实生效的 Warmup + Cosine 衰减链式调度器
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,             # 从 lr * 1e-4 开始预热
            end_factor=1.0,               # 升至满额 lr
            total_iters=self.cfg.warmup_steps
        )
        
        cosine_steps = max(1, self.cfg.max_steps - self.cfg.warmup_steps)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=1e-5                  # 最终最低衰减学习率
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_sched, cosine_sched],
            milestones=[self.cfg.warmup_steps]
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",       # 确保每个 step 推进一次调度器
            }
        }


# ----------------------------------------------------
# 5. 训练主流程
# ----------------------------------------------------
if __name__ == "__main__":
    cfg = GPT2_cl100k_base_CNF

    # 准备真实文本（这里用重复长文本作为演示，实际可读取大型 .txt 文本）
    sample_text = (
        "Machine learning models based on the Transformer architecture have achieved "
        "remarkable success across natural language processing and multimodal tasks. "
        "Pre-training on massive corpora allows models to capture intricate contextual representations. "
    ) * 3000  # 构造成几万个 token 的长字符串

    # 划分训练集与验证集文本 (9:1)
    split_idx = int(len(sample_text) * 0.9)
    train_text = sample_text[:split_idx]
    val_text = sample_text[split_idx:]

    train_ds = RawTextDataset(train_text, context_length=cfg.cntext_lnth)
    val_ds = RawTextDataset(val_text, context_length=cfg.cntext_lnth)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, pin_memory=True)

    lit_model = LitGPT2Trainer(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints_{cfg.cnf_name}/",
        filename="gpt2-best-{step:04d}-{val_loss:.3f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        accelerator="auto",
        precision="16-mixed",         # 开启混合精度降低 100K 词表带来的显存开销
        gradient_clip_val=1.0,        # 梯度裁剪防爆炸
        callbacks=[checkpoint_callback, lr_monitor],
        val_check_interval=50,
        log_every_n_steps=5
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)