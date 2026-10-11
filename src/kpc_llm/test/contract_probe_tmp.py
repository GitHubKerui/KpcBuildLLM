"""临时契约探针：验证 PL 2.6 下 self.log 的 metrics.csv 列名 / checkpoint 文件名 /
best_model_path。只用假数据，不训练真实模型。用完即删。"""
import os

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, ThroughputMonitor
from pytorch_lightning.loggers import CSVLogger


class Dummy(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.lin = torch.nn.Linear(4, 4)
        self.tokens_seen = 0

    def training_step(self, batch, i):
        x, y = batch
        loss = ((self.lin(x) - y) ** 2).mean()
        self.tokens_seen += int(x.numel())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("tokens_seen", float(self.tokens_seen), on_step=True)
        return loss

    def validation_step(self, batch, i):
        x, y = batch
        loss = ((self.lin(x) - y) ** 2).mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-2)


ds = TensorDataset(torch.randn(20 * 2, 4), torch.randn(20 * 2, 4))
dl = DataLoader(ds, batch_size=2)

csv_logger = CSVLogger(save_dir="tmp_probe_logs", name="probe")
ckpt = ModelCheckpoint(
    dirpath="tmp_probe_ckpt",
    filename="ep{epoch:02d}-vl{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
    every_n_epochs=1,
)
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="cpu",
    devices=1,
    precision="32-true",
    val_check_interval=16,
    limit_val_batches=2,
    logger=csv_logger,
    callbacks=[
        ckpt,
        EarlyStopping(monitor="val_loss", mode="min", patience=10),
        # 本项目 batch 是 [input, target]，len 就是 batch_size，input.numel() 就是 token 数
        ThroughputMonitor(
            batch_size_fn=lambda batch: int(batch[0].shape[0]),
            length_fn=lambda batch: int(batch[0].numel()),
        ),
    ],
    log_every_n_steps=5,
    enable_progress_bar=False,
    enable_model_summary=False,
    num_sanity_val_steps=0,
)
trainer.fit(Dummy(), train_dataloaders=dl, val_dataloaders=dl)

csv_path = os.path.join(csv_logger.log_dir, "metrics.csv")
print("METRICS_PATH:", csv_path, "exists:", os.path.exists(csv_path))
with open(csv_path, encoding="utf-8") as f:
    content = f.read()
print("METRICS_RAW_START>>>")
print(content)
print("METRICS_RAW_END<<<")
print("BEST_MODEL_PATH:", trainer.checkpoint_callback.best_model_path)
print("CKPT_FILES:", os.listdir("tmp_probe_ckpt"))