
from kpc_llm.layers.kpc_llm_model.pretraining.train_loss_calcu import calcuOneBatchCrossEnLoss,caluBatchesCrossEnLoss
import torch 
# 评估状态下，对训练数据，和评估数据计算loss值,并返回
def evaluate_model(model, val_loader, device, eval_batch_num=16, cur_train_loss: float | None = None):
    """
    注意：这里不要再去迭代 train_loader。
    外层 for 循环正在遍历 train_loader，评估时若再对同一个 DataLoader 创建第二个迭代器，
    在 persistent_workers=True 下两个迭代器会争抢 worker，导致外层 epoch 迟迟走不完
    （表现为 global_step 一直涨、epoch 却不增加）。
    train loss 直接用训练过程中维护的滑动平均值传入。
    """
    model.eval()
    with torch.no_grad():
        # 只抽样前 eval_batch_num 个 batch：全量评估（batchNum=0）会把训练拖慢十几倍
        val_data_loss = caluBatchesCrossEnLoss(val_loader, model, device, eval_batch_num)
    model.train()
    return (cur_train_loss if cur_train_loss is not None else float("nan")), val_data_loss