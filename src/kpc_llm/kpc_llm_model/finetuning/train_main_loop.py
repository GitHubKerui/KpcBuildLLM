

from collections import deque
from llms_from_scratch.ch04 import GPTModel
from logging import Logger
from kpc_llm.kpc_llm_model.finetuning.classify_sft.cls_sft_loss import multi_batches_loss, multi_batches_random_loss, one_batch_loss
from kpc_llm.kpc_llm_model.finetuning.classify_sft.prepare_cls_fintuning_model import prepare_finetuning_model
from kpc_llm.kpc_llm_model.finetuning.classify_sft.load_pre_gpt2 import  load_GPT2_model
from kpc_llm.kpc_llm_model.finetuning.classify_sft.prepare_dataloader import prepare_dataloader
from kpc_llm.kpc_llm_model.train_cfg import GPT2_LARGE_CNF
from kpc_llm.kpc_llm_version2.GPTModel_load import to_ch04_config
from kpc_llm.utils import  getlogger
from kpc_llm.utils.adaptive_device import get_adaptive_device
import tiktoken
import torch
from kpc_llm.utils.pyplot.plot_values import plot_values
import time
from transformers import get_cosine_schedule_with_warmup

tokenizer = tiktoken.get_encoding("gpt2")
log: Logger =  getlogger()

# 这里设计不好，容易迷糊，应该分两个方法函数，一个是全批量计算准确度（百分比），一个是部分批次计算专区额度
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            with torch.no_grad():
                # 关键点 最后一个token的embeding张量训练成分类标签的特征[x.xx,y.yy] -> sft -> argmax -> label
                logits = model(input_batch)[:, -1, :]  # Logits of last output token
            # 返回的predicted_labels一个batch里的所有的sample的last token embedding中的 max的 index。最后的shape是 batch_size,1
            predicted_labels = torch.argmax(logits, dim=-1)

            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """ 如果是训练过程中，这个方法不适用，因为loss 在前向传播时候就有了，不需要二次计算。 """
    model.eval()
    with torch.no_grad():
        train_loss = multi_batches_loss(train_loader, model, device, eval_iter)
        val_loss = multi_batches_loss(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def eval_by_step(train_loss,batch_step,eval_freq,train_batch_num,eval_btc_nun,epoch,global_step,btc_size,train_losses,val_losses,train_loss_ema):
    
    # 每隔 eval_freq 步评估一次模型,保证最后一个batch 也评估一次
    if batch_step % eval_freq == 0 or batch_step == train_batch_num:

        # 获取当前的训练和验证损失
        model.eval()
        # val_loss已经做了smooth 所以不需要 collections deque 窗口
        val_loss = multi_batches_random_loss(val_loader, model, device, eval_btc_nun)
        model.train() 
        train_losses.append(train_loss_ema)
        val_losses.append(val_loss)
        # 必须存成 python float：smooth_loss 是 cuda 张量，直接交给 matplotlib 会报
        # "can't convert cuda:0 device type tensor to numpy"
        print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")
    return train_losses,val_losses

def warmup_cosin_annealing(train_batch_num,optimizer):
     # 设置学习率的warmup,cosine Annealing 
    warmup_steps = int(train_batch_num*0.15)
    print(f"--- train_batch_num : {train_batch_num} --- warmup_steps : {warmup_steps} ---")
    # 2. 只需要填入两个总步数（按你的数据集实际情况填写）
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,      # 预热步数（比如前 100 步 lr 从 0 线性升到 2e-5）
        num_training_steps=train_batch_num    # 整个训练的总步数（总 batch 数）
    )
    return scheduler

def accuracy_by_step(eval_freq,eval_btc_nun,train_accs,val_accs):
    
    # 计算指定批量的精确度,train的抽取比例和 val 的比例要一致。所以一个eval_freq，一个 eval_iter
    train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_freq)
    val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_btc_nun)
    print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    train_accs.append(train_accuracy)
    val_accs.append(val_accuracy)
    

def train_main_loop(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq_frac=0.1, eval_data_frac=0.5):
    
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    # 训练的ema 损失平滑计算
    train_loss_ema = None
    # 训练批次样本数
    btc_size = train_loader.batch_size
    # 评估频率设置计算量的百分比。evl_percent = 1:eval_iter 这里设置10%
    train_batch_num = len(train_loader)
    val_batch_num = len(val_loader)
    # 保证训练循环中，评估频率至少一次
    eval_freq = max(1,int(train_batch_num*eval_freq_frac)) 
    # 评估从评估数据，随机抽取的比例数
    eval_btc_nun = max(1,int(val_batch_num*eval_data_frac)) 
    print(f"--- 训练批次数量 ：{train_batch_num} --- 评估批次数量 ： {val_batch_num} ---")
    print(f"--- {eval_freq} 个批次数评估一次 --- 每次评估用 {eval_btc_nun} 个批次 ---")

    # 设置学习率的warmup,cosine Annealing 的scheduler
    scheduler = warmup_cosin_annealing(train_batch_num,optimizer)

    for epoch in range(num_epochs):
        model.train() 
        batch_step = 0
        for input_batch, target_batch in train_loader:
            """ 每轮训练前 重置loss梯度为0 """
            optimizer.zero_grad()
            """ 前向计算 == 算误差 """
            train_loss = one_batch_loss(input_batch, target_batch, model, device)
            """ 反向计算 == 算梯度 """
            train_loss.backward() # 用loss张量计算梯度，loss是Model计算出来的，包含全部参数信息。
            """ step == 更新参数 为啥不用update 用step，服了torch"""
            optimizer.step() # 优化器更新参数，优化器设置了model的参数，所以这里不用传model
            scheduler.step() # 用学习率调度器更新学习率
            examples_seen += input_batch.shape[0] # 横坐标的x轴，训练的样本数
            global_step += 1
            batch_step += 1
             # ema 计算的效果似乎 比窗口好而且简单。
            train_loss_ema = train_loss.item() if train_loss_ema is None else 0.93 * train_loss_ema + 0.07 * train_loss.item()

            # 每一步的评估
            eval_by_step(train_loss,batch_step,eval_freq,train_batch_num,eval_btc_nun,epoch,global_step,btc_size,train_losses,val_losses,train_loss_ema)
            
        # 计算指定批量的精确度,train的抽取比例和 val 的比例要一致。所以一个eval_freq，一个 eval_iter
        accuracy_by_step(eval_freq,eval_btc_nun,train_accs,val_accs)
        
    return train_losses, val_losses, train_accs, val_accs, examples_seen

if "__main__" == __name__:

    gpt2_large_cnf = to_ch04_config(GPT2_LARGE_CNF)
    # 换分类头之前固定种子：不种子的话每次跑都是一个全新的随机头，
    # 而其中一部分初始化会让训练在前 100 步停滞（loss 不降反升），结果不可复现。
    # 原书正是在这里 manual_seed(123) 之后再替换 out_head 的。
    # torch.manual_seed(123)
    model = load_GPT2_model()
    model: GPTModel = prepare_finetuning_model(model)
    train_loader,val_loader,test_loader = prepare_dataloader()
    device= get_adaptive_device()
    model.to(device)

    # 自检 1：三个 loader 的类别分布。任何一个是单一类别，就说明切分有偏，指标不可信
    for nm, ld in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        labels = [int(y) for _, y in ld.dataset]
        print(nm, "spam:", sum(labels), "ham:", len(labels) - sum(labels), "总数:", len(labels))

    model.eval()
    with torch.no_grad():
        print("baseline:", multi_batches_loss(train_loader, model, device))

    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    num_epochs = 1
    model.train()
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_main_loop(
        model, train_loader, val_loader, optimizer, device,
        num_epochs, 0.1, 0.5
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    # .cpu()：linspace 默认落在当前默认设备上，一旦默认设备是 cuda，
    # matplotlib 转 numpy 时会报 "can't convert cuda:0 device type tensor to numpy"
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses)).cpu()
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses)).cpu()

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

