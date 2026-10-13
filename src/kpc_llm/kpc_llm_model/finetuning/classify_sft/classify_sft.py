

from llms_from_scratch.ch04 import GPTModel
from logging import Logger

from kpc_llm.kpc_llm_model import KpcLLMModel
from kpc_llm.kpc_llm_model.finetuning.classify_sft.cls_sft_loss import multi_batches_loss, one_batch_loss
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

# 这里设计不好，容易迷糊，应该分成连个方法函数，一个是全批量计算准确度（百分比），一个是部分批次计算专区额度
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
    model.eval()
    with torch.no_grad():
        train_loss = multi_batches_loss(train_loader, model, device, eval_iter)
        val_loss = multi_batches_loss(val_loader, model, device, eval_iter)
    model.train()
    return train_loss, val_loss

def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter):
    # Initialize lists to track losses and examples seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            """ 这一步之前，optimizer必须把model.parameter装进去并制定所在硬件 """
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            """ 前向计算 == 算误差 """
            loss = one_batch_loss(input_batch, target_batch, model, device)
            """ 反向计算 == 算梯度 """
            loss.backward() # Calculate loss gradients
            """ step == 更新参数 为啥不用update 用step，服了torch"""
            optimizer.step() # Update model weights using loss gradients
            examples_seen += input_batch.shape[0] # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

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

    # 自检 2：未训练基线。分类头是随机初始化的，应落在 ln2=0.693 附近（0.65~0.85）
    model.eval()
    with torch.no_grad():
        print("baseline:", multi_batches_loss(train_loader, model, device))

    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
    batch_num = len(train_loader)
    warmup_steps = int(batch_num*0.15)
    print(f"batch_num : {batch_num}")
    print(f"warmup_steps : {warmup_steps}")
    # 2. 只需要填入两个总步数（按你的数据集实际情况填写）
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,      # 预热步数（比如前 100 步 lr 从 0 线性升到 2e-5）
        num_training_steps=batch_num    # 整个训练的总步数（总 batch 数）
    )

    num_epochs = 7

    # 自检 3：前 10 步 batch loss 必须肉眼可见地下降。
    # 不降就说明链路有问题（token 位置/梯度方向/设备），与数据比例无关。
    # 注意这 10 步会真实更新权重，属于训练前的热身，确认无误后可整段注释掉。
    model.train()
    for i, (ib, tb) in enumerate(train_loader):
        if i >= 10:
            break
        optimizer.zero_grad()
        loss = one_batch_loss(ib, tb, model, device)
        loss.backward()
        optimizer.step()
        scheduler.step() # <--- 每次 backward + step 之后更新一次学习率
        print(f"step {i}: {loss.item():.4f}")

    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)

