import time  # 1. 引入时间模块

from logging import Logger


from torch._inductor.config import can_inplace_pad_graph_input
from torch.utils.data import dataloader
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.train_loss_calcu import calcuOneBatchCrossEnLoss,caluBatchesCrossEnLoss
from kpc_llm.layers.kpc_llm_model.token_process.tiktokenizer import qwtokenizer2ids, qwtokenizer2txts
from kpc_llm.layers.kpc_llm_model.test.generate_txt_loop import generate_txt_loop
from kpc_llm.layers.loss_func.kpc_cross_entropy import target_batch
from kpc_llm.layers.kpc_llm_model.pretraining.train_data_div3_load import divDatas2TraValTes
from kpc_llm.data_process.textloader import getTxtStr
from kpc_llm.utils.logger import getlogger
from transformers import AutoTokenizer
import torch

# 写配置
TRAIN_CNF = {
    "vcab_sz" : 151936,
    "cntext_lnth" : 256,
    # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
    "max_cntxt_pstion_lnth" : 256,
    "emb_dim" : 512,
    "heads_num" : 8,
    "trnsf_blocks_num" : 8,
    "drop_rt" : 0.1,
    "qkv_bias" : False,
    "batch_size":4,
    # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
    "returnSoftmax" : False
}

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, start_context):
    # 初始化跟踪训练的参数
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 初始化时间和Token记录变量，用于计算吞吐量
    last_time = time.time()
    last_tokens_seen = 0

    # 主训练循环
    for epoch in range(num_epochs):
        # 开启模型训练模式
        model.train()  
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        print(f"-------------------- Ep {epoch+1} Training -------------------: ")
        for i,(input_batch, target_batch) in enumerate(train_loader):

            print(f"-------------------- Ep {epoch+1} Batch {i+1} Training -------------------: ")
            # 每次学习前，重置之前的内存梯度为0
            optimizer.zero_grad() 
            loss = calcuOneBatchCrossEnLoss(input_batch, target_batch, model, device)
            # 计算损失的梯度
            loss.backward() 
            # 根据梯度更新weights
            optimizer.step() 
            # 计算模型token的总的训练量，也就是人的读书的字数
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_data_loss, val_data_loss = evaluate_model(
                    model, train_loader, val_loader, device)
                train_losses.append(train_data_loss)
                val_losses.append(val_data_loss)
                track_tokens_seen.append(tokens_seen)
                
                # 2. 计算速度核心逻辑
                current_time = time.time()
                time_elapsed = current_time - last_time       # 距离上次评估过去了多少秒
                tokens_processed = tokens_seen - last_tokens_seen # 这期间一共处理了多少Token
                
                # 防止极其罕见的除以0情况（例如 eval_freq 设得极小且运行极快）
                tokens_per_sec = tokens_processed / time_elapsed if time_elapsed > 0 else 0
                
                # 3. 在日志中打印速度
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_data_loss:.3f}, Val loss {val_data_loss:.3f} | "
                      f"Speed: {tokens_per_sec:.0f} tokens/s")
                
                # 4. 更新基准线，为下一次计算做准备
                last_time = current_time
                last_tokens_seen = tokens_seen

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, device, start_context,tokenizer
        )

    return train_losses, val_losses, track_tokens_seen

# 评估状态下，对训练数据，和评估数据计算loss值
def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        # 这是是关掉drop和梯度计算的train数据loss，和训练时候的不一样所以必须重新算一次。
        train_data_loss = caluBatchesCrossEnLoss(train_loader, model, device)
        # 默认全部批次计入loss统计 batchNum
        val_data_loss = caluBatchesCrossEnLoss(val_loader, model, device )
    model.train()
    return train_data_loss, val_data_loss

# 自回归生成预测测试
def generate_and_print_sample(model, device, start_context,tokenizer):
    model.eval()
    
    # context_size = model.pos_emb.weight.shape[0]
    input_ids = qwtokenizer2ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_txt_loop(input_ids,TRAIN_CNF['cntext_lnth'],model,50)
        decoded_text = qwtokenizer2txts(token_ids,tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

torch.manual_seed(825)
model = KpcLLMModel(TRAIN_CNF)
print (f"cuda is {torch.cuda.is_available()}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 2

# get TrainData from data/the-verdict.txt
# en_train_txt = getTxtStr('the-verdict.txt','data')
cn_train_txt = getTxtStr('corpus_zh.txt','data')
train_txtln  =  len(cn_train_txt) if cn_train_txt is not None else 0
print(f"train_txt length : {train_txtln}")
trainDsloader,valDsloader,testDsloader = divDatas2TraValTes(cn_train_txt,TRAIN_CNF['batch_size'],TRAIN_CNF['cntext_lnth'],TRAIN_CNF['cntext_lnth'])

# 初始输入的文字
cn_start_cont = "这是一个小红帽的故事，从前"
en_start_cont = "Long long ago, there is a girl "

train_losses, val_losses, tokens_seen = train_model_simple(
    model, trainDsloader, valDsloader, optimizer, device,
    num_epochs = num_epochs, eval_freq=8,
    start_context = cn_start_cont
)