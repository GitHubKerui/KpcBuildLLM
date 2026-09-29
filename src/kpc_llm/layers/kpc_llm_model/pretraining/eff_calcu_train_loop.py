import time  # 1. 引入时间模块
import multiprocessing

from logging import Logger


from torch._inductor.config import can_inplace_pad_graph_input
from torch.utils.data import dataloader
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.train_loss_calcu import calcuOneBatchCrossEnLoss,caluBatchesCrossEnLoss
from kpc_llm.layers.kpc_llm_model.token_process.tokenizer_hub import qwtokenizer2ids, qwtokenizer2txts,tiktokenizer2ids,tiktokenizer2txts
import tiktoken
from kpc_llm.layers.kpc_llm_model.test.llm_text.generate_txt_loop import generate_txt_loop
from kpc_llm.layers.kpc_llm_model.pretraining.train_data_div3_load import divDatas2TraValTes
from kpc_llm.data_fetch.textloader import getTxtStr
from kpc_llm.utils.logger import getlogger
from transformers import AutoTokenizer
import torch

# 写配置
TRAIN_CNF = {
    "vcab_sz" : 100277,
    "cntext_lnth" : 256,
    # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
    "max_cntxt_pstion_lnth" : 256,
    "emb_dim" : 512,
    "heads_num" : 8,
    "trnsf_blocks_num" : 8,
    "drop_rt" : 0.1,
    "qkv_bias" : False,
    "batch_size":16,
    # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
    "returnSoftmax" : False
}
tokenizerTrain =tiktoken.get_encoding("cl100k_base") 

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, start_context):
    # 初始化跟踪训练的参数
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # 训练集 loss 的滑动平均（EMA），避免评估时再去迭代 train_loader
    train_loss_ema = None

    # 初始化时间和Token记录变量，用于计算吞吐量
    last_time = time.time()
    last_tokens_seen = 0

    # 主训练循环
    for epoch in range(num_epochs):
        # 开启模型训练模式
        model.train()  
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        # tokenizer = tiktoken.get_encoding('gpt2')

        batch_num   = len(train_loader)                        # ← 全部 batch 的数量
        sample_num  = len(train_loader.dataset)                # 数据集样本总数
        one_sample_token_num  = train_loader.dataset.chunk_len # 数据集样本总数
        batch_size  = train_loader.batch_size                  # 每个 batch 的样本数
        token_num   = sample_num * one_sample_token_num   # 一个 epoch 覆盖的 token 总数

        
        print(f"-------------------- Ep {epoch+1}/{num_epochs} Training (batch_num={batch_num}) ------------------: ")
        print(f"--每个样本的token数：{one_sample_token_num}，每批次样本数: {batch_size},总共多少批次: {batch_num},一个Epoch覆盖的token总数: {token_num}--: ")
        for i,(input_batch, target_batch) in enumerate(train_loader):

            # print(f"-------------------- Ep {epoch+1} Batch {i+1} Training -------------------: ")
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
            # 维护训练集 loss 的滑动平均，供评估时直接复用
            loss_val = loss.item()
            train_loss_ema = loss_val if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * loss_val

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_data_loss, val_data_loss = evaluate_model(
                    model, val_loader, device, cur_train_loss=train_loss_ema)
                train_losses.append(train_data_loss)
                val_losses.append(val_data_loss)
                track_tokens_seen.append(tokens_seen)
                
                # 2. 计算速度核心逻辑
                current_time = time.time()
                time_elapsed = current_time - last_time       # 距离上次评估过去了多少秒
                tokens_processed = tokens_seen - last_tokens_seen # 这期间一共处理了多少Token
                
                # 防止极其罕见的除以0情况（例如 eval_freq 设得极小且运行极快）
                tokens_per_sec = tokens_processed / time_elapsed if time_elapsed > 0 else 0
                
                # 3. 在日志中打印
                # 注意：进度要用 epoch 内的 i+1 除以总批次数 batch_num；
                # global_step 是跨 epoch 累计的，用它算进度会超过 100%
                print(f"Ep {epoch+1}/{num_epochs} (BatchAll : {batch_num}) "
                      f"(Batch now : {i+1:06d}/{batch_num:06d}): "
                      f"input_batch.shape: {input_batch.shape} target_batch.shape: {target_batch.shape} "
                      f"Train loss: {train_data_loss:.3f}, Val loss: {val_data_loss:.3f} | "
                      f"Speed: {tokens_per_sec:.0f} tokens/s "
                      f"\nThis Epoch Finish: {(i+1)/batch_num * 100:.3f}% "
                      f"| Global Step: {global_step}"
                      )
                
                # 4. 更新基准线，为下一次计算做准备
                last_time = current_time
                last_tokens_seen = tokens_seen

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, device, start_context,tokenizer=tokenizerTrain
        )

    return train_losses, val_losses, track_tokens_seen

# 评估状态下，对训练数据，和评估数据计算loss值
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

# 自回归生成预测测试
def generate_and_print_sample(model, device, start_context,tokenizer):
    model.eval()
    
    # context_size = model.pos_emb.weight.shape[0]
    # input_ids = qwtokenizer2ids(start_context,tokenizer).to(device)
    input_ids = tiktokenizer2ids(start_context,tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_txt_loop(input_ids,TRAIN_CNF['cntext_lnth'],model,50)
        decoded_text = tiktokenizer2txts(token_ids,tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def main():
    torch.manual_seed(825)
    model = KpcLLMModel(TRAIN_CNF)
    print (f"cuda is {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)

    num_epochs = 30

    # get TrainData from data/the-verdict.txt
    # en_train_txt = getTxtStr('the-verdict.txt','data')
    # cn_train_txt = getTxtStr('corpus_zh_half.txt','data')
    cn_train_txt = getTxtStr('corpus_zh.txt','data')
    train_txtln  =  len(cn_train_txt) if cn_train_txt is not None else 0
    print(f"train_txt length : {train_txtln}")
    # 82 万 id 只占约 6.6MB，放 GPU 没有收益，且 DataLoader 的 spawn 子进程读不到 CUDA 张量；
    # reshape(-1) 把 tiktokenizer2ids 返回的 [1, token_num] 摊平成一维
    train_ids = tiktokenizer2ids(cn_train_txt, tokenizerTrain).reshape(-1).cpu()
    trainDsloader,valDsloader,testDsloader = divDatas2TraValTes(tokenids=train_ids,batch_size=TRAIN_CNF['batch_size'],chunk_len=TRAIN_CNF['cntext_lnth'],stride=TRAIN_CNF['cntext_lnth']//2,num_worker=4)

    # 初始输入的文字
    cn_start_cont = "这是一个小红帽的故事，从前"
    en_start_cont = "Long long ago, there is a girl "

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, trainDsloader, valDsloader, optimizer, device,
        num_epochs = num_epochs, eval_freq=16,
        start_context = cn_start_cont
    )
    return train_losses, val_losses, tokens_seen


# Windows 下 DataLoader(num_workers>0) 用 spawn 启动子进程，子进程会重新导入本模块；
# 执行代码必须放在 __main__ 守卫内，否则会重复跑训练并抛出
# 所以必须放在 __main__ 守卫内，否则会重复跑训练并抛出
# "An attempt has been made to start a new process before ..." RuntimeError。
if __name__ == "__main__":
    # multiprocessing.freeze_support()
    main()