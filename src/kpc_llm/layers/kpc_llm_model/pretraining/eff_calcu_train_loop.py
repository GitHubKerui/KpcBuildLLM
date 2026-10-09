import time  # 1. 引入时间模块
from accelerate import Accelerator
from kpc_llm.layers.kpc_llm_model.kpc_llm_model import KpcLLMModel
from kpc_llm.layers.kpc_llm_model.pretraining.topk_saver import TopKSaver
from kpc_llm.layers.kpc_llm_model.pretraining.train_loss_calcu import calcuOneBatchCrossEnLoss
from kpc_llm.layers.kpc_llm_model.token_process.tokenizer_hub import tiktokenizer2idsUnsq,tokenizer2txtsSq
from kpc_llm.layers.kpc_llm_model.pretraining.train_data_div3_load import divDatas2TraValTes
from kpc_llm.layers.kpc_llm_model.pretraining.generate_and_print_sample import generate_and_print_sample
from kpc_llm.layers.kpc_llm_model.pretraining.evaluate_model import evaluate_model
from kpc_llm.data_fetch.textloader import getTxtStr
from kpc_llm.utils import val_plot
from kpc_llm.utils.logger import getlogger
from kpc_llm.utils.val_plot import plot_loss
from kpc_llm.layers.kpc_llm_model.train_cfg import GPT2_cl100k_base_CNF
from dataclasses import asdict
from pathlib import Path
from kpc_llm.layers.kpc_llm_model.pretraining.resume_training import ResumeManager
import math
import shutil
import tiktoken
import torch
# from transformers import AutoTokenizer
# from torch.utils.data import dataloader
# import multiprocessing


logger = getlogger()

# 配置数据类转字典
TRAIN_CNF = asdict(GPT2_cl100k_base_CNF)

# 写配置
# TRAIN_CNF = {
#     "vcab_sz" : 100277,
#     "cntext_lnth" : 256,
#     # 最大上下文位置长度，用于生成上下文位置id,必须大于等于上面的最大上下文长度
#     "max_cntxt_pstion_lnth" : 256,
#     "emb_dim" : 512,
#     "heads_num" : 8,
#     "trnsf_blocks_num" : 8,
#     "drop_rt" : 0.1,
#     "qkv_bias" : False,
#     "batch_size":16,
#     # 是返回softmax还是返回logits的argmax最大值的index值,也就是 tokenid
#     "returnSoftmax" : False
# }
"""
这是英文的语料，效果不错。一轮20mb就语法基本通顺了。 
"""
# gpt2 tiktokenizer cl100k_base vocab_size 100277
tokenizer =tiktoken.get_encoding("cl100k_base") 
# dataFileName = "the-verdict.txt" ，数据存放的文件夹
dataDoc = "data"
# 训练数据文件名
dataFileName = "tinystories_200mb.txt"
# 开始的文字
start_context = "Long long ago, there is a girl "
# 生成的文字长度
generate_txt_len = 250
# 训练的轮数
num_epochs = 1
# 每多少批次评估一次（同时决定早停判定的粒度）。
# 取 50 太密：val 噪声大，早停容易被一次抖动误判；250 让 val 更稳、更接近趋势。
eval_freq = 250
# 生成的温度
temperature = 0.8
# 生成的topk
top_k = 5


# ===========================================================================
# Accelerate / Top-K / 优化 配置
# ===========================================================================

# 混合精度策略。显式写 "no"（= fp32） 'bf16'= bf16 混合精度
ACCEL_MIXED_PRECISION = "bf16"

# --- 学习率调度（warmup + cosine）与梯度裁剪 ---
# 峰值学习率：warmup 升到它，再余弦衰减到 LR_PEAK * LR_MIN_RATIO
LR_PEAK = 1e-3
# warmup 步数占计划总步数的比例（前这一段线性升温，避免早期大 lr 抖动）
LR_WARMUP_RATIO = 0.02
# 余弦衰减的下限比例（最低降到峰值的 1/10）
LR_MIN_RATIO = 0.1
# 梯度裁剪的最大范数；设为 0/None 则关闭
GRAD_CLIP_NORM = 1.0

# 最优权重目录：checkpoints/topk_safetensors/kpcModel/ 下每份最优权重一个子目录，
# 内含 model.safetensors（Accelerate 的 save_model 默认走 safetensors，无 pickle 风险）
TOPK_SAVE_DIR = "checkpoints/topk_safetensors"
# 保留最优的几份（超出就淘汰并删文件）
TOPK_KEEP = 3
# 连续多少次 val_loss 没刷新最优就早停
TOPK_EXIT_N = 10
# 早停的"改善阈值"：val_loss 要比历史最优再低至少这么多才算刷新，否则只当噪声
TOPK_MIN_DELTA = 1e-3
# 早停至少等训练跑满这么多个 epoch 才允许触发（0 = 不设门槛）。
# 用于避免训练早期（val 还很高、噪声最大）被误判掐断。
TOPK_MIN_EPOCHS_BEFORE_STOP = 1

# 只保留最近几份存档（模型 128M 参数，每份约 0.5GB，份数与磁盘开销需要权衡）
STATE_KEEP = 3

# 断点续训（开关 RESUME_ENABLE / 子目录 SAVE_SUBDIR、LOAD_SUBDIR / 存档查找与加载）
# 见 pretraining/resume_training.py：目录结构为 <accelerate 总目录>/<子目录>/epoch_XXX，
# 总目录来自 configs/project_config.toml 的 [training].accelerate_save_dir。

""" 
这是中文训练的语料模型 
"""
# InternLM2.5-1.8B vocab_size = 92550
# model_id = "internlm/internlm2_5-1_8b-chat"
# model_id = here() / "data" / "tokenizer" / "internlm2_5"
# tokenizerTrain = AutoTokenizer.from_pretrained(
#     model_id,
#     trust_remote_code=True,
#     use_fast=False,
# )
# dataFileName = "corpus_zh.txt"
# dataDoc = "data"
# start_context = "这是一个小红帽的故事，从前 "


def _build_warmup_cosine_lambda(total_steps: int, warmup_ratio: float, min_ratio: float):
    """构造 warmup + cosine 衰减的 lr_lambda（供 LambdaLR 使用）。

    - 前 warmup_ratio*total_steps 步：从 ~0 线性升到 1.0（= optimizer 的基准 lr）
    - 之后：余弦从 1.0 平滑降到 min_ratio
    内部用 step+1 计算，保证第一次 optimizer.step 拿到的不是 0 学习率。
    """
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    decay_steps = max(1, total_steps - warmup_steps)

    def lr_lambda(step: int) -> float:
        s = step + 1
        if s <= warmup_steps:
            return s / warmup_steps
        progress = min(1.0, (s - warmup_steps) / decay_steps)
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def train_model_simple(model, train_loader, val_loader, optimizer, accelerator,
                       train_meta, topk_saver, state_dir, num_epochs,
                       eval_freq, start_context, context_len, scheduler=None):
    """
    训练主循环。保持原生 PyTorch 的双层 for 结构，Accelerate 不接管循环逻辑。

    设备统一由 accelerator.device 提供，不再单独传 device —— 两条设备通道并存
    容易在多卡时打架，所以只留 accelerate 这一个来源。

    Args:
        model:        已经过 accelerator.prepare() 的模型
        train_loader / val_loader: 已经过 prepare()（batch 已自动搬到 device）
        optimizer:    已经过 prepare() 的优化器
        accelerator:  提供 device / backward / unwrap_model / save_state
        train_meta:   dataloader 元数据（batch_num 等），必须在 prepare 之前取好
        topk_saver:   最优权重存档 + 早停判定
        state_dir:    断点存档目录；None 则不存档
        scheduler:    已经过 prepare() 的学习率调度器；None 则不调度（lr 恒定）
    """
    # 初始化跟踪训练的参数
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    # 训练集 loss 的滑动平均（EMA），避免评估时再去迭代 train_loader
    train_loss_ema = None
    # 记录最后一次评估时的 global_step，用于在每个 epoch 结束时兜底
    last_eval_step = -2

    # 初始化时间和Token记录变量，用于计算吞吐量
    last_time = time.time()
    last_tokens_seen = 0

    # 早停标志位。原来是直接在内层循环里 break，但那只跳出了 batch 循环，
    # 外层 epoch 循环会接着跑 —— "连续 exit_n 次没改善就停"其实一直没有生效。
    # 改成置标志位 + 在外层循环开头统一退出。
    should_stop_training = False

    # 设备只从 accelerate 取这一次。prepare() 之后 batch 已经在 device 上，
    # 所以 calcuOneBatchCrossEnLoss 内部那句 .to(device) 是幂等 no-op。
    device = accelerator.device

    # 主训练循环
    for epoch in range(num_epochs):
        if should_stop_training:
            logger.info(f"触发早停，结束训练（已跑完 {epoch} 个 epoch）")
            break
        # 开启模型训练模式
        model.train()  
        # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
        # tokenizer = tiktoken.get_encoding('gpt2')

        # dataloader 元数据由 training_model 在 prepare 之前取好后传入。
        # 原因：prepare() 会把 dataloader 包成 DataLoaderShard，实测 batch_size
        # 会变成 None（取不到），dataset.chunk_len 虽仍可透传但不该依赖这种行为。
        batch_num   = train_meta["batch_num"]                 # ← 全部 batch 的数量
        sample_num  = train_meta["sample_num"]                # 数据集样本总数
        one_sample_token_num  = train_meta["chunk_len"]       # 每个样本的 token 数
        batch_size  = train_meta["batch_size"]                # 每个 batch 的样本数
        token_num   = sample_num * one_sample_token_num   # 一个 epoch 覆盖的 token 总数

        # 如果 eval_freq 比 batch_num 还大，一个 epoch 内永远触发不了第二次评估，
        # 自动把它降到 batch_num，保证每个 epoch 至少能有一次常规评估。
        effective_eval_freq = max(1, min(eval_freq, batch_num))
        
        logger.info(f"-------------------- Ep {epoch+1}/{num_epochs} Training (batch_num={batch_num}) ------------------: ")
        logger.info(f"--每个样本的token数：{one_sample_token_num}，每批次样本数: {batch_size},总共多少批次: {batch_num},一个Epoch覆盖的token总数: {token_num}--: ")
        for i,(input_batch, target_batch) in enumerate(train_loader):

            # print(f"-------------------- Ep {epoch+1} Batch {i+1} Training -------------------: ")
            # 每次学习前，重置之前的内存梯度为0
            optimizer.zero_grad() 
            loss = calcuOneBatchCrossEnLoss(input_batch, target_batch, model, device)
            # 计算损失的梯度。用 accelerator.backward 而不是 loss.backward()：
            # 单卡 + fp32 下两者数值等价，但这是 accelerate 的正式入口
            # （混合精度要靠它内部的 GradScaler，梯度累积也要靠它做梯度同步）。
            # 注意 zero_grad / backward / step 的原有顺序保持不变。
            accelerator.backward(loss)
            # 梯度裁剪：必须在 backward 之后、step 之前。用 accelerator 的版本，
            # 它会先反缩放 bf16 的梯度再裁剪，等价于 fp32 下的 clip_grad_norm_。
            if GRAD_CLIP_NORM:
                accelerator.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            # 根据梯度更新weights
            optimizer.step() 
            # 学习率调度：必须在 optimizer.step() 之后调用（AcceleratedScheduler 的要求）
            if scheduler is not None:
                scheduler.step()
            # 计算模型token的总的训练量，也就是人的读书的字数
            tokens_seen += input_batch.numel()
            global_step += 1
            # 把loss作为tensor的数据从state_dict中切割，只取标量
            loss_val = loss.item()
            # 维护训练集 loss 的滑动平均，比较经典的1/1-0.9 = 10 ，最新的loss只是加权平均的1/10 ，这样可以消除loss的抖动做展示。
            train_loss_ema = loss_val if train_loss_ema is None else 0.7 * train_loss_ema + 0.3 * loss_val

            # 1.训练评估步骤， 每 effective_eval_freq 次batch，对训练进行评估
            if global_step % effective_eval_freq == 0:
                # 获取评估数据。传 unwrap_model 后的裸模型：多卡下 prepare() 返回的
                # 是 DDP 包装，unwrap 后 state_dict 的键才不带 module. 前缀。
                train_data_loss, val_data_loss = evaluate_model(
                    accelerator.unwrap_model(model), val_loader, device,
                    cur_train_loss=train_loss_ema)

                # 2. 记录评估数据
                train_losses.append(train_data_loss)
                val_losses.append(val_data_loss)
                track_tokens_seen.append(tokens_seen)
                last_eval_step = global_step
                
                # 3. 计算训练速度/性能的逻辑(tokens/s)
                current_time = time.time()
                time_elapsed = current_time - last_time       # 距离上次评估过去了多少秒
                tokens_processed = tokens_seen - last_tokens_seen # 这期间一共处理了多少Token
                
                # 4. 防止极其罕见的除以0情况（例如 eval_freq 设得极小且运行极快）
                tokens_per_sec = tokens_processed / time_elapsed if time_elapsed > 0 else 0
                
                # 5. 在日志中打印
                # 注意：进度要用 epoch 内的 i+1 除以总批次数 batch_num；
                # global_step 是跨 epoch 累计的，用它算进度会超过 100%
                logger.info(f"Ep {epoch+1}/{num_epochs} (BatchAll : {batch_num}) "
                      f"(Batch now : {i+1:06d}/{batch_num:06d}): "
                      f"input_batch.shape: {input_batch.shape} target_batch.shape: {target_batch.shape} "
                      f"Train loss: {train_data_loss:.3f}, Val loss: {val_data_loss:.3f} | "
                      f"Speed: {tokens_per_sec:.0f} tokens/s "
                      f"\nThis Epoch Finish: {(i+1)/batch_num * 100:.3f}% "
                      f"| Global Step: {global_step}"
                      )
                
                # 6. 更新基准线，为下一次计算做准备
                last_time = current_time
                last_tokens_seen = tokens_seen

                # 寻求最优的模型。TopKSaver 内部走 accelerator.save_model 存 safetensors，
                # 且在 __init__ 里做了 register_for_checkpointing，所以这份 topk 记录会
                # 跟着 epoch 尾的 save_state 一起存档，断点续训时自动带回。
                # 返回 True = 连续 exit_n 次 val_loss 未刷新最优 -> 置标志位跳出双层循环
                if topk_saver.step(accelerator.unwrap_model(model), train_data_loss,
                                   val_data_loss, epoch, i, last_tokens_seen):
                    should_stop_training = True
                    break

        # 兜底：每个 epoch 结束时如果还没评估过，就强制评估一次，保证至少 num_epochs 个点能画图。
        # 当 effective_eval_freq 已经兜底降到 batch_num 时，通常会在这里触发一次 epoch-end 评估。
        if last_eval_step != global_step:
            train_data_loss, val_data_loss = evaluate_model(
                accelerator.unwrap_model(model), val_loader, device,
                cur_train_loss=train_loss_ema)
            train_losses.append(train_data_loss)
            val_losses.append(val_data_loss)
            track_tokens_seen.append(tokens_seen)
            last_eval_step = global_step
            # 同样更新时间基准，避免下一 epoch 的 speed 把中间等待时间算进去
            last_time = time.time()
            last_tokens_seen = tokens_seen

        # Print a sample text after each epoch
        # 传 unwrap_model 后的裸模型。注意该函数内部会 model.eval() 但不恢复 train 模式，
        # 靠下一个 epoch 开头的 model.train() 抵消 —— 这个时序保持不变，不要额外添加。
        generate_and_print_sample(
            accelerator.unwrap_model(model), device, context_len, start_context,
            tiktokenizer2idsUnsq, tokenizer2txtsSq, tokenizer, generate_txt_len,
            temperature,top_k

        )

        # 断点存档：每个 epoch 一份完整训练状态 —— 模型 + 优化器 + 随机数种子 + topk 记录
        # （topk 记录走 register_for_checkpointing，存在 custom_checkpoint_0.pkl）。
        # save_state 需要在所有 rank 同步调用，单卡下无此问题。
        # 只保留最近 STATE_KEEP 份；ignore_errors 是因为 Windows 下目录正被占用时
        # 不该让训练崩在这里 —— 清理存档只是 housekeeping。
        if state_dir:
            # 存档序号取"现有最大编号 + 1"，而不是直接用 epoch。
            # 原因：从存档恢复时 epoch 计数会归零（模型/优化器/随机数/topk 都恢复了，
            # 但 epoch 与 global_step 是局部变量、不在存档里），若拿 epoch 当序号，
            # 恢复后第一次存档就会写回 epoch_000 —— 正好覆盖正在用作恢复源的目录。
            _existing = [int(p.name.split("_")[1]) for p in Path(state_dir).glob("epoch_*")
                         if p.name.split("_")[1].isdigit()]
            _seq = max(_existing) + 1 if _existing else 0
            save_path = Path(state_dir) / f"epoch_{_seq:03d}"
            accelerator.save_state(str(save_path))
            # 按存档序号保留最近 STATE_KEEP 份（排在前面的就是最旧的）
            stale = sorted(Path(state_dir).glob("epoch_*"))[:-STATE_KEEP]
            for stale_dir in stale:
                shutil.rmtree(stale_dir, ignore_errors=True)
            logger.info(f"[state] 已存档 {save_path.name}（含 topk 记录，只保留最近 {STATE_KEEP} 份）")

    return train_losses, val_losses, track_tokens_seen


def training_model(num_epochs=1):
    """数据准备 + 装配 accelerate + fit。

    断点续训不由参数驱动，而由 pretraining/resume_training.py 统一负责：
    开关 RESUME_ENABLE、子目录 SAVE_SUBDIR / LOAD_SUBDIR 都在那里，
    其生效范围是 configs/project_config.toml 里配置的 accelerate 总目录之下。

    Args:
        num_epochs: 训练轮数
    """
    torch.manual_seed(825)
    logger.info (f"cuda is {torch.cuda.is_available()}")

    """ Accelerate 绝招：使用一个 accelerator.prepare() 方法，它会自动把你的 DataLoader 切成几份（分片），
        分给不同的 GPU 独立去跑，并完美处理底层的梯度聚合。 解决Torch原生的Distribute Data Parallel的数据切片
        并行计算配置的事情
        """
    # 设备与精度交给 accelerate。不再手动 torch.device(...) + model.to(device)
    # —— prepare() 内部会搬。device 由 train_model_simple 内部从 accelerator.device 取，
    # 这里不重复取一份，避免两条设备通道并存。
    accelerator = Accelerator(mixed_precision=ACCEL_MIXED_PRECISION)

    model = KpcLLMModel(TRAIN_CNF)

    #学习率较低0.0001，高权重w衰减率0.1较大，这种配置合适保持原有参数特征，避免新特征过拟合。合适原有参数上的FineTune 微调
    # optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.1)
    # 学习率较低0.001，高权重哦衰减率0.01较小，这种配置可以快速收敛，保证了对新数据的不被过渡泛化导致欠拟合。合适新数据的首次Training
    # LLM训练最佳选择，AdamW，不合适用Adam，Adam容易在梯度突变的高难度学习step中可能导致梯度爆炸的灾难。
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_PEAK, weight_decay=0.01)

    train_txt = getTxtStr(dataFileName,dataDoc)
    logger.info(f"train_txt length : {len(train_txt) if train_txt is not None else 0}")
    # 82 万 id 只占约 6.6MB，放 GPU 没有收益，且 DataLoader 的 spawn 子进程读不到 CUDA 张量；
    # reshape(-1) 把 tiktokenizer2idsUnsq 返回的 [1, token_num] 摊平成一维
    train_ids = tiktokenizer2idsUnsq(train_txt, tokenizer).reshape(-1).cpu()
    trainDsloader,valDsloader,testDsloader = divDatas2TraValTes(tokenids=train_ids,batch_size=TRAIN_CNF['batch_size'],chunk_len=TRAIN_CNF['cntext_lnth'],stride=TRAIN_CNF['cntext_lnth']//2,num_worker=4)

    # 初始输入的文字
    # cn_start_cont = "这是一个小红帽的故事，从前"
    # en_start_cont = "Long long ago, there is a girl "

    # dataloader 元数据必须在 prepare 之前取好：prepare 会把 dataloader 包成
    # DataLoaderShard，实测 batch_size 会变成 None（取不到），dataset.chunk_len
    # 虽仍能透传但不该依赖这种行为。顺带这几个值原本每个 epoch 重算一次，
    # 移到循环外只算一次，循环内也不再混杂元数据探测。
    # 注：dataset 的静态类型标注是 Dataset[协议]，但 create_dataloader 返回的实际是
    # KpcLLMData —— 它有 __len__ 和自定义的 chunk_len 属性（tokenid_ds_loader.py 里定义），
    # 这些都不在 Dataset 协议里，静态检查看不到，故下面两处需要 type: ignore。
    _train_dataset = trainDsloader.dataset
    train_meta = {
        "batch_num": len(trainDsloader),
        "sample_num": len(_train_dataset),                      # type: ignore[arg-type]
        "chunk_len": _train_dataset.chunk_len,                  # type: ignore[attr-defined]
        "batch_size": trainDsloader.batch_size,
    }

    # 学习率调度：warmup + cosine。计划总步数用一个 epoch 的 batch 数 × epoch 数估算；
    # 早停导致的实际步数通常少于此值，只是 lr 不会衰减到底，影响可接受。
    total_steps = max(1, train_meta["batch_num"] * num_epochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=_build_warmup_cosine_lambda(total_steps, LR_WARMUP_RATIO, LR_MIN_RATIO),
    )
    _warmup_steps = max(1, int(total_steps * LR_WARMUP_RATIO))
    logger.info(f"学习率调度：峰值 {LR_PEAK}，warmup {_warmup_steps} 步，"
                f"余弦衰减到 {LR_PEAK * LR_MIN_RATIO:.2e}（计划总步数 {total_steps}）")

    # 一次 prepare 五个对象：模型搬设备、优化器包装、dataloader 分片 + batch 自动上设备、
    # 调度器随优化器一起包成 AcceleratedScheduler（混合精度/梯度累积下才能正确步进）
    model, optimizer, trainDsloader, valDsloader, scheduler = accelerator.prepare(
        model, optimizer, trainDsloader, valDsloader, scheduler)

    # 早停至少等跑满 TOPK_MIN_EPOCHS_BEFORE_STOP 个 epoch 才允许触发。
    # evals_per_epoch 的算法要与 train_model_simple 里 effective_eval_freq 的钳制保持一致。
    _eff_eval_freq = max(1, min(eval_freq, train_meta["batch_num"]))
    _evals_per_epoch = max(1, train_meta["batch_num"] // _eff_eval_freq)
    _min_evals_before_stop = int(TOPK_MIN_EPOCHS_BEFORE_STOP * _evals_per_epoch)

    # TopKSaver 在 __init__ 里 register_for_checkpointing(self)，因此在epoch 尾调用
    # save_state 时，这份 topk 记录会一并存成 custom_checkpoint_0.pkl
    topk_saver = TopKSaver(accelerator, topk=TOPK_KEEP, exit_n=TOPK_EXIT_N,
                           min_delta=TOPK_MIN_DELTA,
                           min_evals_before_stop=_min_evals_before_stop,
                           root_dir=TOPK_SAVE_DIR)
    logger.info(f"早停设置：exit_n={TOPK_EXIT_N}，min_delta={TOPK_MIN_DELTA}，"
                f"至少 {_min_evals_before_stop} 次评估后才允许早停")

    # load_state 必须在 prepare 与 register 之后：否则恢复的是未包装的原始状态，
    # 对不上 AcceleratedOptimizer。
    #
    # 恢复范围与边界（别误以为它能无缝续跑）：
    #   恢复了 —— 模型权重、优化器状态、dropout 等随机数种子、topk 记录与早停计数
    #   没恢复 —— epoch 与 global_step。它们是 train_model_simple 的局部变量，不在存档里，
    #             所以恢复后从 epoch 0 重新计数（存档序号会接着往下排，不会覆盖恢复源）。
    # 这在本项目可接受：create_dataloader 默认 shuffle=False，数据顺序本来就确定，
    # 从头迭代喂的是同一批数据、顺序一致，不会喂错；随机数恢复的价值在 dropout
    # （drop_rt=0.1）这类模型内部的随机性。若将来改成 shuffle=True，就必须额外
    # register_for_checkpointing(train_loader) 才能真正从头续跑。
    # 断点续训：目录定位 + 恢复决策 + 加载，全交给 ResumeManager（见 resume_training.py）。
    # save_dir() 会创建本次任务的存档目录（<accelerate 总目录>/<子目录>）；
    # try_resume() 在开关关闭 / 无可用存档 / 加载失败时返回 False，均安全退回从头训练。
    resumer = ResumeManager()
    state_dir = resumer.save_dir()
    logger.info(f"本次任务存档子目录：{state_dir}")
    resumer.try_resume(accelerator)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, trainDsloader, valDsloader, optimizer, accelerator,
        train_meta, topk_saver, state_dir,
        num_epochs, eval_freq=eval_freq,
        start_context = start_context,
        context_len=TRAIN_CNF['cntext_lnth'],
        scheduler=scheduler,
    )
    return train_losses, val_losses, tokens_seen

# Windows 下 DataLoader(num_workers>0) 用 spawn 启动子进程，子进程会重新导入本模块；
# 执行代码必须放在 __main__ 守卫内，否则会重复跑训练并抛出
# 所以必须放在 __main__ 守卫内，否则会重复跑训练并抛出
# "An attempt has been made to start a new process before ..." RuntimeError。
if __name__ == "__main__":
    # multiprocessing.freeze_support()
    # 断点续训由 pretraining/resume_training.py 控制（RESUME_ENABLE / SAVE_SUBDIR / LOAD_SUBDIR）；
    # accelerate 总保存目录在 configs/project_config.toml 的 [training].accelerate_save_dir。
    train_losses, val_losses, tokens_seen = training_model(num_epochs)
    # plot_loss 的第二个参数必须是与 loss 等长的"累计 token 数"序列（给第二条 X 轴对齐刻度用）。
    # 传标量 len(tokens) 会让 matplotlib 直接抛错：
    # ValueError: x and y must have same first dimension, but have shapes (1,) and (N,)
    # 第一个参数是 epoch 轴，值域应为 0 -> num_epochs（不是评估次数）
    tokens_seen_num =tokens_seen[-1]

    epochs = torch.linspace(0, num_epochs, len(train_losses))
    tokens = torch.linspace(0, tokens_seen_num, len(train_losses))
    logger.info(f"loss 点数: {len(train_losses)} | 学习过的token数量: {tokens_seen_num}")
    plot_loss(epochs, tokens, train_losses, val_losses)