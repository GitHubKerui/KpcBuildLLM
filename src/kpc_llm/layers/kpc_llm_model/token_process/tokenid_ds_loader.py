from torch.utils.data import Dataset, DataLoader
import tiktoken 
import torch
from kpc_llm.data_fetch.textloader import getTxtStr
from kpc_llm.utils.logger import getlogger

"""
Step1 首先以Dataset抽象类创建自定义的Dataset类，KpcLLMData。
Step2 用DataLoader配置并加载KpcLLMData后生成最终符合标准的 InputDate 和 TargetData 对。
"""

logger = getlogger()

# KpcLLMData 类继承自 Dataset 抽象类自定义了 
# tokens_ids是tokenizer处理后的tokenid，chunk_len是一个样本的token数量，也就是llm的上下文长度
# stride是每次移动的token数量，也就是llm的步长，如果是一般的chunk_len，那么llm的训练数据量是两倍的数据集的数量
# 并且右移一位获得target组，最终会得到 context_len,token_len 这个shape的train 和 target 两个dataset
class KpcLLMData(Dataset):
    def __init__(self,tokens_ids,chunk_len,stride) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        # 本质是llm的上下文长度
        self.chunk_len = chunk_len

        # 1 增加对输入的tokenid的鲁棒性，因为输入的tokenid可能是一个tensor，也可能是一个list，也可能shape是[1,token_num]
        #   传进来的可能是 [1, token_num] 的张量，len() 只有 1，会切出空数据集；
        #   若它还在 CUDA 上，循环里每次 torch.tensor(cuda_slice) 都是一次 GPU→CPU 同步拷贝，
        #   既慢又会刷 "To copy construct from a tensor" 的 UserWarning。
        if isinstance(tokens_ids, torch.Tensor):
            # 把tokens_ids从当前计算图中分离出来，转换成1维的tensor，移动到cpu不占用GPU，并且转换成long类型
            ids = tokens_ids.detach().reshape(-1).cpu().to(torch.long)
        else:
            # 如果不是tensor，则直接转换成tensor，并且转换成long类型
            ids = torch.tensor(tokens_ids, dtype=torch.long)

        # 2 打印token的数量
        logger.info(f"token num : {ids.numel()}")

        if ids.numel() <= chunk_len:
            raise ValueError(f"token 数({ids.numel()}) 必须大于 chunk_len({chunk_len})")

        # 3 一次性获得train和target的数据集
        for i in range(0,ids.numel()-chunk_len,stride):
            self.input_ids.append(ids[i:i+chunk_len])
            self.target_ids.append(ids[i+1:i+chunk_len+1])

    def __len__(self):
        length = len(self.input_ids)
        return  length

    def __getitem__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]

# 把带traindata ids的数据集 和 target的数据集，处理成批次
# 可以设置是一个批次多少组数据，否洗牌数据批次，是否丢弃最后可能不完整的批次数据，并行处理数据的线程
def create_dataloader(tokenids,batch_size,chunk_len,stride,shuffle=False,drop_last=False,num_worker=0):

    #用Kpc的Dataset
    kpcLLLData = KpcLLMData(tokenids,chunk_len,stride)
    
    dataLoader = DataLoader(
        # 只是成对的training 和target 的list tuple
        dataset=kpcLLLData,
        # batch_size 是每个batch的样本数量
        batch_size=batch_size,
        # 是否打乱数据集的顺序
        shuffle=shuffle,
        # 是否丢弃最后一个批次的数据，因为可能不完整，影响整体数据对齐
        drop_last=drop_last,
        # 开启几个线程加载数据
        num_workers=num_worker,
        # 5800X3D + 5070 Ti 必开，加速内存到显存的传输
        # 无 CUDA 时 pin_memory 无意义，torch 会告警，故按可用设备动态决定
        pin_memory=torch.cuda.is_available(),
        # 防止每个 Epoch 重新创建线程浪费时间（num_workers=0 时必须关闭，否则 ValueError）
        persistent_workers=num_worker > 0
    )
    return dataLoader

def getDateLoaderTokenNum(dataloader:DataLoader):
    if dataloader is None:
        return 0    
    else:
        input_token_num = 0
        target_token_num = 0
        for input_data,target_data in dataloader:
            input_token_num += input_data.numel()
            target_token_num += target_data.numel()

        return input_token_num,target_token_num

def test_creat():
    txt = getTxtStr('the-verdict.txt','data')
    tokenizer = tiktoken.get_encoding("gpt2")
    tokens_ids = tokenizer.encode(txt,allowed_special={'<|endoftext|>'})
    dataloader = create_dataloader(tokenids=tokens_ids,batch_size=8,chunk_len=4,stride=4,)
    data_iter = iter(dataloader)
    data = next(data_iter)
    logger.info(f'测试查看最终dataLoader的数据样式1 : {data}')
    # second_batch = next(data_iter)
    # logger.info(f'测试查看最终dataLoader的数据样式2 shape: {second_batch}')
    # logger.info(f'测试查看最终dataLoader的数据样式2 : {second_batch}')
    # x代表一个批次的token字符串的数量8个token字符串， y代表一个token字符串里4个token
    # for x, y in dataloader:
    #     print(x.shape, y.shape)
    print( f"input_token_num,target_token_num : {getDateLoaderTokenNum(dataloader)}" )



if __name__ == "__main__":
    test_creat()