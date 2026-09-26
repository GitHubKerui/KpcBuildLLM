from logging import getLogger
from sympy.printing.pretty.pretty_symbology import B
from torch.utils.data import Dataset, DataLoader
import tiktoken 
import torch
from kpc_llm.data_process.textloader import getTxtStr
from kpc_llm.utils.logger import getlogger

"""
Step1 首先以Dataset抽象类创建自定义的Dataset类，KpcLLMData。
Step2 用DataLoader配置并加载KpcLLMData后生成最终符合标准的 InputDate 和 TargetData 对。
"""

logger = getlogger()

# KpcLLMData 类继承自 Dataset 抽象类自定义了 
# 把原始的文本数据 通过tokenizer处理成tokenid，并且按照chunk_len分成组
# chunk_len后面就是llm上下文长度，
# 并且右移一位获得target组，最终会得到 context_len,token_len 这个shape的train 和 target 两个dataset
class KpcLLMData(Dataset):
    def __init__(self,text,tokenizer,chunk_len,stride) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []
        self.chunk_len = chunk_len
        tokens_ids = tokenizer.encode(text,allowed_special={'<|endoftext|>'})

        for i in range(0,len(tokens_ids)-chunk_len,stride):
            input_ids_tensor = torch.tensor(tokens_ids[i:i+chunk_len])
            target_ids_tensor = torch.tensor(tokens_ids[i+1:i+chunk_len+1])
            self.input_ids.append(input_ids_tensor)
            self.target_ids.append(target_ids_tensor)

    def __len__(self):
        length = len(self.input_ids)
        return  length

    def __getitem__(self, idx):
        return self.input_ids[idx],self.target_ids[idx]

# 把带traindata ids的数据集 和 target的数据集，处理成批次
# 可以设置是一个批次多少组数据，否洗牌数据批次，是否丢弃最后可能不完整的批次数据，并行处理数据的线程
def create_dataloader_1(txt,batch_size=4,chunk_len=256,stride=128,shuffle=False,drop_last=True,num_worker=0):
    #用tiktoken的tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #用Kpc的Dataset
    kpcLLLData = KpcLLMData(txt,tokenizer,chunk_len,stride)
    
    dataLoader = DataLoader(
        # 只是成对的training 和target 的list tuple
        dataset=kpcLLLData,
        # batch_size 是每个batch的样本数量
        batch_size=batch_size,
        shuffle=shuffle,
        # 是否丢弃最后一个批次的数据，因为可能不完整，影响整体数据对齐
        drop_last=drop_last,
        # 开启几个线程加载数据
        num_workers=num_worker
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
    dataloader = create_dataloader_1(txt,batch_size=8,chunk_len=4,stride=4)
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