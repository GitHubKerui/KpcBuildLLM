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

class KpcLLMData(Dataset):
    def __init__(self,text,tokenizer,chunk_len,stride) -> None:
        super().__init__()
        self.input_ids = []
        self.target_ids = []
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

def create_dataloader_1(txt,batch_size=4,chunk_len=256,stride=128,shuffle=False,drop_last=True,num_worker=0):
    #用tiktoken的tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #用Kpc的Dataset
    kpcLLLData = KpcLLMData(txt,tokenizer,chunk_len,stride)
    
    dataLoader = DataLoader(
        dataset=kpcLLLData,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_worker
    )
    return dataLoader

def test_creat():
    txt = getTxtStr('the-verdict.txt','data')
    dataloader = create_dataloader_1(txt,batch_size=8,chunk_len=4,stride=4)
    data_iter = iter(dataloader)
    data = next(data_iter)
    logger.info(f'测试查看最终dataLoader的数据样式1 : {data}')
    second_batch = next(data_iter)
    logger.info(f'测试查看最终dataLoader的数据样式2 : {second_batch}')

if __name__ == "__main__":
    test_creat()