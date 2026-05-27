"""
接受tokenLoader输出的Dataset，最终变成模型可训练使用的输入层
输入嵌入层 = tokenid嵌入层 + 输入序列窗口位置嵌入层
"""

import torch
from torch.utils import data
from torch.utils.data import DataLoader

from kpc_llm.data_process import tiktokenizer
from kpc_llm.utils import getlogger
from kpc_llm.data_process.textloader import getTxtStr
from kpc_llm.data_process.token_loader import create_dataloader_1

logger = getlogger()

class TokenId2InputEnbedding():
    def __init__(self,) -> None:
        pass

    @staticmethod
    def buildLayerInputEnbedding(dataloader:DataLoader,embedding_dim):
        data_iter = iter(dataloader)
        #获取输入的tokenid的torch.tensor
        token_input,data_target = next(data_iter)
        logger.info(f'token_input.shape : {token_input.shape}')
        #给位置input enbedding初始化用的 chunk块的长度，因为还没有扩展embedding_dim，所以这里是shape的倒数第1个维度，。
        chunk_len = token_input.shape[-1]
        #获取tiktoken的gpt2的tokenizer的词表的size
        vocab_size = tiktokenizer.tiktokenizer.n_vocab
        #全词表的嵌入层的随机初始化，随机抽样的分向量服从正态分布，方差是 1/embedding_dim ：1/嵌入维度
        token_embedding_layer = torch.nn.Embedding(vocab_size, embedding_dim)
        logger.info(f'vocab_size shape : {vocab_size}')
        #这里做了 one-hot的点积，做了look up 操作
        token_input_embedding = token_embedding_layer(token_input)
        
        logger.info(f'token_input shape : {token_input.shape}')
        pos_embedding_layer = torch.nn.Embedding(chunk_len, embedding_dim)
        #这里torch.arange(tokens_len)的shape本质上跟token_input一样，但是0到tokens_len的排序list
        postion_input_embedding = pos_embedding_layer(torch.arange(chunk_len))
        logger.info(f'torch.arange(tokens_len) : {torch.arange(chunk_len)}')
        logger.info(f'token_input_embedding.shape : {token_input_embedding.shape}')
        logger.info(f'postion_input_embedding.shape : {postion_input_embedding.shape}')
        #这里会自动广播机制全加
        layer_input_enbedding = token_input_embedding + postion_input_embedding
        return layer_input_enbedding


if __name__ == '__main__':
    txt = getTxtStr('the-verdict.txt','data')
    dataloader = create_dataloader_1(txt,batch_size=8,chunk_len=4,stride=4)
    layer_input_enbedding = TokenId2InputEnbedding.buildLayerInputEnbedding(dataloader,256)
    logger.info(f'layer_input_enbedding shape : {layer_input_enbedding.shape}')
    logger.info(f'layer_input_enbedding : {layer_input_enbedding}')
