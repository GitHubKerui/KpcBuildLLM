
from importlib.metadata import version
import tiktoken
from kpc_llm.utils import getlogger
from torch import tensor
# 这里先默认是用效率最高的 tiktokenizer做转换 后续根据需求可以换 BEP等其他的组件

logger = getlogger()
version = version("tiktoken")

"""
这里可以选择中英文都比较合适的 cl100k_base
也可以选择gpt2 小一倍词表量 但是不合适中文 合适英文
"""
tiktokenizer = tiktoken.get_encoding('gpt2')
# tiktokenizer = tiktoken.get_encoding('cl100k_base')

# tiktokenizer 转换文本为 token_ids_tensor
def tiktokenizer2ids(txts):
    ids = tiktokenizer.encode(txts,allowed_special={'<|endoftext|>'})
    # 用 unsqueeze 增加了 batch的维度，增加的是 0 维度，所以这里的 shape 是 [1,token_num]
    ids_tensor = tensor(ids).unsqueeze(0)
    return ids_tensor   


# tiktokenizer 转换 token_ids_tensor 为文本
def tiktokenizer2txts(token_ids_tensor):
    # 这里是把0维度的batch给去掉了。
    token_ids = token_ids_tensor.squeeze(0).tolist()
    token_str = tiktokenizer.decode(token_ids)
    return token_str

if __name__=="__main__":
    logger.info(f'tiktoken 的版本 : {version}')
    logger.info(f'tiktoken 的__version__版本 : {tiktoken.__version__}')
    inputStr = "The morning sunlight is filtering through the leaves, casting dancing shadows on the ground.<|endoftext|> In thesunlit terraces of someunknownPlace."
    token_ids = tiktokenizer2ids(inputStr)
    logger.info(f'tiktokenizer2ids 的测试 : {token_ids}')
    logger.info(f'tiktokenizer2txts 的测试 : {tiktokenizer2txts(token_ids)}')