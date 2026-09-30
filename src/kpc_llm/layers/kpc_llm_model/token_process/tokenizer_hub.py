
from importlib.metadata import version
import tiktoken
from kpc_llm.utils import getlogger
from torch import tensor
from transformers import AutoTokenizer

# 这里先默认是用效率最高的 tiktokenizer做转换 后续根据需求可以换 BEP等其他的组件


# 
# tokenizer_hub.py 里已有
def internlm25_2idsUnsq(txts, tokenizer):
    ids= tokenizer.encode(txts, add_special_tokens=False)
    ids_tensor = tensor(ids).unsqueeze(0)
    return ids_tensor   

# tiktokenizer 转换文本为 token_ids_tensor
def tiktokenizer2idsUnsq(txts,tokenizer):
    ids = tokenizer.encode(txts,allowed_special={'<|endoftext|>'})
    # 用 unsqueeze 增加了 batch的维度，增加的是 0 维度，所以这里的 shape 是 [1,token_num]
    ids_tensor = tensor(ids).unsqueeze(0)
    return ids_tensor   

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
# 151936 词表规模
def qwtokenizer2idsUnsq(txts,tokenizer):
    # tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    ids = tokenizer.encode(txts, add_special_tokens=False) 
    # 用 unsqueeze 增加了 batch的维度，增加的是 0 维度，所以这里的 shape 是 [1,token_num]
    ids_tensor = tensor(ids).unsqueeze(0)
    return ids_tensor   

# tiktokenizer 转换 token_ids_tensor 为文本
# internlm25 转换 token_ids_tensor 为文本
# qw25 tokenizer 转换 token_ids_tensor 为文本  三种的解码都一样
def tokenizer2txtsSq(token_ids_tensor,tiktokenizer):
    # 这里是把0维度的batch给去掉了。
    token_ids = token_ids_tensor.squeeze(0).tolist()
    token_str = tiktokenizer.decode(token_ids)
    return token_str

""" --------------------------------------------No Squeeze---------------------------------------------------------- """

# tiktokenizer 转换文本为 token_ids_tensor no unsqueeze
def tiktokenizer2idsNoUnSq(txts,tiktokenizer):
    ids = tiktokenizer.encode(txts,allowed_special={'<|endoftext|>'})
    return ids   

# from transformers import AutoTokenizer no unsqueeze
# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
def qwtokenizer2idsNoUnsq(txts,tokenizer):
    ids = tokenizer.encode(txts, add_special_tokens=False) 
    return ids   

# tiktokenizer 转换 token_ids 为文本 no squeeze版本
def tiktokenizer2txtsNoSq(token_ids,tiktokenizer):
    # 这里是把0维度的batch给去掉了。
    token_str = tiktokenizer.decode(token_ids)
    return token_str

# qw25 tokenizer 转换 token_ids 为文本 no squeeze版本
def tokenizer2txtsNoSq(token_ids,tokenizer):
    # 这里是把0维度的batch给去掉了。
    token_str = tokenizer.decode(token_ids)
    return token_str


def test():
    logger = getlogger()
    version2 = version("tiktoken")
    print(f"version : {version2}")
    """
    这里可以选择中英文都比较合适的 cl100k_base
    也可以选择gpt2 小一倍词表量 但是不合适中文 合适英文
    """
    tiktokenizer = tiktoken.get_encoding('gpt2')
    # tiktokenizer = tiktoken.get_encoding('cl100k_base')
    logger.info(f'tiktoken 的版本 : {version}')
    logger.info(f'tiktoken 的__version__版本 : {tiktoken.__version__}')
    inputStr = "The morning sunlight is filtering through the leaves, casting dancing shadows on the ground.<|endoftext|> In thesunlit terraces of someunknownPlace."
    token_ids = tiktokenizer2idsUnsq(inputStr,tiktokenizer)
    logger.info(f'tiktokenizer2ids 的测试 : {token_ids}')
    logger.info(f'tiktokenizer2txts 的测试 : {tiktokenizer2txtsSq(token_ids,tiktokenizer)}')

if __name__=="__main__":
    test()
    