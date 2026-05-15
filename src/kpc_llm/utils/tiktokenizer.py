"""
tiktoken from GPT-2 
- GPT-2 used BytePair encoding (BPE) as its tokenizer
- it allows the model to break down words that aren't in its predefined vocabulary into smaller subword units or even individual characters, 
- enabling it to handle out-of-vocabulary words.
first -  uv add tiktoken
"""

import importlib
import tiktoken
from kpc_llm.utils import getlogger
import kpc_llm.utils.logger 

logger = getlogger()
version = importlib.metadata.version("tiktoken")
tiktokenizer = tiktoken.get_encoding('gpt2')


def use_sample_tiktoken_encode():
    inputStr = "The morning sunlight is filtering through the leaves, casting dancing shadows on the ground.<|endoftext|> In thesunlit terraces of someunknownPlace."
    ids = tiktokenizer.encode(inputStr,allowed_special={'<|endoftext|>'})
    return ids

def use_sample_tiktoken_decode(ids):
    token_str = tiktokenizer.decode(ids)
    return token_str



if __name__=="__main__":
    logger.info(f'tiktoken 的版本 : {version}')
    logger.info(f'use_sample_tiktoken_encode 的测试 : {use_sample_tiktoken_encode()}')
    logger.info(f'use_sample_tiktoken_decode 的测试 : {use_sample_tiktoken_decode(use_sample_tiktoken_encode())}')