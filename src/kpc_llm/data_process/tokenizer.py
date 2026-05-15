"""
训练数据文本文件中获取的训练数据并处理成符合训练规格的格式。
行话叫Tokenizing txt 词元化文本文件，或和叫文本分词
"""

import logging
import os 
import re
import token

from kpc_llm.data_process.textloader import getTxtStr,getVerdictTxtStr
from kpc_llm.utils.logger import getlogger

logger = getlogger()


class Tokenizer:
    VOCAB = None
    RE_VOCAB = None
    def __init__(self,fileName:str,subDir:str) -> None:
        """
        把文本训练数据的词表创建并存储到类内部
        """
        self.fileName = fileName
        self.subDir = subDir
        Tokenizer.VOCAB = Tokenizer.converTokens2Vocab(fileName,subDir)
        Tokenizer.RE_VOCAB = { id:token for token,id in Tokenizer.VOCAB.items()}


    @staticmethod
    def converTokens2Vocab(fileName:str,subDir:str) -> dict:
        """
        把训练文本转成词表
        """
        # 获取所有词汇的词表,很简单就是去重，然后按规则排序默认用sorted按照Unicode排序 step 1
        words = Tokenizer.getPreProcessed(fileName,subDir)
        vocabInner =  Tokenizer.converTokens2VocabFromStr(words)
        
        return vocabInner

    @staticmethod
    def converTokens2VocabFromStr(words:str):
        if words is not None:
            words = sorted(set(words))
            # 在词表最后增加未知字符映射，和不同文章的分割符号映射
            words.extend(["<|endoftext|>", "<|unk|>"])
            logger.info(f'getVocab fileName length : {len(words)}')
            # 给所有词汇排序编码的dict step 2
            vocabInner = { token:id for id,token in enumerate(words) }
        return vocabInner
    
    @staticmethod
    def preProcessFromStr(raw_txt:str):
        """
        预处理原始字符串 step 1
        """
        # r‘’写法让 \n \t \s 这种回车 tab 和 空格的表示失效 \就是\本身 。不用\\表示
        # 如果要用 \n  则 类似  r'123' + '\n'
        # 以所有这些特殊符号来拆分原始文本得到字符串数组
        pattern = r'([,.:;?_!"()\']|--|\s)'
        preprocess_l = re.split(pattern,raw_txt)

        # 去掉数组中的空白字符元素
        preprocess_l = [ item.strip() for item in preprocess_l if item.strip()]
       
        #处理后文本数组
        return preprocess_l

    @staticmethod
    def getPreProcessed(fileName:str,saveSubDir: str = None):
        """
        从文件中获取字符串并处理成合适进行词元处理的list step 2_1
        """
        # 从文件获取原始文本字符串
        raw_txt = getTxtStr(fileName,saveSubDir)
        return Tokenizer.preProcessFromStr(raw_txt)

    @staticmethod
    def getPreTheVerdictProcessed():
        """
        从文件中获取字符串并处理成合适进行词元处理的list step 2_1
        """
        # 从文件获取原始文本字符串
        raw_txt = getVerdictTxtStr()
        return Tokenizer.preProcessFromStr(raw_txt)


    def encode(self):
        """
        把训练数据的文章的token变成 token ID，一个token ID 相当于词表这个张量空间里的一个分量。
        最终把文章转换成一个张量空间，可以输入 LLM的layers进行训练，行话这种张量的输入不叫输入叫嵌入到训练模型层。
        所以嵌入训练数据必须是首先把数据处理成张量形式的。
        """

        # 把文章分词数组变成 toke ID的 数组 step 1
        #VOCAB.items() 方法把dict 变成 可迭代对象
        tokens = Tokenizer.getPreProcessed(self.fileName,self.subDir)
        logger.info(f' Tokenizer.VOCAB year : {Tokenizer.VOCAB.get('year')}')
        tokenIds = [Tokenizer.VOCAB.get(word) for word in tokens ]
        return tokenIds

    def encodeFromStr(self,words:str):
        """
        把训练数据的文章的token变成 token ID，一个token ID 相当于词表这个张量空间里的一个分量。
        最终把文章转换成一个张量空间，可以输入 LLM的layers进行训练，行话这种张量的输入不叫输入叫嵌入到训练模型层。
        所以嵌入训练数据必须是首先把数据处理成张量形式的。
        """

        # 把文章分词数组变成 toke ID的 数组 step 1
        #VOCAB.items() 方法把dict 变成 可迭代对象
        tokens = Tokenizer.preProcessFromStr(words)
        logger.info(f' Tokenizer.VOCAB year : {Tokenizer.VOCAB.get('year')}')
        tokenIds = [Tokenizer.VOCAB.get(word) for word in tokens ]
        return tokenIds

    def decode(self,tokeIds:list) -> list:
        """
        通过tokenId 返回 Token列表 ,这里增加了对未知单词的处理，
        """
        tokens = [Tokenizer.RE_VOCAB.get(tokenId) if tokenId in Tokenizer.RE_VOCAB else '<|unk|>' for tokenId in tokeIds]
        return tokens
    
    def decode2Txt(self,tokeIds:list) -> str:
        """
        通过Token列表 返回 文本
        """
        tokens = self.decode(tokeIds)
        outTxt = " ".join(tokens)
        #正则表达式，所有特殊标点符号前面的空格都用替换工具去掉
        outTxt = re.sub(r'\s+([,.?!"()\'])', r'\1', outTxt)
        return outTxt


