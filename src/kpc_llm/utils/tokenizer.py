"""
训练数据文本文件中获取的训练数据并处理成符合训练规格的格式。
行话叫Tokenizing txt 词元化文本文件，或和叫文本分词
"""

import os 
import re
from kpc_llm.utils.dataloader import getTxtStr,getVerdictTxtStr
from kpc_llm.utils.logger import getlogger

logger = getlogger()

class Tokenizer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def preProcess(raw_txt:str):
        """
        预处理原始字符串 step1
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
        从文件中获取字符串并处理成合适进行词元处理的list step2
        """
        # 从文件获取原始文本字符串
        raw_txt = getTxtStr(fileName,saveSubDir)
        return Tokenizer.preProcess(raw_txt)

    @staticmethod
    def getPreTheVerdictProcessed():
        """
        从文件中获取字符串并处理成合适进行词元处理的list step2
        """
        # 从文件获取原始文本字符串
        raw_txt = getVerdictTxtStr()
        return Tokenizer.preProcess(raw_txt)


if __name__ == "__main__":
    """
    测试用
    """
    try:
        words = Tokenizer.getPreProcessed('the-verdict.txt','data')
        logger.info(f'处理后的字符串长度 : {len(words)}')
        logger.info(f'测试看看处理后的前10个字符串 : {words[:10]}')
        logger.info(f'测试成功')
    except Exception as e :
        logger.debug(f'出现错误 : {e}',exc_info=True)

