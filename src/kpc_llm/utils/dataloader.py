import os
import urllib.request
from kpc_llm.utils.logger import getlogger
from pyprojroot import here


# 获得日志对象
logger = getlogger()
# 获得项目根目录对象 Path
root = here()
# 输出日志
logger.info(f"root_path : {root}")

def downVerdictFile():
    """
    下载并返回 the-verdict.txt 数据集的路径 step1
    """
    fileUrl = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
    return getFileFromUrl(fileUrl, "data")

def getFileFromUrl(fileUrl: str, saveSubDir: str = None):
    """
    从 URL 下载文件到本地目录，按下载名称保存 step2

    Args:
        fileUrl: 文件下载地址
        saveSubDir: 保存的子目录，相对于项目根目录名称

    Returns:
        下载文件的完整路径
    """
    if saveSubDir:
        # 指定根目录文件夹时候的保存方法
        fileLocalSavePath = root / saveSubDir / fileUrl.split("/")[-1]
    else:
        # 未指定则原名称存到根目录
        fileLocalSavePath = root / fileUrl.split("/")[-1]

    if not os.path.exists(fileLocalSavePath):
        # 数据集来源：LLMs-from-scratch 开源项目
        url = ("https://raw.githubusercontent.com/rasbt/"
               "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
               "the-verdict.txt")
        # 检索网上文档到本地，就是下载网上文档
        urllib.request.urlretrieve(url, filename=str(fileLocalSavePath))
        logger.info(f"数据集已下载到: {fileLocalSavePath}")
    else:
        logger.info(f"数据集已存在: {fileLocalSavePath}")

    return fileLocalSavePath


def getVerdictTxtStr():
    """
    从data/the-verdict.txt数据集文件读取字符串 step3
    """
    downVerdictFile()
    return getTxtStr('the-verdict.txt','data')
    
    
def getTxtStr(fileName:str,saveSubDir: str = None):
    """
    从指定的根目录的文件夹里获取txt文件内容的字符串 step4
     Args:
        fileName: 文件名
        saveSubDir: 保存的子目录，相对于项目根目录名称

    Returns:
        文件的原始文本内容
    """

    if saveSubDir:
        # 指定根目录文件夹时候的保存方法
        fileLocalSavePath = root / saveSubDir / fileName
    else:
        # 未指定则原名称存到根目录
        fileLocalSavePath = root / fileName
    
    if os.path.exists(fileLocalSavePath):
        with open(file = fileLocalSavePath,mode = 'r',encoding = 'utf-8') as file:
            rawStr = file.read()
        return rawStr
    else:
        logger.info(f"数据集不存在: {fileLocalSavePath}")
        return None
