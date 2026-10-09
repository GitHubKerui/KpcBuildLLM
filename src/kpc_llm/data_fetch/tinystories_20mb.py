""" 
这个语料跑的模型跑20mb基本能出正确的语法了。
"""
from kpc_llm.data_fetch.getTrainTxtFUrl import download
from kpc_llm.utils.logger import getlogger
logger = getlogger()

if __name__ == "__main__":
    DATASET_ID = "roneneldan/TinyStories"
    SAVE_NAME = "tinystories_200mb.txt"
    FileSize = 200 
    # 命令行参数可能不是合法正整数，解析失败时回退到默认 10 MB
    try:
        download(DATASET_ID,SAVE_NAME,FileSize)
    except Exception:
        logger.error(f"下载获取{DATASET_ID}数据失败", exc_info=True)