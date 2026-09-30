"""下载中文预训练语料到 data/corpus_zh.txt。

用法：
    python src/kpc_llm/data_process/corpus_zh_download.py          # 默认 10 MB
    python src/kpc_llm/data_process/corpus_zh_download.py 30       # 指定 30 MB
"""
from kpc_llm.data_fetch.getTrainTxtFUrl import download
from kpc_llm.utils.logger import getlogger
logger = getlogger()


if __name__ == "__main__":
    DATASET_ID = "opencsg/chinese-cosmopedia"
    SAVE_NAME = "corpus_zh_15mb.txt"
    FileSize = 15 
    # 命令行参数可能不是合法正整数，解析失败时回退到默认 10 MB
    try:
        download(DATASET_ID,SAVE_NAME,FileSize)
    except Exception:
        logger.error(f"下载获取{DATASET_ID}数据失败", exc_info=True)
    
