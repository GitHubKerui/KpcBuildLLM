"""下载中文预训练语料到 data/corpus_zh.txt。

用法：
    python src/kpc_llm/data_process/corpus_zh_download.py          # 默认 10 MB
    python src/kpc_llm/data_process/corpus_zh_download.py 30       # 指定 30 MB
"""
import os
import sys

from datasets import load_dataset
from pyprojroot import here

DATASET_ID = "opencsg/chinese-cosmopedia"
SAVE_NAME = "corpus_zh.txt"


def download(target_mb: int = 10):
    root = here()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / SAVE_NAME
    target_bytes = target_mb * 1024 * 1024

    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    doc_num = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for ex in ds:
            # 不同数据集字段名不统一，这里取最长的那个字符串字段当正文
            text = max((v for v in ex.values() if isinstance(v, str)),
                       key=len, default="")
            text = text.strip()
            if text:
                f.write(text + "\n")
                doc_num += 1
            if f.tell() >= target_bytes:
                break

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"docs : {doc_num}")
    print(f"path : {save_path}")
    print(f"size : {size_mb:.2f} MB")


if __name__ == "__main__":
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    download(mb)
