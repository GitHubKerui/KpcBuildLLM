
"""下载中文预训练语料到 data/corpus_zh.txt。

用法：
    python src/kpc_llm/data_process/corpus_zh_download.py          # 默认 10 MB
    python src/kpc_llm/data_process/corpus_zh_download.py 30       # 指定 30 MB
"""

import os
from datasets import load_dataset
from pyprojroot import here

# DATASET_ID = "opencsg/chinese-cosmopedia"
# SAVE_NAME = "corpus_zh.txt"

def download(dataset_id,save_name,target_mb: int = 10):
    root = here()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    save_path = data_dir / save_name
    target_bytes = target_mb * 1024 * 1024

    ds = load_dataset(dataset_id, split="train", streaming=True)

    doc_num = 0
    # 文本模式下 f.tell() 返回的是不透明 cookie，统计中文（多字节）体积不准确，
    # 这里显式按 UTF-8 编码后的字节数累加
    written_bytes = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for ex in ds:
            # 不同数据集字段名不统一，这里取最长的那个字符串字段当正文
            text = max((v for v in ex.values() if isinstance(v, str)),
                       key=len, default="")
            text = text.strip()
            if text:
                line = text + "\n"
                f.write(line)
                written_bytes += len(line.encode("utf-8"))
                doc_num += 1
            if written_bytes >= target_bytes:
                break

    size_mb = os.path.getsize(save_path) / 1024 / 1024
    print(f"docs : {doc_num}")
    print(f"path : {save_path}")
    print(f"size : {size_mb:.2f} MB")
