

from torch.utils.data import Dataset
import torch  
import pandas as pd

from kpc_llm.test.algorithm_test import tiktokenizer


class SpamDataset(Dataset):
    def __init__(self,csv_file,tokenizer, max_length=None,pad_token_id=50256) -> None:
        # 获取保存到csv中的数据
        self.data = pd.read_csv(csv_file)
        # 转成tokenid
        self.en_tokenids = [tokenizer.encode(text) for text in self.data["Text"]]
        
        # 如果没给最长seq则设置
        if max_length is None:
            max_length = self._longest_encoded_length()
        
        self.max_length = max_length
        # 超出长度的序列直接斩断
        self.en_tokenids = [ ele[:max_length] for ele in self.en_tokenids]
        # 不够长度的后面填充padding的id
        self.en_tokenids = [ele + [pad_token_id] * (max_length - len(ele)) for ele in self.en_tokenids]

    def _longest_encoded_length(self):
        # 返回token序例里的最长序例长度
        return max((len(token_seq) for token_seq in self.en_tokenids))

    def __getitem__(self, index):
        label = self.data.iloc[index]["Label"]
        tokenids = self.en_tokenids[index]
        return (
            torch.tensor(tokenids, dtype=torch.long),
            torch.tensor(int(label), dtype=torch.long)
            )
    
    def __len__(self):
        return len(self.data)