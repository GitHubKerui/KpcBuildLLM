

from torch.utils.data import Dataset
import torch  
import tiktoken
import json
from kpc_llm.utils import download_file, getlogger,get_project_root
import tiktoken
from pathlib import Path

def load_json_data(file_path):
    """ 加载数据到json对象 """
    if not file_path.exists():
        file_path = SftDataset._fetch_data()

    with open(str(file_path), "r") as file:
        data = json.load(file)
    return data

class SftDataset(Dataset):

    @staticmethod
    def _fetch_data():
        # print_pkgs_version()
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
        save_path = get_project_root() / "data" / "download"
        dwnfilepath = download_file(url,save_path)
        return dwnfilepath
    
    # 转成tokenid
    savepath = get_project_root() / "data" / "download" / "instruction-data.json"
    tokenizer = tiktoken.get_encoding("gpt2")

    def __init__(self) -> None:
        # _dataset_format 内部已经调过 load_json_data，直接返回
        self.data_txts, self.instraction_lens,self.data = self._dataset_format(SftDataset.savepath)

    def __getitem__(self, index):
        """ collate_fn 的 batch 参数里每条就是这里的返回值 """
        # prompt_len 必须是 int：包成 list 的话，collate 里拿它当切片上界会抛
        # "slice indices must be integers or None or have an __index__ method"
        return self.data[index], self.instraction_lens[index]
    
    def __len__(self):
        return len(self.data)

    def _format_input(self,entry):
        head: str =(
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request.")
        instruction = (f"\n\n### Instruction:\n{entry['instruction']}")
        input = (f"\n\n### input:\n{entry['input']}") if entry['input']  else ""
        re = head + instruction + input
        # log.info(f"\n\n{re}")
        return re

    def _format_response(self,entry):
        response = (f"\n\n### Response:\n{entry['output']}")
        return response   

    def _format_data(self,entry):
        return self._format_input(entry) + self._format_response(entry)

    def _dataset_format(self,fileSavePath):
        json_dtset = load_json_data(fileSavePath)
        dataset_fm = []
        dataset_fm_token = []
        instraction_lens = []
        if json_dtset:
            for data in json_dtset:
                input_str = self._format_input(data)
                response_str = self._format_response(data)
                # prompt 长度必须是【token 数】：len(str) 拿到的是字符数，英文约 1 token ≈ 4 字符，
                # 直接当 token 下标用会把 -100 屏蔽区间盖到 response 里去
                prompt_ids = SftDataset.tokenizer.encode(input_str)
                dataset_fm.append(input_str + response_str)
                # 分开 encode 再拼接：encode(prompt) + encode(response) 和
                # encode(prompt + response) 在 BPE 边界上可能差 1~2 个 token
                dataset_fm_token.append(prompt_ids + SftDataset.tokenizer.encode(response_str))
                instraction_lens.append(len(prompt_ids))
            return dataset_fm,instraction_lens,dataset_fm_token
        else:
            print(f"json_dtset 错误！Dataset文件地址：{fileSavePath}")
            # __init__ 解 3 个值，这里少返一个会在 json 读失败时抛 ValueError
            return [], [], []
            

# tokenid : [50256]
def print_special_token(special_token="<|endoftext|>"):
    print(f"<|endoftext|> tokenid : {SftDataset.tokenizer.encode(special_token,allowed_special={"<|endoftext|>"})}")

if "__main__" == __name__:
    print_special_token()