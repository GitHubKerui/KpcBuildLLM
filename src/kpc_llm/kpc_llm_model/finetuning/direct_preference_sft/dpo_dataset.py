

import json
from pathlib import Path
import pprint
import tiktoken
from torch.utils.data import Dataset

from kpc_llm.utils import download_file, get_project_root

data_path = get_project_root() / "data" / "download" / "instruction-data-with-preference.json"
tokenizer = tiktoken.get_encoding("gpt2")
class PreferDataset(Dataset):
    def __init__(self,data) -> None:
        self.data = data
        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            # 这的prompt输出可能不一样 注意！！！！！！！
            prompt = prompt_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            # 都是包含了 prompt 和input的 chosen 和 rejected
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.encoded_texts)

def fetch_data():
    # print_pkgs_version()
    # K:\KpcAIDev\KpcBuildLLM\data\download\instruction-data-with-preference.json
    url = (
    "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
    "/main/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json")    
    save_path = get_project_root() / "data" / "download"
    dwnfilepath = download_file(url,save_path)
    return dwnfilepath

def load_data(file_path):
    """ 
    {   
        'instruction': 'Evaluate the following phrase by transforming it into the spelling given.', 
        'input': 'freind --> friend', 
        'output': 'The spelling of the given phrase "freind" is incorrect, the correct spelling is "friend".', 
        'rejected': 'The spelling of the given phrase "freind" is flat out wrong, get it together, the correct spelling is "friend".', 
        'chosen': 'The spelling of the given phrase "freind" is incorrect, the correct spelling is "friend".'
        }
    """
    if not file_path.exists():
        file_path = fetch_data()

    with open(str(file_path), "r") as file:
        data = json.load(file)
    return data

def prompt_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    # common_response = f"### Response:\n{data['output']}"
    # desired_response = f"### Response:\n{data['chosen']}"
    # possible_response = f"### Response:\n{data['rejected']}"

    re = instruction_text + input_text
    return re

def div_data(data):
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)    # 10% for testing

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    return train_data, test_data, val_data

if "__main__" == __name__:
    # data = load_data(data_path)
    # pprint.pp(data[:2])
    # example_dataset = PreferDataset(data)
    pass
