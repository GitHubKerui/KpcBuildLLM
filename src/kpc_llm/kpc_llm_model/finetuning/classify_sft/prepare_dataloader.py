

from dataclasses import asdict

from pandas.core.frame import DataFrame
from pathlib import Path
import random
from sys import exception
from pandas.core.common import random_state
import pandas as pd
from kpc_llm.kpc_llm_model import KpcLLMModel
from kpc_llm.kpc_llm_model.finetuning.classify_sft.SpamDataset import SpamDataset
from kpc_llm.kpc_llm_model.pretraining.model_save_load import load_model_only
from kpc_llm.kpc_llm_model.train_cfg import GPT2_LARGE_CNF, GPT2_cl100k_base_CNF
from kpc_llm.utils import download_file, getlogger,unzip_file,get_project_root,move_rename_file,print_pkgs_version
from kpc_llm.utils.logger import log_fn_cost_time, log_fn_io
import tiktoken
from torch.utils.data import DataLoader

tokenizer = tiktoken.get_encoding("gpt2")


log =  getlogger()

# 打印包版本
# print_pkgs_version()
""" Stage 1 Dataset Preparation"""

def fetch_data():
    """ 1.0 下载文件,解压文件，文件移动重命名成tsv后缀"""
    print_pkgs_version()
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    save_path = get_project_root() / "data" / "download"
    dwnfilepath = download_file(url,save_path)
    unzip_dir = unzip_file(dwnfilepath)
    final_file_path = ""
    if unzip_dir:
       final_file_path = move_rename_file(unzip_dir / "SMSSpamCollection",
                        get_project_root() / "data" / "SMSSpamCollection.tsv")
    return final_file_path

def view_data(file_path):
    """ 查看数据 """
    df: DataFrame = pd.read_csv(file_path,sep="\t",header=None,names=["Label","Text"]) 
    # log.info(f"\n\n {df}")
    # label的分类统计数量value_counts
    # log.info(f"\n\n {df["Label"].value_counts()}")
    return df

def balance_df(df):
    """ 平衡数据 """
    num_spam = df[df["Label"] == "spam"].shape[0]
    # region log
    # iloc用integer 定位
    # loc 用名字定位。
    # log.info(f"\n\n num_spam : {num_spam}")
    # log.info(f"\n\n num_spam shape : {df[df["Label"] == "spam"].shape}")
    # log.info(f"\n\n spam[0:2] : {df[df["Label"] == "spam"].iloc[:2]}")
    # 从多的Ham中抽取 跟 spam一样多的
    # endregion
    ham_samples = df[df["Label"]=="ham"].sample(num_spam,random_state=825) 
    balanced_df = pd.concat([df[df["Label"] == "spam"],ham_samples])
    # 必须先整体打乱：concat 之后是「前 747 行全 spam、后 747 行全 ham」的有序排列，
    # 而 div_df 是纯按位置切 80/10/10，不打乱的话 val/test 会整段落进 ham 区，
    # 变成 100% 单一类别，验证集和测试集就完全失去评估意义了。frac 是无放回抽取比例
    balanced_df = balanced_df.sample(frac=1.0,random_state=825)
    # sample后index混乱，reset重置，这个是输出结果重置，不是直接重置
    balanced_df = balanced_df.reset_index(drop=True)
    # 替换 ham 0 spam 1 
    # label2id = {"spam": 0, "ham": 1}
    # balanced_df["Label"] = balanced_df["Label"].map(label2id)
    balanced_df["Label"] = balanced_df["Label"].replace({"ham":0,"spam":1})
    
    # log.info(f"\n\n {balanced_df.iloc[:2]}")
    return balanced_df

# @log_fn_cost_time    
# @log_fn_io
def div_df(df:DataFrame):
    """ 切数据 """
    train_tail = int(len(df)*0.7)
    val_tail = int(len(df)*0.85)

    train_df = df[:train_tail]
    val_df = df[train_tail:val_tail]
    test_df = df[val_tail:]

    # log.info(f"\n train_df: {train_df.size}")
    # log.info(f"\n val_df : {val_df.size}")
    # log.info(f"\n test_df : {test_df.size}")
    return train_df,val_df,test_df

""" Windows 上每个 worker 都是独立进程（spawn 启动），各自要加载一遍完整的 torch + CUDA DLL
（shm.dll / libtorch_cuda.dll，GB 量级）。num_workers 开大很容易把系统提交内存打满，
先后报过两种错误但根因同一个：
  1) OSError: [WinError 1455] 页面文件太小 —— worker 启动时映射 shm.dll 直接失败；
  2) OSError: [Errno 28] No space left on device —— 保存 checkpoint 时内存尖峰撑爆提交量，
     Windows 把「提交失败」误报成「磁盘满」（当时 K 盘实际还有 92.9GB 空闲）。
persistent_workers=True 时中途 Ctrl+C 不会回收 worker，跑几轮会攒下一堆孤儿 python.exe 撑爆内存，
排查前先 taskkill /F /IM python.exe。
本机 2~4 就够：数据集只有 1045 条、batch=8 共 130 步，瓶颈 100% 在 GPU 前向，不在数据加载。
调试期可传 0，让数据在主进程加载，最省内存且报错栈最短。 """
def prepara_all_loader(train_df,val_df,test_df,batch_size:int=8,num_worker:int=2):
        # 保存到 csv 文件
    train_df.to_csv("train.csv", index=False)
    val_df.to_csv("val.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    train_dataset = SpamDataset(csv_file="train.csv",tokenizer=tokenizer,max_length=None)
    val_dataset = SpamDataset(csv_file="val.csv",tokenizer=tokenizer,max_length=None)
    test_dataset = SpamDataset(csv_file="test.csv",tokenizer=tokenizer,max_length=None)

    train_loader = DataLoader(train_dataset,batch_size,True,num_workers=num_worker,drop_last=True,persistent_workers=True)
    val_loader = DataLoader(val_dataset,batch_size,num_workers=num_worker,drop_last=False,persistent_workers=True)
    test_loader = DataLoader(test_dataset,batch_size,num_workers=num_worker,drop_last=False,persistent_workers=True)
    # log.info(f"\n train_loader: {len(train_loader)}")
    # log.info(f"\n val_loader  : {len(val_loader)}")
    # log.info(f"\n test_loader : {len(test_loader)}")
    return train_loader,val_loader,test_loader

def prepare_dataloader():
    # file_path = stage1_fetch_data()
    file_path = get_project_root() / "data" / "SMSSpamCollection.tsv"
    df = view_data(file_path)
    df = balance_df(df)
    train_df,val_df,test_df = div_df(df)
    train_loader,val_loader,test_loader = prepara_all_loader(train_df,val_df,test_df)
    return train_loader,val_loader,test_loader


if "__main__" == __name__:
    # print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
    # # file_path = stage1_fetch_data()
    # file_path = get_project_root() / "data" / "SMSSpamCollection.tsv"
    # df = view_data(file_path)
    # df = balance_df(df)
    # train_df,val_df,test_df = div_df(df)
    # train_loader,val_loader,test_loader = prepara_all_loader(train_df,val_df,test_df)

    pass
