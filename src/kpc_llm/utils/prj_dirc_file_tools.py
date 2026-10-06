""" 项目目录文件工具 """

from pathlib import Path
from kpc_llm.utils.logger import getlogger

lgr = getlogger()


# 获取项目根目录
def get_project_root() -> Path:
    """自动向上寻找包含特征文件的目录，将其认定为项目根目录"""
    current_path = Path(__file__).resolve()
    
    # 只要这三个文件/文件夹其中一个存在，就说明找到了根目录,（根目录的标志文件或文件夹名称）
    root_indicators = [".git", "requirements.txt", "pyproject.toml"]
    # 当前路径和其所有父目录
    all_parents = [current_path] + list(current_path.parents)
    # lgr.info(f"current_path: {current_path}")
    # lgr.info(f"all_parents: {all_parents}")
    # 当前文件路径和其所有父目录上查找是否存在根目录标志文件
    for parent in all_parents:
        if any((parent / indicator).exists() for indicator in root_indicators):
            return parent
            
    # 如果实在找不到，就兜底返回当前执行脚本的目录
    return current_path.parent

# 项目根目录创建文件夹
def create_dir_under_prj(dir_name: str) -> Path:
    # 无论你在哪里运行这段代码，ROOT_DIR 永远是项目最外层根目录
    PRJ_ROOT_DIR = get_project_root()
    # lgr.info(f"Project root localized at: {PRJ_ROOT_DIR}")
    """创建项目根目录下的文件夹"""
    dir_path = PRJ_ROOT_DIR / dir_name
    dir_path.mkdir(exist_ok=True)
    return dir_path

# 给文件名增加时间戳
def add_flnm_timestamp(fileName:str):
    import time
    if fileName is not None and fileName !="":
        fileName = fileName + time.strftime("_%Y%m%d_%H%M%S")
    return fileName
    


if __name__ == "__main__":
    returnPath = create_dir_under_prj("testkpc")
    lgr.info(f"Project at: {returnPath}")