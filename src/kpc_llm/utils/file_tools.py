""" 项目目录文件工具 """

from pathlib import Path
from venv import logger

from sympy import false
from kpc_llm.utils.logger import getlogger
import zipfile
import urllib.request
import os

log = getlogger()


def move_rename_file(origin_path:Path,dest_path:Path):
    """ 移动文件并改名 ,返回新的文件地址"""
    if not origin_path:
        raise ValueError(f"origin_path 错误值 {origin_path}")
    else:
        if not dest_path.exists:
            os.rename(origin_path,dest_path)
            log.info(f"移动文件 : {origin_path} ")
            log.info(f"并改名到 : {dest_path} 成功！")
        else:
            log.info(f"无法移动文件 : {origin_path} ")
            log.info(f"并改名到 : {dest_path} ！因为文件已经存在！")
        return Path(dest_path)

def download_file(url=None, save_path:Path=Path("download"), file_name=None,is_cover:bool=False):
    try:
        """下载指定 URL 的文件并保存到本地。
        Args:
            url: 文件下载地址；为空或 None 时抛 ValueError。
            save_path: 保存目录（Path 或 str 均可，内部统一转成 Path）。
                若 file_name 为空，则文件名取自 URL 最后一段，最终路径为 save_path / 文件名。
            file_name: 可选，显式指定保存的文件名；为空时从 URL 推断。
            is_cover:是否覆盖
        Returns:
            实际保存的文件路径（Path）。
        """
        # 测试用参数 后面可以注释
        # url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
        # save_path = get_project_root() /"data"/save_path
        
        # 如果url不等于空字符串 或者none
        if not url :
            raise ValueError("URL 不能为空！")
        else:
            if not file_name :
                # 如果文件名空，则用下载文件名
                file_name = url.split("/")[-1]
                log.info(f"下载地址 : {url}")
                log.info(f"准备保存文件 : {file_name}")
                save_path = save_path / file_name
                # 如果 文件存在则 不覆盖
                if save_path.exists() and not is_cover :
                    log.info(f"提示 ：文件 {save_path} 已经存在！不覆盖文件！")
                    log.info("退出下载！")
                    return Path(save_path)

                # 如果覆盖，或者文件不存在保存，都需要下载
                save_path.parent.mkdir(parents=True,exist_ok=True)
                with urllib.request.urlopen(url) as response:
                    with open(save_path,"wb") as file:
                        file.write(response.read())
                        # 如果覆盖 则返回覆盖成功
                        if is_cover :
                            log.info(f"下载并覆盖文件 : {save_path} 成功")
                        else:
                            log.info(f"下载文件 : {save_path} 成功")
                        return Path(save_path)
    except Exception as e:
        log.exception(f"下载文件错误: {e}")

# 1.2 解压文件
def unzip_file(zip_file_path):
    """ 解压文件 默认解压到文件的同目录的文件名的新目录里
    Args:
        zip_file_path: 解压文件的保存地址。
    Returns:
        解压成功的目录
    """
    # zip_file_path.parent 文件目录
    # zip_file_path.stem 去掉后缀的文件名
    try:
        if zip_file_path :
                upzip_dir=   zip_file_path.parent / zip_file_path.stem
                with zipfile.ZipFile(zip_file_path,"r") as zip_ref:
                    zip_ref.extractall(upzip_dir)
                log.info(f"解压到文件夹 : {upzip_dir} 成功！")
                return Path(upzip_dir)
    except Exception as e:
        log.exception(f"解压错误 {e}")

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
    dir_path.mkdir(parents=True,exist_ok=True)
    return dir_path

# 给文件名增加时间戳
def add_flnm_timestamp(fileName:str):
    import time
    if fileName is not None and fileName !="":
        fileName = fileName + time.strftime("_%Y%m%d_%H%M%S")
    return fileName
    
# 从url下载资源保存到项目根目录下的指定文件夹
def getResourceFromUrl(fileUrl: str, savePrjSubDir: str):
    """
    从 URL 下载文件到本地目录，按下载名称保存 step2
    Args:
        fileUrl: 文件下载地址
        saveSubDir: 保存的子目录，相对于项目根目录名称
    Returns:
        下载文件的完整路径
    """

    # 下载url最后 要有文件名才合法
    try:

        savefileName = fileUrl.split("/")[-1]
        fullDir: Path = create_dir_under_prj(savePrjSubDir)
        logger.info(f" url : {fileUrl}")
        logger.info(f" savefileName : {savefileName}")
        logger.info(f" fullDir : {fullDir}")
        if savePrjSubDir:
            # 指定根目录文件夹时候的保存方法
            fileLocalSavePath = fullDir / savefileName
        else:
            # 未指定则原名称存到根目录的temp文件夹下
            fileLocalSavePath =  get_project_root()/"temp"/ savefileName

        if not os.path.exists(fileLocalSavePath):
            urllib.request.urlretrieve(fileUrl, filename=str(fileLocalSavePath))
            log.info(f"已下载到: {fileLocalSavePath}")
        else:
            log.info(f"文件已经存在: {fileLocalSavePath}")
    except Exception as e:
        logger.error(f"fileUrl : {fileUrl} ; savePrjSubDir :{savePrjSubDir} ;{e}出错！",exc_info=True)

    return fileLocalSavePath

# if __name__ == "__main__":
#     returnPath = create_dir_under_prj("testkpc")
#     lgr.info(f"Project at: {returnPath}")