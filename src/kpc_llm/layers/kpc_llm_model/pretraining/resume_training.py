"""断点续训：存档目录解析 + 恢复决策（reasoning）+ 状态加载。

职责边界：
    只负责"从哪个目录加载"的判断与加载动作，以及本次任务的存档目录定位；
    不负责训练过程中的存档写入（那是 train_model_simple 按 state_dir 落盘的事）。

目录结构：<accelerate 总目录>/<子目录>/epoch_XXX
    - accelerate 总目录：configs/project_config.toml 的 [training].accelerate_save_dir
      （只配总目录，其下不同子目录可能存放不同任务/模型的存档，互不覆盖）
    - 子目录：由本文件的 SAVE_SUBDIR / LOAD_SUBDIR 决定
"""

from pathlib import Path

from kpc_llm.utils.config_loader import ConfigLoader
from kpc_llm.utils.logger import getlogger
from kpc_llm.utils.prj_dirc_file_tools import create_dir_under_prj, get_project_root

logger = getlogger()

# ===========================================================================
# 续训开关面板（本模块是断点续训的唯一配置入口）
# ===========================================================================
# 续训总开关：False = 每次都从头训练（下面的两个子目录被忽略）
RESUME_ENABLE = True
# 默认子目录名：SAVE_SUBDIR 未显式指定时使用，因此"首次训练会存到默认子目录"
DEFAULT_SUBDIR = "default"
# 本次任务保存 checkpoints 的子目录（accelerate 总目录之下）；留空则用 DEFAULT_SUBDIR
SAVE_SUBDIR = ""
# 从哪个子目录加载模型续训；留空则与 SAVE_SUBDIR 相同（即从本任务上次的存档续训）
LOAD_SUBDIR = ""


def looks_like_state_dir(path: Path) -> bool:
    """目录本身是否是 accelerate 存档（自身直接含模型权重文件）。"""
    return any(path.glob("model*.safetensors")) or any(path.glob("pytorch_model*.bin"))


def latest_state_dir(subdir: Path) -> Path | None:
    """在给定子目录里找一份可用的 accelerate 存档。

    优先用目录本身（自身直接含权重文件）；否则取序号最大的 epoch_* 子目录。
    目录不存在、或里面没有任何可用存档时返回 None（调用方据此从头训练）。
    """
    if not subdir.is_dir():
        return None
    # 情况 1：本身就是一份存档
    if looks_like_state_dir(subdir):
        return subdir
    # 情况 2：子目录下按 epoch_NNN 分列，挑序号最大的
    epochs = [p for p in subdir.glob("epoch_*") if p.is_dir()]
    if not epochs:
        return None

    def _seq(p: Path) -> int:
        tail = p.name.split("_")[-1]
        return int(tail) if tail.isdigit() else -1

    return max(epochs, key=_seq)


class ResumeManager:
    """封装断点续训的"目录定位 + 恢复决策 + 加载"。

    用法::

        rm = ResumeManager()
        state_dir = rm.save_dir()          # 本次任务存档目录（已创建）
        rm.try_resume(accelerator)         # 有可用存档则加载，否则跳过（从头训练）

    Args:
        resume_enable: 是否续训；False 则 try_resume 直接跳过。
        save_subdir:   本次任务保存的子目录；留空回退 default_subdir。
        load_subdir:   加载来源子目录；留空回退到实际使用的保存子目录。
        default_subdir: 默认子目录名。
        accel_root:    accelerate 总目录（相对项目根）；None 则读配置文件。
    """

    def __init__(
        self,
        resume_enable: bool = RESUME_ENABLE,
        save_subdir: str = SAVE_SUBDIR,
        load_subdir: str = LOAD_SUBDIR,
        default_subdir: str = DEFAULT_SUBDIR,
        accel_root: str | None = None,
    ) -> None:
        self.resume_enable = resume_enable
        self.default_subdir = default_subdir
        self.accel_root = (
            accel_root if accel_root is not None else ConfigLoader().accelerate_save_dir
        )
        # 保存子目录：留空 -> 默认子目录；加载子目录：留空 -> 与保存子目录相同
        self.save_subdir = save_subdir or default_subdir
        self.load_subdir = load_subdir or self.save_subdir

    # ------------------------------------------------------------------
    # 目录定位
    # ------------------------------------------------------------------
    def save_dir(self) -> Path:
        """本次任务保存 checkpoints 的目录（锚定项目根并创建）。"""
        return create_dir_under_prj(f"{self.accel_root}/{self.save_subdir}")

    def load_root(self) -> Path:
        """续训来源子目录（可能不存在）。"""
        return get_project_root() / self.accel_root / self.load_subdir

    # ------------------------------------------------------------------
    # 恢复决策 + 加载
    # ------------------------------------------------------------------
    def resolve_load_dir(self) -> Path | None:
        """判断：在续训子目录里找一份可用存档；找不到返回 None（从头训练）。"""
        return latest_state_dir(self.load_root())

    def try_resume(self, accelerator) -> bool:
        """尝试从存档恢复训练状态；返回是否真的恢复了。

        任一步不满足（开关关闭 / 无可用存档 / 加载报错）都返回 False 并打印警告，
        调用方据此从头训练，避免把训练直接崩在恢复阶段。
        """
        if not self.resume_enable:
            logger.info("RESUME_ENABLE=False，从头开始训练")
            return False

        load_dir = self.resolve_load_dir()
        if load_dir is None:
            logger.warning(
                f"续训子目录 {self.load_subdir!r} 下没有可用存档"
                f"（{self.load_root()}）；将从头开始训练"
            )
            return False

        try:
            accelerator.load_state(str(load_dir))
        except Exception as e:
            logger.warning(f"加载存档失败（{load_dir}）：{e}；将从头开始训练")
            return False

        logger.info(
            f"已从 {load_dir} 恢复训练状态"
            f"（模型 + 优化器 + 随机数种子 + topk 记录 + 早停计数）"
        )
        logger.info(
            "注意：epoch/global_step 计数从 0 重新开始（不在存档里）；"
            "因 shuffle=False，数据顺序与原训练一致"
        )
        return True
