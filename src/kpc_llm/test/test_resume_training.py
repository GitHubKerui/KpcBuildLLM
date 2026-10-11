"""断点续训模块的闭环测试。

覆盖：
    1. 纯函数 latest_state_dir / looks_like_state_dir 的选择逻辑；
    2. ResumeManager 的子目录回退、续训开关、配置来源；
    3. 真实 Accelerator 的"存 -> 判断 -> 载"往返：save_state 落盘，
       换一组初始权重后 try_resume 应把权重还原成存档里的值。

牵涉真实存档的用例统一写到项目内 checkpoints/_test_resume_training 下，用例结束清理。
"""

import shutil
from pathlib import Path

import pytest
import torch
from accelerate import Accelerator

from kpc_llm.kpc_llm_model.pretraining.resume_training import (
    ResumeManager,
    latest_state_dir,
    looks_like_state_dir,
)
from kpc_llm.utils.config_loader import ConfigLoader
from kpc_llm.utils.logger import getlogger
from kpc_llm.utils.prj_dirc_file_tools import get_project_root

logger = getlogger()

# 真实存档用例的落盘根（相对项目根），用后即删
TEST_ROOT_REL = "checkpoints/_test_resume_training"


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _tiny_model(seed: int) -> torch.nn.Module:
    """构造一个确定性的小模型；不同 seed -> 不同初始权重。"""
    torch.manual_seed(seed)
    return torch.nn.Linear(4, 3)


def _prepared(seed: int = 0):
    """在 CPU 上准备一份 (accelerator, model, optimizer)。

    必须走 prepare：load_state 只恢复已注册的对象，未 prepare 时会"无对象可载"
    而静默返回成功，测不出真实的加载失败。
    """
    model = _tiny_model(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    accelerator = Accelerator(cpu=True)
    model, optimizer = accelerator.prepare(model, optimizer)
    return accelerator, model, optimizer


@pytest.fixture
def test_root():
    """项目内临时存档根：进入前清一次，退出再清一次。"""
    root = get_project_root() / TEST_ROOT_REL
    shutil.rmtree(root, ignore_errors=True)
    yield root
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# 1. 纯函数
# ---------------------------------------------------------------------------
def test_latest_state_dir_picks_max_epoch(tmp_path: Path):
    for name in ("epoch_000", "epoch_002", "epoch_010"):
        (tmp_path / name).mkdir()
    picked = latest_state_dir(tmp_path)
    assert picked is not None and picked.name == "epoch_010"


def test_latest_state_dir_prefers_self_when_it_is_a_state(tmp_path: Path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    assert latest_state_dir(tmp_path) == tmp_path
    assert looks_like_state_dir(tmp_path) is True


def test_latest_state_dir_none_cases(tmp_path: Path):
    # 存在但空
    assert latest_state_dir(tmp_path) is None
    # 不存在
    assert latest_state_dir(tmp_path / "not_exist") is None


# ---------------------------------------------------------------------------
# 2. ResumeManager 的配置逻辑（不需要 Accelerator）
# ---------------------------------------------------------------------------
def test_subdir_fallbacks():
    # 都留空 -> 用默认子目录
    rm = ResumeManager(save_subdir="", load_subdir="", accel_root="x")
    assert rm.save_subdir == "default"
    assert rm.load_subdir == "default"

    # 只配保存 -> 加载回退到保存子目录（从本任务上次的存档续训）
    rm2 = ResumeManager(save_subdir="taskA", load_subdir="", accel_root="x")
    assert rm2.save_subdir == "taskA"
    assert rm2.load_subdir == "taskA"

    # 显式指定加载子目录
    rm3 = ResumeManager(save_subdir="taskA", load_subdir="taskB", accel_root="x")
    assert rm3.load_subdir == "taskB"


def test_default_accel_root_comes_from_config():
    rm = ResumeManager()
    assert rm.accel_root  # 非空
    assert rm.accel_root == ConfigLoader().accelerate_save_dir


def test_resume_disabled_returns_false():
    rm = ResumeManager(resume_enable=False, accel_root=TEST_ROOT_REL, save_subdir="x")
    assert rm.try_resume(Accelerator(cpu=True)) is False


def test_missing_state_returns_false(test_root: Path):
    rm = ResumeManager(accel_root=TEST_ROOT_REL, save_subdir="nope")
    assert rm.resolve_load_dir() is None
    assert rm.try_resume(Accelerator(cpu=True)) is False


# ---------------------------------------------------------------------------
# 3. 真实往返：save_state -> try_resume -> 权重还原
# ---------------------------------------------------------------------------
def test_save_then_resume_roundtrip(test_root: Path):
    sub = "taskA"

    # --- 存：训一步后存档到 <root>/taskA/epoch_000 ---
    accelerator, model, optimizer = _prepared(0)
    model(torch.zeros(1, 4)).sum().backward()   # 制造与初始权重不同的参数
    optimizer.step()
    accelerator.save_state(str(test_root / sub / "epoch_000"))

    saved_weight = model.state_dict()["weight"].detach().clone()
    saved_bias = model.state_dict()["bias"].detach().clone()

    # --- 载：换一组不同的初始权重，try_resume 应还原成 saved ---
    accelerator2, fresh, _ = _prepared(1)
    assert not torch.equal(fresh.state_dict()["weight"], saved_weight), \
        "前置条件：两次初始权重应不同"

    rm = ResumeManager(accel_root=TEST_ROOT_REL, save_subdir=sub)
    assert rm.resolve_load_dir() == test_root / sub / "epoch_000", "应能定位到唯一存档"

    assert rm.try_resume(accelerator2) is True, "存在合法存档时应恢复成功"
    assert torch.equal(fresh.state_dict()["weight"], saved_weight), "权重未还原成存档值"
    assert torch.equal(fresh.state_dict()["bias"], saved_bias), "bias 未还原成存档值"
    logger.info("闭环通过：save_state -> try_resume 后权重与存档完全一致")


def test_corrupt_state_returns_false(test_root: Path):
    """存档存在但内容损坏时，try_resume 应吞掉异常并返回 False（从头训练）。"""
    bogus = test_root / "taskC" / "epoch_000"
    bogus.mkdir(parents=True)
    (bogus / "model.safetensors").write_text("this is not a valid safetensors file")

    rm = ResumeManager(accel_root=TEST_ROOT_REL, save_subdir="taskC")
    assert rm.resolve_load_dir() == bogus, "损坏存档仍应被定位到"

    # 必须 prepare 出可加载的对象，load_state 才会真的去读那个损坏文件而报错
    accelerator, _, _ = _prepared(0)
    assert rm.try_resume(accelerator) is False, "加载失败应安全退回从头训练"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
