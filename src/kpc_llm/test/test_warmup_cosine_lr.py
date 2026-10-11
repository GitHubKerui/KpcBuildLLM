"""学习率调度（warmup + cosine）的测试。"""

import pytest

from kpc_llm.kpc_llm_model.pretraining.eff_calcu_train_loop import (
    _build_warmup_cosine_lambda,
)


def test_warmup_is_linear_and_nonzero_at_first_step():
    f = _build_warmup_cosine_lambda(total_steps=1000, warmup_ratio=0.1, min_ratio=0.1)
    warmup_steps = 100
    # 第一次 step（入参 0）拿到的应是 1/warmup_steps，而不是 0（否则首步不更新）
    assert f(0) == pytest.approx(1 / warmup_steps)
    assert f(49) == pytest.approx(50 / warmup_steps)
    # warmup 结束处到达峰值 1.0
    assert f(99) == pytest.approx(1.0)


def test_decay_ends_at_min_ratio():
    f = _build_warmup_cosine_lambda(total_steps=1000, warmup_ratio=0.1, min_ratio=0.1)
    assert f(999) == pytest.approx(0.1, abs=1e-6)


def test_warmup_up_then_cosine_down():
    f = _build_warmup_cosine_lambda(total_steps=1000, warmup_ratio=0.1, min_ratio=0.1)
    vals = [f(i) for i in range(1000)]
    # warmup 段单调不降
    assert all(vals[i] <= vals[i + 1] + 1e-12 for i in range(99))
    # 衰减段单调不增
    assert all(vals[i] >= vals[i + 1] - 1e-12 for i in range(100, 999))


def test_never_exceeds_one():
    f = _build_warmup_cosine_lambda(total_steps=500, warmup_ratio=0.02, min_ratio=0.1)
    assert max(f(i) for i in range(600)) <= 1.0 + 1e-9


def test_beyond_total_steps_stays_at_floor():
    f = _build_warmup_cosine_lambda(total_steps=100, warmup_ratio=0.1, min_ratio=0.1)
    # 超出计划步数（早停导致的实际步数可能更多/更少）也不该反弹
    assert f(500) == pytest.approx(0.1, abs=1e-6)


def test_accelerate_scheduler_and_clip_integration():
    """真实 Accelerator 下：调度器能随优化器步进、梯度裁剪不报错，warmup 段 lr 抬升。"""
    import torch
    from accelerate import Accelerator

    model = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _build_warmup_cosine_lambda(total_steps=100, warmup_ratio=0.1, min_ratio=0.1)
    )
    accelerator = Accelerator(cpu=True)
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)

    lrs = []
    for _ in range(5):
        optimizer.zero_grad()
        accelerator.backward(model(torch.zeros(1, 4)).sum())
        accelerator.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    assert lrs == sorted(lrs), "warmup 段 lr 应单调抬升"
    assert lrs[0] > 0, "第一次 optimizer.step 的学习率不应为 0"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
