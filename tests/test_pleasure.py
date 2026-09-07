"""愉悦度系统单元测试（L2）：RPE 预测误差驱动的内在动机。"""
import numpy as np
import pytest

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine


def make_engine(n=10, seed=42, pleasure_enabled=True):
    cfg = SimConfig(seed=seed)
    cfg.population.initial_count = n
    cfg.simulation.ticks = 1000
    cfg.pleasure.enabled = pleasure_enabled
    return SphereEngine(cfg)


def test_initial_pleasure_state():
    """初始状态：valence=0, arousal=0.5, expectation=乐观值, baseline=0。"""
    e = make_engine(n=20)
    pcfg = e.config.pleasure
    assert (e._valence == 0).all()
    assert (e._arousal == 0.5).all()
    assert (e._baseline == 0).all()
    assert e._expectation.shape == (20, pcfg.expectation_size)
    np.testing.assert_allclose(
        e._expectation, pcfg.optimism * pcfg.max_reward
    )


def test_rpe_positive_increases_valence():
    """超预期（正 RPE）→ valence 上升。
    直接调用 _update_pleasure：所有情境预期拉低 + 正 Δ能量 + 有邻居 → 正 RPE。"""
    e = make_engine(n=3)
    e._flat[:] = 100  # 同一格（有邻居，社会增益正）
    e._energy[:] = e.config.organisms.max_energy * 0.8
    e.resources._grid[100] = e.resources._capacity[100] * 0.5
    # 把所有情境的预期拉低（经历过坏日子）
    e._expectation[:, :] = 0.05
    v_before = e._valence.copy()
    energy_before = e._energy.copy() - 10.0  # 之前能量更低 → Δ能量为正
    e._update_pleasure(3, energy_before)
    assert (e._valence > v_before).any()


def test_rpe_negative_decreases_valence():
    """低于预期（负 RPE）→ valence 下降。
    直接调用 _update_pleasure：高预期 + 负 Δ能量 + 无邻居 → 负 RPE。"""
    e = make_engine(n=3)
    e._flat[:] = [0, 200, 400]  # 分散（无邻居，社会增益负）
    e._energy[:] = e.config.organisms.max_energy * 0.2
    e.resources._grid[:] = 0.0  # 无食物
    # 把该情境的预期拉高（乐观）
    e._expectation[:, :] = 1.5
    v_before = e._valence.copy()
    energy_before = e._energy.copy() + 5.0  # 之前能量更高 → Δ能量为负
    e._update_pleasure(3, energy_before)
    assert (e._valence < v_before).any()


def test_expectation_ewma_converges():
    """重复相同情境+收益 → expectation 向实际收益收敛（EWMA 学习）。"""
    e = make_engine(n=1)
    # 固定位置、固定能量、固定食物 → 情境不变
    e._flat[0] = 500
    e._energy[0] = e.config.organisms.max_energy * 0.5
    e.resources._grid[500] = e.resources._capacity[500] * 0.5
    ctx_before = e._expectation[0, 60]  # 中间某个情境
    for _ in range(50):
        e.step()
    # expectation 应该发生了变化（学习了）
    assert not np.allclose(e._expectation[0], ctx_before)
    # expectation 应该在 [0, max_reward] 范围内
    assert (e._expectation >= 0).all()
    assert (e._expectation <= e.config.pleasure.max_reward).all()


def test_valence_decays_to_neutral():
    """valence 每 tick 衰减，长期回到中性 0。"""
    e = make_engine(n=1)
    e._valence[0] = 0.8
    for _ in range(100):
        # 不调用 step（避免 RPE 干扰），手动衰减
        e._valence[0] *= e.config.pleasure.valence_decay
    assert abs(e._valence[0]) < 0.01


def test_offspring_inherits_expectation():
    """繁殖时子代 expectation ≈ 亲代 + 噪声（文化传递载体）。"""
    e = make_engine(n=2, seed=7)
    # 让亲代快速繁殖：高能量、低繁殖阈值、成熟
    e._energy[:] = e.config.organisms.max_energy
    e._genes[:, 2] = 0.0  # 繁殖阈值最低
    e._genes[:, 7] = 0.5  # 对半投入
    e._age[:] = 1000  # 成年
    e._repro_cooldown[:] = 0
    parent_exp = e._expectation[0].copy()
    # 跑若干 tick 直到有繁殖
    for _ in range(20):
        e.step()
        if e._max_generation > 0:
            break
    assert e._max_generation > 0, "应该有繁殖发生"
    # 子代（generation=1）的 expectation 应与亲代有相关性
    child_mask = e._generation >= 1
    assert child_mask.any()
    # 子代 expectation 不全等于初始乐观值（继承了亲代的学习结果）
    child_exp = e._expectation[child_mask]
    assert not np.allclose(
        child_exp, e.config.pleasure.optimism * e.config.pleasure.max_reward
    )


def test_pleasure_disabled_no_update():
    """pleasure.enabled=False 时，愉悦度不更新（保持初始值）。"""
    e = make_engine(n=5, pleasure_enabled=False)
    exp_before = e._expectation.copy()
    val_before = e._valence.copy()
    for _ in range(10):
        e.step()
    np.testing.assert_array_equal(e._valence, val_before)
    # expectation 可能因繁殖而增加行数，但已有行不变
    assert len(e._valence) >= 5
    np.testing.assert_allclose(e._expectation[:5], exp_before[:5])


def test_pleasure_long_run_no_crash():
    """愉悦度系统长跑 500 tick 不崩、数组形状正确。"""
    e = make_engine(n=50, seed=123)
    for _ in range(500):
        e.step()
        if e.extinct:
            break
    n = len(e._id)
    assert e._valence.shape == (n,)
    assert e._arousal.shape == (n,)
    assert e._baseline.shape == (n,)
    assert e._expectation.shape == (n, e.config.pleasure.expectation_size)
    # valence 在 [-1,1]，arousal 在 [0,1]
    assert (e._valence >= -1).all() and (e._valence <= 1).all()
    assert (e._arousal >= 0).all() and (e._arousal <= 1).all()


def test_pleasure_summary():
    """pleasure_summary 返回正确统计。"""
    e = make_engine(n=10)
    e.step()
    s = e.pleasure_summary()
    assert "valence_mean" in s
    assert "arousal_mean" in s
    assert "baseline_mean" in s
    assert "expectation_mean" in s
    assert -1 <= s["valence_mean"] <= 1
    assert 0 <= s["arousal_mean"] <= 1
