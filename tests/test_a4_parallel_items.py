"""A4 并行项回归测试：C3（neutral 只冻结 g14/g15）与 B1（R2 声誉权重可开关）。

任务出处：`_share/待办与交接.md` C3 / B1；设计依据：
docs/决策与评审/评估-EVAL-可行性分析-g15高位payoff-20260911.md §三（候选 R2）。
"""
import numpy as np

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(**kw):
    cfg = SimConfig(seed=7)
    cfg.population.initial_count = 40
    cfg.population.max_count = 5000
    cfg.simulation.ticks = 300
    for k, v in kw.items():
        setattr(cfg, k, v)
    return SphereEngine(cfg)


def _run_and_state(ticks=120, **kw):
    """建引擎并跑完，返回 (P, genes)。

    注意：D2 感知噪声用【进程级全局 np.random】（已知 F-D2 缺陷），
    其 seed 在 __init__ 时重置。因此必须"建一个就跑完"再建下一个，
    两个引擎才不会互相污染全局 RNG 流。
    """
    e = _engine(**kw)
    for _ in range(ticks):
        e.step()
        if e.extinct:
            break
    P = len(e._id)
    return P, (e._genes[:P].copy() if P else np.zeros((0, 0)))


# ---------------- C3：neutral 零模型只冻结 g14/g15 ----------------

def test_neutral_freezes_only_g14_g15():
    """neutral_genes=True 时，只有 g14/g15 被钉在 0.5，其余基因照常演化。"""
    P, genes = _run_and_state(neutral_genes=True)
    assert P > 0, "只冻结 g14/g15 不应灭绝（旧全冻结 N=2~5 会灭）"
    assert np.allclose(genes[:, 14], 0.5), "g14 应被冻结为 0.5"
    assert np.allclose(genes[:, 15], 0.5), "g15 应被冻结为 0.5"
    others = np.delete(genes, [14, 15], axis=1)
    assert not np.allclose(others, 0.5), "g0–g13 不应被冻结为 0.5"


def test_neutral_off_genes_evolve():
    """neutral_genes=False 时 g14/g15 不被钉死。"""
    P, genes = _run_and_state(neutral_genes=False)
    assert P > 0
    assert not (np.allclose(genes[:, 14], 0.5) and np.allclose(genes[:, 15], 0.5))


# ---------------- B1：R2 声誉权重 ----------------

def test_reputation_weight_default_zero_matches_baseline():
    """reputation_weight 默认 0 → 与显式 0.0 逐位一致（开关关闭不改行为）。"""
    pa, ga = _run_and_state(info_structure=InfoStructureConfig(
        enabled=True, perception_radius=4, perception_noise=0.05, softmax_tau=0.15))
    pb, gb = _run_and_state(info_structure=InfoStructureConfig(
        enabled=True, perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
        reputation_weight=0.0))
    assert pa == pb
    if pa:
        assert np.array_equal(ga, gb)


def test_reputation_weight_is_wired():
    """reputation_weight>0 时移动得分信号项被 trust 放大 → 轨迹与关闭时不同。"""
    pa, ga = _run_and_state(info_structure=InfoStructureConfig(
        enabled=True, perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
        reputation_weight=0.0))
    pb, gb = _run_and_state(info_structure=InfoStructureConfig(
        enabled=True, perception_radius=4, perception_noise=0.05, softmax_tau=0.15,
        reputation_weight=1.5))
    same = (pa == pb) and (pa == 0 or np.array_equal(ga, gb))
    assert not same, "开启声誉权重后轨迹应发生变化（证明参数已接线）"


def test_reputation_weight_default_is_off():
    """默认配置必须是关闭态（R6：只开发不判读，默认不得改变既有行为）。"""
    assert InfoStructureConfig().reputation_weight == 0.0
