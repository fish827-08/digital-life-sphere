"""R77：`max_gen` 口径化 —— 当刻最深 vs 历史高水位，两个口径必须可分。

背景（D-24 实测）：同一个列名 `max_gen` 在不同脚本里指不同东西——
`observatory.statistics.d2_metrics` 用「当刻最深」（只看存活），
`experiments/a4_verify_capacity.py` 用「历史高水位」（`_max_generation`）⇒
跨脚本比较必然出错。本测试锁死两者语义。
"""
from __future__ import annotations

from simulation.config import SimConfig
from simulation.sphere_engine import SphereEngine
from observatory.statistics import (
    d2_metrics, max_generation_current, max_generation_highwater,
)


def _engine(seed: int = 42, max_count: int = 200) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    return SphereEngine(cfg)


def _run(e: SphereEngine, n: int) -> None:
    for _ in range(n):
        if e.extinct:
            break
        e.step()


def test_current_never_exceeds_highwater():
    e = _engine()
    for _ in range(300):
        if e.extinct:
            break
        e.step()
        assert max_generation_current(e) <= max_generation_highwater(e)


def test_highwater_never_decreases():
    e = _engine()
    prev = 0
    for _ in range(300):
        if e.extinct:
            break
        e.step()
        hw = max_generation_highwater(e)
        assert hw >= prev, "历史高水位不得回落"
        prev = hw


def test_current_is_minus_one_when_extinct():
    e = _engine()
    _run(e, 200)
    e._id = e._id[:0]          # 模拟灭绝（空种群）
    assert max_generation_current(e) == -1


def test_two_definitions_diverge_by_construction():
    """两个口径在语义上**必须可区分**：把高水位手动抬高后，当刻口径不受影响。

    （白盒构造：直接改 `_max_generation` 只为分离语义，不模拟真实演化。）
    """
    e = _engine()
    _run(e, 120)
    cur = max_generation_current(e)
    assert cur >= 0, "120 tick 后应已有存活个体"
    e._max_generation = cur + 999          # 高水位抬高
    assert max_generation_highwater(e) == cur + 999
    assert max_generation_current(e) == cur, "当刻口径不得被高水位污染"


def test_d2_metrics_uses_current_semantics():
    """`D2Metrics.max_gen` 官方口径 = 当刻最深（不能是高水位）。"""
    e = _engine()
    _run(e, 150)
    m = d2_metrics(e, tick=e._tick)
    assert m.max_gen == max_generation_current(e)
    e._max_generation = m.max_gen + 500
    assert d2_metrics(e, tick=e._tick).max_gen == m.max_gen, (
        "d2_metrics 必须用当刻口径（R77）"
    )
