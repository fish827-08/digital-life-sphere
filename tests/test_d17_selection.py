"""D-17：⑤ 个体层选择梯度（R42 / V-7 §2.2 口径）。

验收条件（V-7 给 [本地开发] 的硬约束）：
- RS 按 **_id 键控**（死亡压缩重排槽位 ⇒ 禁槽位键控）
- **必须含死亡个体**
- **饱和窗单独诊断**（两层不得合并）
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from simulation.config import InfoStructureConfig, SimConfig  # noqa: E402
from simulation.sphere_engine import SphereEngine  # noqa: E402
from observatory.statistics import selection_gradient  # noqa: E402


def _engine(measure: bool = True, max_count: int | None = None,
            lifespan_mult: float | None = None, **kw) -> SphereEngine:
    cfg = SimConfig(seed=42, **kw)
    cfg.simulation.use_sim_core = False
    if max_count is not None:
        cfg.population.max_count = max_count
    if lifespan_mult is not None:
        cfg.organisms.lifespan_mult = lifespan_mult
    d2 = InfoStructureConfig(enabled=True, learning_rate=0.05)
    d2.measure_signal_response = measure
    cfg.info_structure = d2
    return SphereEngine(cfg)


def _run(e: SphereEngine, ticks: int) -> None:
    for _ in range(ticks):
        if e.extinct:
            break
        e.step()


def test_id_ledgers_keyed_and_consistent():
    """账本按 _id 键控：长度=_next_id；子代总数=出生总数（键控正确的机械校验）。"""
    e = _engine()
    _run(e, 300)
    assert len(e._rs_children) == e._next_id
    assert int(e._rs_children.sum()) == e.total_born
    # 每个出生者恰有一个亲代记账：_parent 非负者的亲代 id 都在账本范围内
    parents = e._parent[e._parent >= 0]
    assert int(e._rs_children[parents].sum()) == e.total_born


def test_includes_dead_individuals():
    """R42：必须含死亡个体——已观测 id 集合必须大于"当前存活"集合。"""
    e = _engine()
    _run(e, 400)
    observed = np.flatnonzero(e._rs_observed)
    alive = set(e._id.tolist())
    dead_observed = [int(i) for i in observed if int(i) not in alive]
    assert e.total_died > 0, "400 tick 内应发生过死亡"
    assert len(dead_observed) > 0, "已死亡个体必须保留在 ⑤ 观测账本里"
    # 且死亡个体的 fitness 记录可取（children − cc0 ≥ 0）
    d = np.array(dead_observed, dtype=np.int64)
    fit = e._rs_children[d].astype(np.int64) - e._rs_cc0[d].astype(np.int64)
    assert (fit >= 0).all()


def test_saturated_window_is_separate_cohort():
    """max_count=initial ⇒ 从头饱和：非饱和 cohort 应为空，饱和诊断 cohort 有样本。"""
    e = _engine(measure=True, max_count=200)
    _run(e, 200)
    res = selection_gradient(e)
    assert res["non_sat"]["n"] == 0
    assert res["sat_diagnostic"]["n"] > 0


def test_selection_gradient_shapes():
    """默认非饱和开局 ⇒ non_sat 有样本；字段齐全且斜率/ρ 有限或 None。

    lifespan_mult=0.2 压短寿命 ⇒ 测试窗口内就能有个体育龄繁殖（fitness 有方差，
    否则 OLS 因 fitness 零方差而正确地返回 None——那也是合法仪器输出）。
    """
    e = _engine(measure=True, lifespan_mult=0.2)
    _run(e, 800)
    res = selection_gradient(e)
    assert set(res) == {"non_sat", "sat_diagnostic"}
    for sub in res.values():
        assert set(sub) >= {"n", "slope_g15", "spearman_rho", "mean_g15", "mean_fitness"}
        if sub["n"] >= 10 and sub["slope_g15"] is not None:
            assert -1e9 < sub["slope_g15"] < 1e9
            assert -1.0 <= sub["spearman_rho"] <= 1.0
    # 压短寿命 + 800 tick ⇒ 必然已有繁殖发生 ⇒ non_sat 的 fitness 方差非零 ⇒ 可估
    assert res["non_sat"]["n"] >= 10
    assert res["non_sat"]["slope_g15"] is not None
    assert int(e._rs_children.sum()) == e.total_born


def test_probe_config_default_off_and_dict_fallback():
    """C-6/C-7 惯例：默认关；旧存档缺 measure_signal_response 键时回退默认。"""
    assert InfoStructureConfig().measure_signal_response is False
    d = SimConfig(seed=1).to_dict()
    d["info_structure"].pop("measure_signal_response", None)
    c2 = SimConfig.from_dict(d)
    assert c2.info_structure.measure_signal_response is False


def test_snapshot_roundtrip_preserves_ledger():
    """⑤ 账本必须进快照：存→载后账本逐位一致，且可继续累计。"""
    e = _engine()
    _run(e, 120)
    path = Path(__file__).parent / "_tmp_d17_snap.npz"
    e.save_snapshot(str(path))
    e2 = SphereEngine.load_snapshot(str(path))
    assert np.array_equal(e2._rs_children, e._rs_children)
    assert np.array_equal(e2._rs_observed, e._rs_observed)
    assert np.allclose(e2._rs_g15, e._rs_g15)
    n_before = int(e2._rs_observed.sum())
    _run(e2, 30)
    assert int(e2._rs_observed.sum()) >= n_before
    path.unlink(missing_ok=True)
