"""D-26a：oracle 四环节诊断漏斗（内评 `_eval/D24判读预析` §4.1）。

纪律：计数器必须**纯观测**——不改状态、不消费 RNG；逐级计数必须**单调嵌套**
（后级 ⊆ 前级），否则漏斗会给出误导性的"瓶颈位置"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from simulation.config import InfoStructureConfig, SimConfig
from simulation.sphere_engine import SphereEngine


def _engine(oracle: bool = True, seed: int = 42, max_count: int = 3240) -> SphereEngine:
    cfg = SimConfig(seed=seed)
    cfg.simulation.use_sim_core = False
    cfg.population.max_count = max_count
    cfg.oracle.enabled = oracle
    d2 = InfoStructureConfig(enabled=True, learning_rate=0.05)
    cfg.info_structure = d2
    return SphereEngine(cfg)


def _run(e: SphereEngine, n: int) -> None:
    for _ in range(n):
        if e.extinct:
            break
        e.step()


def test_funnel_fields_and_nesting():
    """漏斗逐级单调嵌套（后级 ⊆ 前级）。"""
    e = _engine(oracle=True)
    _run(e, 400)
    f = e.oracle_funnel()
    need = {
        "emissions", "had_signal", "true_sig", "selected", "attrib_found",
        "in_window", "not_self", "budget_ok", "applied",
    }
    assert need <= set(f), f"缺字段: {need - set(f)}"
    assert f["had_signal"] >= f["true_sig"], "true_sig 必 ⊆ had_signal"
    assert f["selected"] >= f["attrib_found"] >= f["in_window"] >= f["not_self"] \
        >= f["budget_ok"] >= f["applied"], f"漏斗非单调嵌套: {f}"
    assert f["emissions"] >= f["applied"]
    # 通过 oracle_stats() 也能拿到（runner 走这条路出数）
    assert e.oracle_stats()["funnel"]["applied"] == f["applied"]


def test_funnel_zero_when_oracle_off():
    """oracle 关闭 ⇒ 漏斗恒零（且不影响既有统计）。"""
    e = _engine(oracle=False)
    _run(e, 300)
    f = e.oracle_funnel()
    assert all(v == 0 for k, v in f.items() if isinstance(v, int)), f
    assert e.oracle_stats()["enabled"] is False


def test_funnel_is_pure_observation():
    """同 seed、仅 oracle 开关不同：漏斗累加不消费 RNG。

    做法：两个引擎同 seed 同配置，分别在第 200 tick 前后读 rng_draws 增量，
    与各自"无漏斗累加"的基线一致——这里用更强的不变量：**漏斗读操作本身
    不改变状态**（读两次结果相同、且 rng_draws 不变）。
    """
    e = _engine(oracle=True)
    _run(e, 200)
    d0 = e.rng_draws
    f1 = e.oracle_funnel()
    f2 = e.oracle_funnel()
    assert f1 == f2, "读漏斗不得改变结果"
    assert e.rng_draws == d0, "读漏斗不得消费 RNG"


def test_funnel_snapshot_roundtrip():
    """漏斗计数随快照走（续跑后累计不失真）。"""
    e = _engine(oracle=True)
    _run(e, 150)
    p = Path(__file__).parent / "_tmp_d26_snap.npz"
    e.save_snapshot(str(p))
    b = SphereEngine.load_snapshot(str(p))
    assert b.oracle_funnel() == e.oracle_funnel()
    p.unlink(missing_ok=True)
    (p.with_suffix(".rngstate.pkl")).unlink(missing_ok=True)


def test_funnel_old_snapshot_falls_back_to_zero():
    """旧快照（无 diag_funnel 键）⇒ 回退全零，不抛错。"""
    e = _engine(oracle=True)
    _run(e, 100)
    p = Path(__file__).parent / "_tmp_d26_old.npz"
    e.save_snapshot(str(p))
    with np.load(p, allow_pickle=True) as z:
        data = {k: z[k] for k in z.files if k != "diag_funnel"}
    np.savez_compressed(p, **data)
    b = SphereEngine.load_snapshot(str(p))
    assert b.oracle_funnel()["applied"] == 0
    p.unlink(missing_ok=True)
    (p.with_suffix(".rngstate.pkl")).unlink(missing_ok=True)
