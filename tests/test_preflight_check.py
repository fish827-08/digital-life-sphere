"""Pre-Flight 检查工具（R61-C3/C4 + R78-3）的纯函数单测。

纪律（R47 精神）：**仪器自己也要被验证**——本工具的作用是在实验前拦截静默配置事故，
若它自己的比较/断言逻辑有错，就会变成"又一件坏仪器"。
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location(
    "preflight_check", ROOT / "experiments" / "preflight_check.py"
)
pf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pf)


def _rows(pairs):
    return [{"tick": str(t), **{k: str(v) for k, v in d.items()}} for t, d in pairs]


def test_compare_trajectories_identical_detected():
    """F-R9 症状：逐字段完全一致 ⇒ 必须被判定为"完全一致"（=失败信号）。"""
    a = _rows([(1000, {"N": 300, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 3}),
               (2000, {"N": 500, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 5})])
    b = [dict(r) for r in a]
    identical, first = pf.compare_trajectories(a, b)
    assert identical is True and first is None


def test_compare_trajectories_divergence_detected():
    a = _rows([(1000, {"N": 300, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 3}),
               (2000, {"N": 500, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 5})])
    b = _rows([(1000, {"N": 300, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 3}),
               (2000, {"N": 501, "g14": 0.5, "g15": 0.4, "trust": 0.5, "max_gen": 5})])
    identical, first = pf.compare_trajectories(a, b)
    assert identical is False and first == 2000


def test_compare_trajectories_length_mismatch_is_difference():
    a = _rows([(1000, {"N": 1, "g14": 0, "g15": 0, "trust": 0, "max_gen": 0})])
    b = _rows([(1000, {"N": 1, "g14": 0, "g15": 0, "trust": 0, "max_gen": 0}),
               (2000, {"N": 2, "g14": 0, "g15": 0, "trust": 0, "max_gen": 1})])
    identical, first = pf.compare_trajectories(a, b)
    assert identical is False and first is None


def test_seed_sensitivity_r78_3():
    """R78-3：std≡0 ⇒ 拒判；有差异 ⇒ 通过。"""
    ok, sd = pf.seed_sensitivity([0.42, 0.42, 0.42])
    assert ok is False and sd == 0.0
    ok2, sd2 = pf.seed_sensitivity([0.42, 0.51, 0.47])
    assert ok2 is True and sd2 > 0
    # 样本不足 ⇒ 不能断言敏感（保守：判失败）
    ok3, _ = pf.seed_sensitivity([0.42])
    assert ok3 is False


def test_readback_problems_catches_f_r9_symptom():
    """F-R9 症状：control 臂若读回 arbitrary_codebook=True ⇒ 必须报问题。"""
    bad = {"arbitrary_codebook": True, "neutral_genes": False,
           "signal_disabled": False, "oracle_enabled": False}
    probs = pf.readback_problems("control", bad)
    assert any("arbitrary_codebook" in p for p in probs)

    good = dict(bad, arbitrary_codebook=False)
    assert pf.readback_problems("control", good) == []
    # 未知臂名 ⇒ 报错而不是静默通过
    assert pf.readback_problems("nope", good) != []


def test_expected_switches_covers_runner_arm_choices():
    """防漂移守卫：runner 的 --arm choices 每新增一个臂，本表必须同步（否则读回核对静默失效）。"""
    src = (ROOT / "experiments" / "a4_verify_capacity.py").read_text(encoding="utf-8")
    m = re.search(r'--arm",\s*choices=\(([^)]*)\)', src)
    assert m, "未能解析 runner 的 --arm choices（若实现改了，请同步本测试）"
    choices = [c.strip().strip('"\'') for c in m.group(1).split(",") if c.strip()]
    assert set(choices) == set(pf.EXPECTED_SWITCHES), (
        f"runner 臂集 {sorted(choices)} 与 preflight 表 {sorted(pf.EXPECTED_SWITCHES)} 不一致"
    )
