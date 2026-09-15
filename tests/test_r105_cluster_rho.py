"""R105 / ρ v3 簇级合并的验收（内评 `_share/预注册-⑤主指标spearman_rho-v3-...md` §五 五例 + 两处回归）。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observatory.statistics import (  # noqa: E402
    fisher_z, inv_fisher_z, merge_rho_cluster, merge_rho_cluster_paired, t_quantile,
)

# ---------- ① t 临界值：与解析值一致 + **α 语义回归** ----------

def test_t_quantile_matches_analytic_and_alpha_semantics():
    """`t_{.95,df}` 与解析值相差 <1%；且 `alpha` = **单侧尾部概率** ⇒ 必为正。

    🔴 回归（我首版真 bug）：调用处误传 `1−alpha` ⇒ 临界值变**负** ⇒ 连"全负 ρ"都判 pass。
    """
    assert t_quantile(5) == pytest.approx(2.015048, rel=0.01)
    assert t_quantile(2) == pytest.approx(2.919986, rel=0.01)
    for df in (2, 3, 5, 10):
        assert t_quantile(df) > 0, "单侧 t 临界值必须为正（负值 = α 被重复应用）"
    assert t_quantile(5, 0.10) < t_quantile(5, 0.05)   # 尾部概率越大 ⇒ 临界值越小


# ---------- ② 簇级合并：符号正确 ----------

def test_merge_rho_cluster_signs():
    pos = merge_rho_cluster([0.02, 0.11, 0.007, 0.05, 0.08, 0.03])
    neg = merge_rho_cluster([0.02, -0.05, -0.11, 0.03, -0.08, -0.09])
    assert pos["pass"] is True and pos["ci_low_z"] > 0
    assert neg["pass"] is False and neg["ci_low_z"] < 0
    assert pos["n_seeds"] == 6
    assert "z 尺度" in pos["scale"] and "禁止代入" in pos["scale"]
    # 报告在 ρ 尺度：ci_low_rho = tanh(ci_low_z)
    assert pos["ci_low_rho"] == pytest.approx(float(np.tanh(pos["ci_low_z"])), abs=1e-6)


def test_fisher_roundtrip_and_clipping():
    for r in (-0.9, -0.2, 0.0, 0.3, 0.95):
        assert inv_fisher_z(fisher_z(r)) == pytest.approx(r, abs=1e-6)
    assert np.isfinite(fisher_z(1.0)) and np.isfinite(fisher_z(-1.0))   # 裁剪避免发散


# ---------- ③ 一个负 seed ⇒ 不必然失败（R2 相对 R1 的**关键差异**）----------

def test_one_negative_seed_does_not_necessarily_fail():
    """内评 §五③ **必须正面测**：R2 相对 R1（全一致为正）的差别就在这里。"""
    r = merge_rho_cluster([0.12, -0.03, 0.09, 0.11, -0.01, 0.08])
    assert r["pass"] is True, "一个负 seed 且整体强正 ⇒ R2 应通过（R1 会失败）"
    assert min(r["rho_by_seed"]) < 0
    # 反例：整体为负 ⇒ 必须失败
    assert merge_rho_cluster([0.12, -0.13, -0.19, -0.11, -0.21, -0.08])["pass"] is False


# ---------- ④ 配对簇级（同 seed 配对）----------

def test_paired_cluster_uses_matched_seeds():
    a = [0.20, 0.24, 0.19, 0.22, 0.21, 0.23]
    b = [0.01, 0.03, -0.02, 0.02, 0.00, 0.01]
    r = merge_rho_cluster_paired(a, b)
    assert r["n_seeds"] == 6 and r["pass"] is True and r["dbar"] > 0
    # 反序 ⇒ 方向反转、不通过
    assert merge_rho_cluster_paired(b, a)["pass"] is False
    # 配对不足 ⇒ 不判（不得因 n 小就"通过"）
    r1 = merge_rho_cluster_paired([0.2], [0.0])
    assert r1["pass"] is False and "不判" in r1["note"]


# ---------- ⑤ 个体级 n 不得进入公式（R96 防伪重复回归）----------

def test_individual_level_n_is_rejected():
    """内评 §五⑤：个体级 n（10³–10⁵）**禁止代入** —— 必须**报错**而非给出更"显著"的结果。"""
    with pytest.raises(ValueError) as ei:
        merge_rho_cluster([0.01] * 5000)
    assert "R96" in str(ei.value) and "个体级" in str(ei.value)
    with pytest.raises(ValueError):
        merge_rho_cluster_paired([0.01] * 1000, [0.0] * 1000)
    # 合法规模（≤64）不受影响
    assert merge_rho_cluster([0.05] * 32)["n_seeds"] == 32
