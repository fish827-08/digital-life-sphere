"""R44 功效预跑工具的回归测试（纯函数 + 一处**我自己犯过的**臂标签 bug）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments import r44_power_prerun as r44  # noqa: E402


def test_between_seed_sd_is_sample_sd():
    assert r44.between_seed_sd([1.0, 1.0, 1.0]) == 0.0
    assert r44.between_seed_sd([0.0, 2.0]) == pytest.approx(np.sqrt(2.0))
    assert np.isnan(r44.between_seed_sd([1.0]))          # 单点无法估方差 ⇒ NaN（不得当 0）


def test_all_positive_rule_is_anti_power():
    """🔴 R44 核心发现：『各 seed 全为正』功效 = p+^n ⇒ **单调递减**。

    若哪天有人把它改成"随 n 上升"，本测试会失败——这正是要守住的结论。
    """
    sd, mu = 0.0769, 0.10
    p3 = r44.power_all_positive(3, mu, sd)
    p6 = r44.power_all_positive(6, mu, sd)
    p12 = r44.power_all_positive(12, mu, sd)
    assert p3 > p6 > p12
    assert p3 < 0.80, "n=3 时该规则也不到 0.8 ⇒ 它对任何 n 都不可达 0.8"


def test_ci_rule_power_increases_with_n():
    sd, mu = 0.0769, 0.10
    assert r44.power_ci_zero(4, mu, sd) > r44.power_ci_zero(3, mu, sd)
    assert r44.power_ci_zero(10, mu, sd) > 0.95


def test_floor_rule_power_is_half_when_alt_equals_floor():
    """『合并 ρ ≥ floor』在真实效应恰为 floor 时功效恒 ≈ 0.5（与 n 无关）——该口径的固有性质。"""
    assert r44.power_floor(5, 0.10, 0.0769, floor=0.10) == pytest.approx(0.5, abs=0.01)
    assert r44.power_floor(20, 0.10, 0.0769, floor=0.10) == pytest.approx(0.5, abs=0.01)


def test_load_runs_separates_oracle_by_donation(tmp_path):
    """🔴 回归（我首跑踩到的真 bug）：不同 `donation` 的 oracle 跑**不得并入同一臂**。

    否则跨剂量差异会被当成跨 seed 方差 ⇒ 方差被虚增、σ 失真 ⇒ seed 数算错。
    """
    for don, rho in ((0.674, -0.03), (1.5, 0.10)):
        (tmp_path / f"oracle_{don}.summary.json").write_text(json.dumps({
            "switches": {"arm": "oracle", "seed": 42, "oracle_donation": don},
            "result": {"selection_gradient": {"non_sat": {"spearman_rho": rho, "n": 100}},
                       "signal_response": {"resp_triple": 0.01}},
        }), encoding="utf-8")
    runs = r44.load_runs([tmp_path])
    assert "oracle@0.674" in runs and "oracle@1.5" in runs
    assert "oracle" not in runs, "不得存在不区分剂量的裸 oracle 臂"


def test_load_runs_without_donation_keeps_plain_arm(tmp_path):
    (tmp_path / "zero_s42.summary.json").write_text(json.dumps({
        "switches": {"arm": "zero", "seed": 42},
        "result": {"selection_gradient": {"non_sat": {"spearman_rho": 0.01, "n": 50}},
                   "signal_response": {}},
    }), encoding="utf-8")
    runs = r44.load_runs([tmp_path])
    assert list(runs) == ["zero"]
