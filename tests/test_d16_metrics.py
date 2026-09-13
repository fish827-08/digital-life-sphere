"""D-16：`codebook_conv` / `pred_frac` 入 CSV（判据列的单一口径实现）。

依据：R31③ / R38③。实现落在 `observatory/statistics.py`（规范函数，供所有 runner 复用）。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from observatory.statistics import (  # noqa: E402
    codebook_convergence, predation_fraction,
)


def test_codebook_convergence_identity_is_one():
    """恒等码本（人人相同且等于 state）⇒ 完全趋同 = 1.0。"""
    cb = np.tile(np.arange(16, dtype=np.uint8), (50, 1))
    assert abs(codebook_convergence(cb) - 1.0) < 1e-9


def test_codebook_convergence_random_is_below_one():
    """随机码本 ⇒ 明显小于 1（有判别力）。"""
    rng = np.random.default_rng(0)
    cb = rng.integers(0, 16, size=(200, 16)).astype(np.uint8)
    v = codebook_convergence(cb)
    assert 0.0 < v < 1.0


def test_codebook_convergence_empty_is_zero():
    assert codebook_convergence(np.zeros((0, 16), dtype=np.uint8)) == 0.0
    assert codebook_convergence(None) == 0.0


def test_predation_fraction_values():
    assert predation_fraction({"DeathCause.PREDATION": 10}) == 1.0
    assert predation_fraction({"DeathCause.STARVATION": 10}) == 0.0
    assert abs(predation_fraction({"DeathCause.PREDATION": 3,
                                   "DeathCause.STARVATION": 1}) - 0.75) < 1e-9


def test_predation_fraction_empty():
    assert predation_fraction({}) == 0.0
    assert predation_fraction(None) == 0.0
