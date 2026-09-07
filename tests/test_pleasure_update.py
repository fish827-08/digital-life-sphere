"""C2 愉悦度更新下沉：函数级对拍测试（Python 参考 vs Rust pleasure_update）。"""
import numpy as np
import pytest

import sim_core


def _python_reference(flat, energy_now, energy_before, densities,
                       resource_grid, resource_capacity, signal_marks,
                       valence, arousal, expectation, baseline,
                       max_energy, alpha, valence_decay, arousal_decay,
                       baseline_rate, max_reward, w_energy, w_info, w_social):
    """Python 参考实现：复制 sphere_engine._update_pleasure 精确逻辑。

    注意：内部对所有就地修改的数组做 copy，不污染调用方传入的 fixture。
    """
    valence = valence.copy()
    arousal = arousal.copy()
    expectation = expectation.copy()  # (N, 120)
    baseline = baseline.copy()
    P = len(flat)
    idx = np.arange(P)
    max_e = max_energy

    # 1) 情境编码
    e_bin = np.clip((energy_now / max_e) * 5, 0, 4).astype(np.int64)
    food_ratio = np.clip(
        resource_grid[flat] / np.maximum(resource_capacity[flat], 1e-9),
        0.0, 1.0,
    )
    f_bin = np.clip((food_ratio * 4).astype(np.int64), 0, 3)
    n_count = densities[flat]
    n_bin = np.where(n_count == 0, 0, np.where(n_count <= 2, 1, 2))
    s_bin = np.zeros(P, dtype=np.int64)
    context = e_bin * 24 + f_bin * 6 + n_bin * 2 + s_bin

    # 2) 事件收益
    delta_e = np.clip((energy_now - energy_before) / max_e, -1.0, 1.0)
    social = np.where(n_count > 0, 0.2, -0.1)
    info = np.where(signal_marks[flat] > 0, 0.5, 0.0)
    reward = w_energy * delta_e + w_info * info + w_social * social

    # 3) RPE
    expected = expectation[idx, context]
    rpe = reward - expected

    # 4) 更新
    valence = valence + rpe * 0.3
    valence = valence * valence_decay
    valence = np.clip(valence, -1.0, 1.0)

    arousal = arousal + np.abs(rpe) * 0.2
    arousal = arousal * arousal_decay
    arousal = np.clip(arousal, 0.0, 1.0)

    expectation[idx, context] = expectation[idx, context] + alpha * rpe
    expectation = np.clip(expectation, 0.0, max_reward)

    baseline = baseline * (1.0 - baseline_rate) + valence * baseline_rate

    return valence, arousal, expectation, baseline


class TestPleasureBitwise:
    """函数级对拍：同输入，Python vs Rust 逐位相等。"""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)
        n = 100
        n_cells = 7200
        flat = rng.integers(0, n_cells, size=n).astype(np.int64)
        energy_now = rng.uniform(0, 300, size=n).astype(np.float64)
        energy_before = rng.uniform(0, 300, size=n).astype(np.float64)
        densities = rng.uniform(0, 5, size=n_cells).astype(np.float64)
        resource_grid = rng.uniform(0, 40, size=n_cells).astype(np.float64)
        resource_capacity = rng.uniform(20, 50, size=n_cells).astype(np.float64)
        signal_marks = rng.integers(0, 16, size=n_cells).astype(np.uint8)
        valence = rng.uniform(-1, 1, size=n).astype(np.float64)
        arousal = rng.uniform(0, 1, size=n).astype(np.float64)
        expectation = rng.uniform(0, 2, size=(n, 120)).astype(np.float64)
        baseline = rng.uniform(-1, 1, size=n).astype(np.float64)
        return {
            "flat": flat, "energy_now": energy_now, "energy_before": energy_before,
            "densities": densities, "resource_grid": resource_grid,
            "resource_capacity": resource_capacity, "signal_marks": signal_marks,
            "valence": valence, "arousal": arousal, "expectation": expectation,
            "baseline": baseline,
            "max_energy": 300.0, "alpha": 0.05, "valence_decay": 0.95,
            "arousal_decay": 0.97, "baseline_rate": 0.001, "max_reward": 2.0,
            "w_energy": 0.5, "w_info": 0.3, "w_social": 0.2,
        }

    def test_valence_bitwise_equal(self, setup):
        py_v, _, _, _ = _python_reference(**setup)
        rust_v = setup["valence"].copy()
        rust_a = setup["arousal"].copy()
        rust_exp = setup["expectation"].copy().reshape(-1)
        rust_bl = setup["baseline"].copy()
        sim_core.pleasure_update(
            setup["flat"], setup["energy_now"], setup["energy_before"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            setup["signal_marks"],
            rust_v, rust_a, rust_exp, rust_bl,
            setup["max_energy"], setup["alpha"], setup["valence_decay"],
            setup["arousal_decay"], setup["baseline_rate"], setup["max_reward"],
            setup["w_energy"], setup["w_info"], setup["w_social"],
        )
        np.testing.assert_array_equal(rust_v, py_v)

    def test_arousal_bitwise_equal(self, setup):
        _, py_a, _, _ = _python_reference(**setup)
        rust_v = setup["valence"].copy()
        rust_a = setup["arousal"].copy()
        rust_exp = setup["expectation"].copy().reshape(-1)
        rust_bl = setup["baseline"].copy()
        sim_core.pleasure_update(
            setup["flat"], setup["energy_now"], setup["energy_before"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            setup["signal_marks"],
            rust_v, rust_a, rust_exp, rust_bl,
            setup["max_energy"], setup["alpha"], setup["valence_decay"],
            setup["arousal_decay"], setup["baseline_rate"], setup["max_reward"],
            setup["w_energy"], setup["w_info"], setup["w_social"],
        )
        np.testing.assert_array_equal(rust_a, py_a)

    def test_expectation_bitwise_equal(self, setup):
        _, _, py_exp, _ = _python_reference(**setup)
        rust_v = setup["valence"].copy()
        rust_a = setup["arousal"].copy()
        rust_exp = setup["expectation"].copy().reshape(-1)
        rust_bl = setup["baseline"].copy()
        sim_core.pleasure_update(
            setup["flat"], setup["energy_now"], setup["energy_before"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            setup["signal_marks"],
            rust_v, rust_a, rust_exp, rust_bl,
            setup["max_energy"], setup["alpha"], setup["valence_decay"],
            setup["arousal_decay"], setup["baseline_rate"], setup["max_reward"],
            setup["w_energy"], setup["w_info"], setup["w_social"],
        )
        np.testing.assert_array_equal(rust_exp.reshape(-1, 120), py_exp)

    def test_baseline_bitwise_equal(self, setup):
        _, _, _, py_bl = _python_reference(**setup)
        rust_v = setup["valence"].copy()
        rust_a = setup["arousal"].copy()
        rust_exp = setup["expectation"].copy().reshape(-1)
        rust_bl = setup["baseline"].copy()
        sim_core.pleasure_update(
            setup["flat"], setup["energy_now"], setup["energy_before"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            setup["signal_marks"],
            rust_v, rust_a, rust_exp, rust_bl,
            setup["max_energy"], setup["alpha"], setup["valence_decay"],
            setup["arousal_decay"], setup["baseline_rate"], setup["max_reward"],
            setup["w_energy"], setup["w_info"], setup["w_social"],
        )
        np.testing.assert_array_equal(rust_bl, py_bl)

    def test_context_encoding_coverage(self):
        """120 情境编码全覆盖：构造边界输入验证所有 context 值。"""
        rng = np.random.default_rng(99)
        n = 500
        n_cells = 7200
        # 构造覆盖所有能量档/食物档/邻居档的输入
        flat = rng.integers(0, n_cells, size=n).astype(np.int64)
        energy_now = np.linspace(0, 300, n).astype(np.float64)
        energy_before = np.linspace(0, 300, n).astype(np.float64)
        densities = np.zeros(n_cells, dtype=np.float64)
        densities[flat] = rng.choice([0, 1, 2, 3, 5], size=n)
        resource_grid = np.linspace(0, 50, n_cells).astype(np.float64)
        resource_capacity = np.full(n_cells, 40.0, dtype=np.float64)
        signal_marks = rng.integers(0, 2, size=n_cells).astype(np.uint8)
        valence = np.zeros(n, dtype=np.float64)
        arousal = np.full(n, 0.5, dtype=np.float64)
        expectation = np.zeros((n, 120), dtype=np.float64)
        baseline = np.zeros(n, dtype=np.float64)

        setup = {
            "flat": flat, "energy_now": energy_now, "energy_before": energy_before,
            "densities": densities, "resource_grid": resource_grid,
            "resource_capacity": resource_capacity, "signal_marks": signal_marks,
            "valence": valence, "arousal": arousal, "expectation": expectation,
            "baseline": baseline,
            "max_energy": 300.0, "alpha": 0.05, "valence_decay": 0.95,
            "arousal_decay": 0.97, "baseline_rate": 0.001, "max_reward": 2.0,
            "w_energy": 0.5, "w_info": 0.3, "w_social": 0.2,
        }
        py_v, py_a, py_exp, py_bl = _python_reference(**setup)
        rust_v = valence.copy()
        rust_a = arousal.copy()
        rust_exp = expectation.copy().reshape(-1)
        rust_bl = baseline.copy()
        sim_core.pleasure_update(
            flat, energy_now, energy_before, densities, resource_grid,
            resource_capacity, signal_marks,
            rust_v, rust_a, rust_exp, rust_bl,
            300.0, 0.05, 0.95, 0.97, 0.001, 2.0, 0.5, 0.3, 0.2,
        )
        np.testing.assert_array_equal(rust_v, py_v)
        np.testing.assert_array_equal(rust_a, py_a)
        np.testing.assert_array_equal(rust_exp.reshape(-1, 120), py_exp)
        np.testing.assert_array_equal(rust_bl, py_bl)
