"""C1 信号发射下沉：函数级对拍测试（Python 参考 vs Rust signal_emit）。"""
import numpy as np
import pytest

import sim_core


def _python_reference(flat, energy, g15, rand_emit, densities,
                       resource_grid, resource_capacity, emit_cost, max_energy, duration):
    """Python 参考实现：复制 sphere_engine 步骤 4.5 精确逻辑。"""
    energy = energy.copy()
    signal_marks = np.zeros(len(densities), dtype=np.uint8)
    signal_age = np.zeros(len(densities), dtype=np.int32)

    emitters = np.flatnonzero(rand_emit < g15)
    if len(emitters):
        can_afford = energy[emitters] >= emit_cost
        emitters = emitters[can_afford]
        if len(emitters):
            energy[emitters] -= emit_cost
            e_flat = flat[emitters]
            e_bin = np.clip(
                (energy[emitters] / max(1e-9, max_energy) * 4).astype(np.int64), 0, 3
            )
            f_bit = (
                resource_grid[e_flat] > 0.5 * resource_capacity[e_flat]
            ).astype(np.int64)
            n_bit = (densities[e_flat] > 1).astype(np.int64)
            patterns = (e_bin * 4 + f_bit * 2 + n_bit).astype(np.uint8)
            signal_marks[e_flat] = patterns
            signal_age[e_flat] = np.where(patterns > 0, duration, 0).astype(np.int32)

    return energy, signal_marks, signal_age, len(emitters)


class TestSignalEmitBitwise:
    """函数级对拍：同输入，Python vs Rust 逐位相等。"""

    @pytest.fixture
    def setup(self):
        rng = np.random.default_rng(42)
        n = 100
        n_cells = 7200
        flat = rng.integers(0, n_cells, size=n).astype(np.int64)
        energy = rng.uniform(0, 300, size=n).astype(np.float64)
        g15 = rng.uniform(0, 1, size=n).astype(np.float64)
        rand_emit = rng.uniform(0, 1, size=n).astype(np.float64)
        densities = rng.uniform(0, 5, size=n_cells).astype(np.float64)
        resource_grid = rng.uniform(0, 40, size=n_cells).astype(np.float64)
        resource_capacity = rng.uniform(20, 50, size=n_cells).astype(np.float64)
        return {
            "flat": flat, "energy": energy, "g15": g15, "rand_emit": rand_emit,
            "densities": densities, "resource_grid": resource_grid,
            "resource_capacity": resource_capacity,
            "emit_cost": 0.1, "max_energy": 300.0, "duration": 50,
        }

    def test_energy_bitwise_equal(self, setup):
        py_energy, _, _, _ = _python_reference(**setup)
        rust_energy = setup["energy"].copy()
        rust_marks = np.zeros(len(setup["densities"]), dtype=np.uint8)
        rust_age = np.zeros(len(setup["densities"]), dtype=np.int32)
        sim_core.signal_emit(
            setup["flat"], rust_energy, setup["g15"], setup["rand_emit"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            rust_marks, rust_age,
            setup["emit_cost"], setup["max_energy"], setup["duration"],
        )
        np.testing.assert_array_equal(rust_energy, py_energy)

    def test_signal_marks_bitwise_equal(self, setup):
        _, py_marks, _, _ = _python_reference(**setup)
        rust_energy = setup["energy"].copy()
        rust_marks = np.zeros(len(setup["densities"]), dtype=np.uint8)
        rust_age = np.zeros(len(setup["densities"]), dtype=np.int32)
        sim_core.signal_emit(
            setup["flat"], rust_energy, setup["g15"], setup["rand_emit"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            rust_marks, rust_age,
            setup["emit_cost"], setup["max_energy"], setup["duration"],
        )
        np.testing.assert_array_equal(rust_marks, py_marks)

    def test_signal_age_bitwise_equal(self, setup):
        _, _, py_age, _ = _python_reference(**setup)
        rust_energy = setup["energy"].copy()
        rust_marks = np.zeros(len(setup["densities"]), dtype=np.uint8)
        rust_age = np.zeros(len(setup["densities"]), dtype=np.int32)
        sim_core.signal_emit(
            setup["flat"], rust_energy, setup["g15"], setup["rand_emit"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            rust_marks, rust_age,
            setup["emit_cost"], setup["max_energy"], setup["duration"],
        )
        np.testing.assert_array_equal(rust_age, py_age)

    def test_emit_count_equal(self, setup):
        _, _, _, py_count = _python_reference(**setup)
        rust_energy = setup["energy"].copy()
        rust_marks = np.zeros(len(setup["densities"]), dtype=np.uint8)
        rust_age = np.zeros(len(setup["densities"]), dtype=np.int32)
        rust_count = sim_core.signal_emit(
            setup["flat"], rust_energy, setup["g15"], setup["rand_emit"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            rust_marks, rust_age,
            setup["emit_cost"], setup["max_energy"], setup["duration"],
        )
        assert rust_count == py_count

    def test_no_emitters_when_g15_zero(self, setup):
        """g15 全 0 时无发射，energy 不变。"""
        g15_zero = np.zeros_like(setup["g15"])
        rust_energy = setup["energy"].copy()
        rust_marks = np.zeros(len(setup["densities"]), dtype=np.uint8)
        rust_age = np.zeros(len(setup["densities"]), dtype=np.int32)
        count = sim_core.signal_emit(
            setup["flat"], rust_energy, g15_zero, setup["rand_emit"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            rust_marks, rust_age,
            setup["emit_cost"], setup["max_energy"], setup["duration"],
        )
        assert count == 0
        np.testing.assert_array_equal(rust_energy, setup["energy"])

    def test_low_energy_no_emit(self, setup):
        """能量低于 emit_cost 时不发射。"""
        low_energy = np.full_like(setup["energy"], 0.05)  # < 0.1
        g15_high = np.ones_like(setup["g15"])
        rust_energy = low_energy.copy()
        rust_marks = np.zeros(len(setup["densities"]), dtype=np.uint8)
        rust_age = np.zeros(len(setup["densities"]), dtype=np.int32)
        count = sim_core.signal_emit(
            setup["flat"], rust_energy, g15_high, setup["rand_emit"],
            setup["densities"], setup["resource_grid"], setup["resource_capacity"],
            rust_marks, rust_age,
            setup["emit_cost"], setup["max_energy"], setup["duration"],
        )
        assert count == 0
        np.testing.assert_array_equal(rust_energy, low_energy)
